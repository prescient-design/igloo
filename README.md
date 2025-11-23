# Igloo: Tokenizing Loops of Antibodies

<img src="assets/igloo_logo.png" alt="Igloo" width="300"/>

[Preprint](https://arxiv.org/abs/2509.08707)

Authors
* Ada Fang
* Rob Alberstein
* Simon Kelow
* Frédéric Dreyer

## :seedling: Getting started

Clone the repo: `git clone https://github.com/prescient-design/ibex.git`

Create an environment and install igloo locally
```
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

Alternatively, you can install the `prescient-igloo` package
```
pip install prescient-igloo
```
which provides the `model/` directory as `import igloo` and the `finetune_igbert/` directory as `import igloo.plm`.

## :rocket: Run Igloo
### For loops with sequences and structures ( :star2: recommended)
If structures are available use this approach

**1. Prepare input to Igloo**
Prepare a CSV file, see `example/sample_igloo_sequences.csv` containing sequences of heavy and light chains. Required columns:
* `fv_heavy_aho` and `fv_light_aho` sequences of aho aligned heavy and light chains. For aho alignments of sequences please refer to [ANARCI](https://github.com/oxpig/ANARCI).
* `id` unique identifier for each antibody that should correspond to the file name `<id>.pdb` in the `structure_dir`.

```
python process_data/process_dihedrals.py \
    --id_key "id" --aho_light_key "fv_light_aho" --aho_heavy_key "fv_heavy_aho" \
    --structure_dir my_pdbs/ \
    --df_path example/sample_igloo_sequences.csv \
    --parquet_output_path example/sample_igloo_input.parquet
```
The output file will have loops with `loop_id`, where it is the sequence id with `_{loop_type}` as a suffix and `loop_type` is one of `[H1, H2, H3, H3, L1, L2, L3, L4]`.

Alternatively, you can write your own processing script to output something like the example `example/sample_igloo_input.parquet`.

**2. Igloo Inference**
```
/homefs/home/fanga5/micromamba/envs/pyenv/bin/python run_igloo.py \
    --model_ckpt checkpoints/igloo_weights.pt \
    --model_config checkpoints/igloo_config.json \
    --loop_dataset_path example/sample_igloo_input.parquet \
    --output_path example/sample_igloo_output.parquet
```

### For loops with sequences and predicted structures  ( :star: recommended)
Igloo can be used for library design by:
* Finding sequences close to the seed which are in the same Igloo cluster to the seed
* Downsampling a large library by maximizing coverage over the Igloo clusters

**1. Prepare your sequences**

CSV file, see `example/sample_igloo_sequences.csv`, containing sequences of heavy and light chains. Required columns:
* `fv_heavy` and `fv_light` sequences of heavy and light chains.
* `fv_heavy_aho` and `fv_light_aho` sequences of aho aligned heavy and light chains. For aho alignments of sequences please refer to [ANARCI](https://github.com/oxpig/ANARCI).
* `id` unique identifier for each antibody chain sequence, can be just a unique number for each sequence.

**2. Run structure prediction**

Igloo can tokenize loops with sequence only, but performs better if it has structures of the antibodies. Generate structures with a structure predictor, e.g. Ibex which is provided in the Prescient repo.
```
pip install prescient-ibex
ibex --csv example/sample_igloo_sequences.csv --output ibex_predictions_dir/
```

**3. Prepare input to Igloo**

```
python process_data/process_dihedrals.py \
    --id_key "id" --aho_light_key "fv_light_aho" --aho_heavy_key "fv_heavy_aho" \
    --structure_dir ibex_predictions_dir/ \
    --df_path example/sample_igloo_sequences.csv \
    --parquet_output_path example/sample_igloo_input.parquet
```
The output file will have loops with `loop_id`, where it is the sequence id with `_{loop_type}` as a suffix and `loop_type` is one of `[H1, H2, H3, H3, L1, L2, L3, L4]`.

**4. Igloo Inference**
```
python run_igloo.py \
    --model_ckpt checkpoints/igloo_weights.pt \
    --model_config checkpoints/igloo_config.json \
    --loop_dataset_path example/sample_igloo_input.parquet \
    --output_path example/sample_igloo_output.parquet
```

### For loops with only sequences and *without* predicted structures
This may be preferable if there are many sequences (i.e. millions) and running structure prediction on all of the sequences is too compute intensive.

To run Igloo with sequence only, prepare a CSV file with the columns:
* `loop_id`: Unique identifier for each loop
* `loop_sequence`: One letter amino acid sequence for loop

An example is provided at `example/sample_igloo_input_sequence_only.csv`.
```
python run_igloo.py \
    --model_ckpt checkpoints/igloo_weights.pt \
    --model_config checkpoints/igloo_config.json \
    --loop_dataset_path example/sample_igloo_input_sequence_only.csv \
    --output_path example/sample_igloo_out_sequence_only.parquet
```

### Igloo output
The output is a parquet file with the following columns:
* `loop_id`
* `encoded`: Continuous Igloo representation
* `quantized`: Discrete Igloo representation, this is the `encoded` representation after it is passed through the Vector Quantize layer
* `quantized_indices`: An integer indicating which discrete Igloo token

## :snowflake: Training Igloo
Igloo was first trained on SAbDab and Ibex-predicted pOAS structures. Then finetuned on just SAbDab.
```
python train.py \
    --train_data_path poas_sabdab_train.jsonl \
    --val_data_path sabdab_val.jsonl \
    --batch_size 64 \
    --learning_rate 1e-3 \
    --project_name "Phase 1: train on sabdab+pOAS" \
    --device cuda \
    --num_epochs 100 \
    --codebook_size 8192 \
    --num_encoder_layers 4 \
    --commit_loss_weight 0.5 \
    --save_dir Igloo_models \
    --embedding_dim 128 \
    --unit_circle_transform_weight 0.01 \
    --loop_length_tolerance 0 \
    --dihedral_loss \
    --learnable_codebook \
    --use_wandb

python train.py \
    --train_data_path sabdab_train.jsonl \
    --val_data_path sabdab_val.jsonl \
    --batch_size 64 \
    --learning_rate 5e-5 \
    --project_name "Phase 2: finetune on sabdab only" \
    --device cuda \
    --num_epochs 100 \
    --codebook_size 8192 \
    --num_encoder_layers 4 \
    --commit_loss_weight 0.5 \
    --save_dir Igloo_models \
    --embedding_dim 128 \
    --unit_circle_transform_weight 0.01 \
    --loop_length_tolerance 0 \
    --dihedral_loss \
    --learnable_codebook \
    --codebook_learning_rate 1e-3 \
    --weight_decay 1e-5 \
    --pretrained_model_weights Igloo_models/best_checkpoint_from_above.pt \
    --pretrained_model_config Igloo_models/model_config_from_above.json \
    --use_wandb
```

## :robot: IglooLM and IglooALM
Please refer to [finetune_igbert/README.md](https://github.com/prescient-design/igloo/blob/main/finetune_igbert/README.md).

## :bulb: Tutorials and reproducing paper analyses

### Recovery of canonical clusters
See: `paper_analyses/1_recovery_of_canonical_clusters/recovery_of_canonical_clusters.ipynb`

We show how well Igloo can recovery the canonical clusters ([North et al. 2011](https://pubmed.ncbi.nlm.nih.gov/21035459/), [Kelow et al. 2022](https://www.biorxiv.org/content/10.1101/2022.10.12.511988)) across SAbDab with dihedral distance cutoffs of 0.1 and 0.47. The results can be seen at the bottom of the jupyter notebook.

### Retrieval of similar structured loops with Igloo tokens
See: `paper_analyses/2_retrieve_similar_loops/sabdab_test_set.ipynb`

We show how to use Igloo embeddings to retrieve similar structured loops. Baselines can be run at `paper_analyses/0_baselines`.

### Predicting binding affinity with IglooLM embeddings on AbBiBench
See: `paper_analyses/3_abbibench/run_abbibench.py`. Baselines can be run at `paper_analyses/0_baselines`. 

### Sampling structurally-consistent loop sequences with IglooALM
See: `paper_analyses/4_sampled_cdrs/analyse_sampled_cdrs.ipynb`.

## Citation
```bibtex
@misc{fang2025tokenizingloopsantibodies,
      title={Tokenizing Loops of Antibodies}, 
      author={Ada Fang and Robert G. Alberstein and Simon Kelow and Frédéric A. Dreyer},
      year={2025},
      eprint={2509.08707},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2509.08707}, 
}
```

## :hammer_and_wrench: Support
Please feel free to contact Ada Fang ([ada_fang@g.harvard.edu](mailto:ada_fang@g.harvard.edu)) of Frédéric Dreyer ([dreyer.frederic@gene.com](mailto:dreyer.frederic@gene.com)) for any questions or help with running Igloo.
