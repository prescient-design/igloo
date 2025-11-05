from torch.utils.data import DataLoader
import json
from tqdm import tqdm
import torch
import argparse
import pandas as pd

from igloo.vqvae import VQVAE
from dataset import LoopSequenceOnlyDataset, LoopSequenceDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Run IgLoo model on loop sequences.")
    parser.add_argument('--model_config', type=str, required=True, help='Path to the model configuration file.')
    parser.add_argument('--model_ckpt', type=str, required=True, help='Path to the model checkpoint file.')
    parser.add_argument('--loop_dataset_path', type=str, required=True, help='Path to the loop dataset file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the output embeddings.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model on.')
    parser.add_argument('--save_as_parquet', action='store_true', help='If set, save the output as a parquet file instead of JSON lines.')
    parser.add_argument('--use_sequence_only', action='store_true', help='If set, use sequence only without dihedrals.')
    parser.add_argument('--use_sequence_and_dihedrals', action='store_true', help='If set, use sequence and dihedrals.')
    return parser.parse_args()


def igloo_sequence_only(
    model_config: str,
    model_ckpt: str,
    loop_dataset_path: str,
    output_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_as_parquet: bool = False,
):
    model = VQVAE.load_from_config_and_weights(model_config, model_ckpt, strict=False)
    dataset = LoopSequenceOnlyDataset(loop_dataset_path)
    dataset.inference = True
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=dataset.collate_fn)

    model.eval()
    model.to(device)

    output = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        encoded_cls, quantized, quantized_indices = model.inference_sequence_only(batch)
        for i, item_id in enumerate(batch['id']):
            output.append({
                'loop_id': item_id,
                'encoded': encoded_cls[i].tolist(),
                'quantized': quantized[i].tolist(),
                'quantized_indices': quantized_indices[i].tolist(),
                'loop_length': batch['sequence_length'][i].item(),
            })
    
    if output_path.endswith(".parquet") or save_as_parquet:
        df = pd.DataFrame(output)
        df.to_parquet(output_path, index=False)
    else:
        with open(output_path, 'w') as f:
            for item in output:
                f.write(json.dumps(item) + '\n')
    print(f"Output saved to {output_path}")

def igloo_sequence_and_dihedrals(
    model_config: str,
    model_ckpt: str,
    loop_dataset_path: str,
    output_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_as_parquet: bool = False,
):
    model = VQVAE.load_from_config_and_weights(model_config, model_ckpt, strict=False)
    dataset = LoopSequenceDataset(loop_dataset_path)
    dataset.inference = True # VERY IMPORTANT: so no masking is applied
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=dataset.collate_fn)

    model.eval()
    model.to(device)

    output = []
    for batch in tqdm(dataloader, total=len(dataloader)):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        model_output = model.forward(batch)

        for i, item_id in enumerate(batch['id']):
            output.append({
                'loop_id': item_id,
                'encoded': model_output.encoded[i].tolist(),
                'quantized': model_output.quantized[i].tolist(),
                'quantized_indices': model_output.quantized_indices[i].tolist(),
                'loop_length': batch['sequence_length'][i].item(),
            })

    if output_path.endswith(".parquet") or save_as_parquet:
        df = pd.DataFrame(output)
        df.to_parquet(output_path, index=False)
    else:
        with open(output_path, 'w') as f:
            for item in output:
                f.write(json.dumps(item) + '\n')
    print(f"Output saved to {output_path}")


if __name__ == "__main__":
    args = parse_args()

    if args.use_sequence_only or args.use_sequence_and_dihedrals:
        if args.use_sequence_only and args.use_sequence_and_dihedrals:
            raise ValueError("Cannot use both --use_sequence_only and --use_sequence_and_dihedrals at the same time.")
        use_sequence_and_dihedrals = args.use_sequence_and_dihedrals
    else:  
        # auto detect if sequence and dihedrals are present
        if args.loop_dataset_path.endswith('.jsonl'):
            with open(args.loop_dataset_path, 'r') as f:
                loop_item = json.loads(f.readline())
                use_sequence_and_dihedrals = 'phi' in loop_item and 'psi' in loop_item and 'omega' in loop_item
        elif args.loop_dataset_path.endswith('.parquet'):
            df = pd.read_parquet(args.loop_dataset_path)
            use_sequence_and_dihedrals = 'phi' in df.columns and 'psi' in df.columns and 'omega' in df.columns
        elif args.loop_dataset_path.endswith('.csv'):
            df = pd.read_csv(args.loop_dataset_path)
            use_sequence_and_dihedrals = 'phi' in df.columns and 'psi' in df.columns and 'omega' in df.columns
        else:
            raise ValueError("Unsupported loop dataset format. Supported formats are .jsonl, .parquet, and .csv.")

    if use_sequence_and_dihedrals:
        print("Running Igloo with sequence and dihedrals")
        igloo_sequence_and_dihedrals(
            model_config=args.model_config,
            model_ckpt=args.model_ckpt,
            loop_dataset_path=args.loop_dataset_path,
            output_path=args.output_path,
            device=args.device,
            save_as_parquet=args.save_as_parquet,
        )
    else:
        print("Running Igloo with sequence only")
        igloo_sequence_only(
            model_config=args.model_config,
            model_ckpt=args.model_ckpt,
            loop_dataset_path=args.loop_dataset_path,
            output_path=args.output_path,
            device=args.device,
            save_as_parquet=args.save_as_parquet,
        )