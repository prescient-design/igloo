#!/usr/bin/env python
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for masked language modeling (BERT, ALBERT, RoBERTa...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=fill-mask
"""
# You can also adapt this script on your own mlm task. Pointers for this are left as comments.

import argparse
import json
import logging
import os
import random
import numpy as np
import pandas as pd

import datasets
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from glob import glob
from collections import defaultdict

import transformers
from transformers import (
    AutoConfig,
    BertModel,
    AutoModelForMaskedLM,
    AutoTokenizer,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from transformers import DataCollatorForLanguageModeling

from .embedding_model import BertLoopModelForMaskedLM
from .constants import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.53.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--streaming", action="store_true", help="Whether to use streaming mode for the datasets library."
    )
    parser.add_argument(
        "--load_in_memory", action="store_true", help="Whether to load the dataset in memory."
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--use_loop_tokens",
        action="store_true",
        help="If passed the model is trained with loop token embeddings",
    )
    parser.add_argument(
        "--use_loop_tokens_whole_sequence",
        action="store_true",
        help="If passed the model is trained with loop token embeddings",
    )
    parser.add_argument(
        "--use_quantized_loop_tokens",
        action="store_true",
        help="If passed the model is trained with quantized loop token embeddings",
    )
    parser.add_argument(
        "--use_special_cdr_tokens",
        action="store_true",
        help="If passed, the model will use special CDR tokens to mark the start and end of CDR sections.",
    )
    parser.add_argument(
        "--loop_token_model_weights",
        type=str,
        default=None,
        help="Path to pretrained model weights for loop tokenizer"
    )
    parser.add_argument(
        "--loop_token_model_config",
        type=str,
        default=None,
        help="Path to pretrained model config for loop tokenizer"
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument("--output_file", type=str, default=None, help="Where to store the final sequences.")
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached evaluation sets"
    )
    parser.add_argument(
        "--sample_section", 
        choices=['CDR1', 'CDR2', 'CDR3', 'CDR4'], 
        default='CDR3',
        help=(
            "The model will sample only the specified CDR section."
        ),
    )
    parser.add_argument(
        "--sampling_temperature",
        type=float,
        default=1.0,
        help="The temperature to use for sampling."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate for each input sequence."
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    args = parser.parse_args()

    # Sanity checks
    extension = args.validation_file.split(".")[-1]
    if extension not in ["csv", "json", "txt", "parquet"]:
        raise ValueError("`validation_file` should be a csv, json, txt, or parquet file.")

    if args.use_loop_tokens or args.use_quantized_loop_tokens or args.use_loop_tokens_whole_sequence:
        assert args.use_special_cdr_tokens, "Using loop tokens requires using special CDR tokens."

    if args.load_in_memory and args.streaming:
        raise ValueError("Cannot use `load_in_memory` and `streaming` at the same time. Please set one of them to False.")

    return args


def setup_accelerator(args):
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    accelerator = Accelerator(gradient_accumulation_steps=1, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    accelerator.wait_for_everyone()
    return accelerator


def setup_datasets(args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    data_files = {}
    data_files["validation"] = args.validation_file
    extension = args.validation_file.split(".")[-1]
    
    if args.load_in_memory:
        raw_datasets = datasets.DatasetDict(
            {
                "validation": datasets.Dataset.from_pandas(pd.read_parquet(args.validation_file)),
            }
        )
    else:
        raw_datasets = load_dataset(extension, data_files=data_files, streaming=args.streaming)
    
    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=not args.use_slow_tokenizer, trust_remote_code=args.trust_remote_code
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script. "
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return raw_datasets, tokenizer


def tokenize_datasets(raw_datasets, tokenizer, accelerator, args):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["validation"].column_names

    if args.max_seq_length is None:
        max_seq_length = tokenizer.model_max_length
        if max_seq_length > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            max_seq_length = 1024
    else:
        if args.max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({args.max_seq_length}) is larger than the maximum length for the "
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        max_seq_length = min(args.max_seq_length, tokenizer.model_max_length)

    # When using line_by_line, we just tokenize each nonempty line.
    padding = "max_length" if args.pad_to_max_length else False

    def process_one_example(examples, idx):
        sequence = []
        special_tokens_mask = [1]
        for section_i, section in enumerate(SECTIONS):
            section_seq = list(examples[section][idx])
            if "FW" in section:
                sequence.extend(section_seq)
                special_tokens_mask.extend([1]*len(section_seq))
            else:
                if section == args.sample_section:
                    if args.use_special_cdr_tokens:
                        sequence.extend([SPECIAL_TOKENS[0]] + section_seq + [SPECIAL_TOKENS[1]])
                        special_tokens_mask.extend([1] + [0]*len(section_seq) + [1])
                    else:
                        sequence.extend(section_seq)
                        special_tokens_mask.extend([0]*len(section_seq))
                else:
                    # mark all tokens as special so that it does not get masked
                    if args.use_special_cdr_tokens:
                        sequence.extend([SPECIAL_TOKENS[0]] + section_seq + [SPECIAL_TOKENS[1]])
                        special_tokens_mask.extend([1] + [1]*len(section_seq) + [1])
                    else:
                        sequence.extend(section_seq)
                        special_tokens_mask.extend([1]*len(section_seq))
        special_tokens_mask.append(1)
        if padding == "max_length":
            special_tokens_mask.extend([1] * (max_seq_length - len(special_tokens_mask))) # Pad special tokens mask
        return " ".join(sequence), special_tokens_mask

    def tokenize_function(examples):
        # Remove empty lines
        collated_examples = [
            process_one_example(examples, idx) for idx in range(len(examples[column_names[0]]))
        ]
        sequences = [collated_example[0] for collated_example in collated_examples]
        special_tokens_masks = [collated_example[1] for collated_example in collated_examples]
        tokenized = tokenizer(
            sequences,
            padding=padding,
            truncation=True,
            max_length=max_seq_length,
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=False,
        )
        tokenized['special_tokens_mask'] = special_tokens_masks
        # to evaluate masking mask out the whole CDR section
        return tokenized

    with accelerator.main_process_first():
        if args.use_loop_tokens or args.use_quantized_loop_tokens or args.use_loop_tokens_whole_sequence:
            remove_columns = [x for x in column_names if x != "angles"]
        else:
            remove_columns = column_names
        if args.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=remove_columns,
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                remove_columns=remove_columns,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    return tokenized_datasets


def sampling_inference(model, eval_dataloader, tokenizer, temperature, num_samples=1):
    perplexities = []
    sampled_sequences = defaultdict(list)
    model.eval()
    for step, batch in tqdm(enumerate(eval_dataloader), desc="Running sampling", total=len(eval_dataloader)):
        with torch.no_grad():
            outputs = model(**batch)
            masked_token_indices = torch.where(batch["labels"] != -100)
            masked_logits = outputs.logits[masked_token_indices]
            masked_logits = masked_logits / temperature
            masked_prob = torch.softmax(masked_logits, dim=-1)
            sampled_token_ids = torch.multinomial(masked_prob, num_samples=num_samples, replacement=True)
            
            target_ids = batch["labels"][masked_token_indices]
            log_probs = torch.log(masked_prob + 1e-12)
            token_log_probs = log_probs[torch.arange(log_probs.size(0)), target_ids]
            start = 0
            for i in range(batch["labels"].shape[0]):
                num_masked = (batch["labels"][i] != -100).sum().item()
                mean_nll = -token_log_probs[start:start + num_masked].mean()
                perplexity = torch.exp(mean_nll)
                perplexities.append(perplexity.item())
                start += num_masked
            
            for i in range(num_samples):
                sampled_input_ids = batch["input_ids"].clone()
                sampled_input_ids[masked_token_indices] = sampled_token_ids[:, i]
                sequences = tokenizer.batch_decode(sampled_input_ids, skip_special_tokens=True)
                sampled_sequences[i].extend(sequences)
    return sampled_sequences, perplexities


def get_save_dir(save_dir):
    curr_versions = glob(f"{save_dir}/version_*", recursive=False)
    if not curr_versions:
        save_dir = f"{save_dir}/version_1"
    else:
        latest_version = max(curr_versions, key=lambda x: int(x.split('_')[-1]))
        latest_version_num = int(latest_version.split('_')[-1])
        save_dir = f"{save_dir}/version_{latest_version_num + 1}"
    os.makedirs(save_dir, exist_ok=True)
    return save_dir
    

def main():
    args = parse_args()

    accelerator = setup_accelerator(args)
    raw_datasets, tokenizer = setup_datasets(args)
    
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if (args.use_loop_tokens or args.use_quantized_loop_tokens or args.use_loop_tokens_whole_sequence):
        with open(args.loop_token_model_config, "r") as f:
            loop_config = json.load(f)
        # load model from local path
        model = BertLoopModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            loop_config=loop_config,
            use_quantized_loop_embeddings=False,
            trust_remote_code=args.trust_remote_code,
            use_loop_tokens_whole_sequence=args.use_loop_tokens_whole_sequence,
        )
        model.bert.embeddings.convert_ids_to_tokens = tokenizer.convert_ids_to_tokens # set the convert_ids_to_tokens method, needed for loop embeddings
    else:
        logger.info("Pretrained model with no loop embeddings")
        model = AutoModelForMaskedLM.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=args.trust_remote_code,
        )

    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer, accelerator, args)
    eval_dataset = tokenized_datasets["validation"]

    # Conditional for small test subsets
    if args.streaming:
        for i, example in enumerate(eval_dataset.take(3)):
            logger.info(f"Sample {i} of the training set: {example}.")
    else:
        if len(eval_dataset) > 3:
            # Log a few random samples from the training set:
            for index in random.sample(range(len(eval_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=1.0,
        mask_replace_prob=1.0,
        random_replace_prob=0.0,
    )

    # DataLoaders creation:
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Prepare everything with our `accelerator`.
    model, eval_dataloader = accelerator.prepare(model, eval_dataloader)

    sampled_sequences, perplexities = sampling_inference(model, eval_dataloader, tokenizer, args.sampling_temperature, num_samples=args.num_samples)

    seq_ids = pd.read_parquet(args.validation_file)["seqid"].tolist()
    output_df_list = []
    for i in range(args.num_samples):
        output_df = pd.DataFrame({
            "seqid": seq_ids,
            "sequence": sampled_sequences[i],
            "perplexity": perplexities,
        })
        output_df["sample_id"] = i
        output_df_list.append(output_df)
    output_df = pd.concat(output_df_list, ignore_index=True)
    output_df['sequence'] = output_df['sequence'].str.replace(" ", "")
    output_df.to_csv(args.output_file, index=False)
    print(f"Saved sampled sequences to {args.output_file}")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()