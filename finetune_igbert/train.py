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
import math
import os
import random
from itertools import chain
from pathlib import Path
import pandas as pd

import datasets

import torch
from accelerate import Accelerator, DistributedType, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset, load_dataset_builder
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from glob import glob

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    BertConfig,
    BertModel,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
)
from custom_data_collator import DataCollatorForLanguageModelingNoSpecialTokens
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from .embedding_model import BertLoopModelForMaskedLM
from .constants import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.53.0.dev0")

logger = get_logger(__name__)
require_version("datasets>=2.14.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
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
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name of the run for logging purposes.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
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
        "--use_special_cdr_tokens",
        action="store_true",
        help="If passed, the model will use special CDR tokens to mark the start and end of CDR sections.",
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
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
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
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help=(
            "Whether to trust the execution of code from datasets/models defined on the Hub."
            " This option should only be set to `True` for repositories you trust and in which you have read the"
            " code, as it will execute code present on the Hub on your local machine."
        ),
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    args = parser.parse_args()

    # Sanity checks
    if args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            if extension not in ["csv", "json", "txt", "parquet"]:
                raise ValueError("`train_file` should be a csv, json, txt, or parquet file.")
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            if extension not in ["csv", "json", "txt", "parquet"]:
                raise ValueError("`validation_file` should be a csv, json, txt, or parquet file.")

    if args.push_to_hub:
        if args.output_dir is None:
            raise ValueError("Need an `output_dir` to create a repo when `--push_to_hub` is passed.")

    if args.use_loop_tokens or args.use_quantized_loop_tokens or args.use_loop_tokens_whole_sequence:
        assert args.use_special_cdr_tokens, "Using loop tokens requires using special CDR tokens."

    if args.load_in_memory and args.streaming:
        raise ValueError("Cannot use `load_in_memory` and `streaming` at the same time. Please set one of them to False.")

    return args


def setup_accelerator(args):
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mlm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    if args.with_tracking:
        accelerator_log_kwargs["log_with"] = args.report_to
        accelerator_log_kwargs["project_dir"] = args.output_dir

    if args.use_loop_tokens or args.use_quantized_loop_tokens or args.use_loop_tokens_whole_sequence:
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs_handlers = [ddp_handler]
    else:
        kwargs_handlers = []
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, kwargs_handlers=kwargs_handlers, **accelerator_log_kwargs)

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

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            # Retrieve of infer repo_name
            repo_name = args.hub_model_id
            if repo_name is None:
                repo_name = Path(args.output_dir).absolute().name
            # Create repo and retrieve repo_id
            api = HfApi()
            repo_id = api.create_repo(repo_name, exist_ok=True, token=args.hub_token).repo_id

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            args.output_dir = get_save_dir(args.output_dir) if args.output_dir else None
            with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
                json.dump(vars(args), f, indent=4)
    accelerator.wait_for_everyone()
    return accelerator, repo_id if args.push_to_hub else None, api if args.push_to_hub else None


def setup_datasets(args, accelerator):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            args.dataset_name, args.dataset_config_name, trust_remote_code=args.trust_remote_code
        )
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                trust_remote_code=args.trust_remote_code,
                streaming=args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                trust_remote_code=args.trust_remote_code,
                streaming=args.streaming,
            )
    else:
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
            extension = args.train_file.split(".")[-1]
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
            extension = args.validation_file.split(".")[-1]
        if extension == "txt":
            extension = "text"

        if args.load_in_memory:
            with accelerator.main_process_first():
                raw_datasets = datasets.DatasetDict(
                    {
                        "train": datasets.Dataset.from_pandas(pd.read_parquet(args.train_file)),
                        "validation": datasets.Dataset.from_pandas(pd.read_parquet(args.validation_file)),
                    }
                )
            # datasets.config.IN_MEMORY_MAX_SIZE = 200 * 1024 * 1024 * 1024 # 200 GB
            # raw_datasets = load_dataset(extension, data_files=data_files, keep_in_memory=True)
        else:
            raw_datasets = load_dataset(extension, data_files=data_files, streaming=args.streaming, cache_dir=os.path.join(args.output_dir, "dataset_cache"))

        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                streaming=args.streaming,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                streaming=args.streaming,
            )
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
    if args.use_special_cdr_tokens:
        num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
        if num_added_tokens > 0:
            logger.info(f"Added {num_added_tokens} new special tokens to the tokenizer: {SPECIAL_TOKENS}")
        for token_str in SPECIAL_TOKENS:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            is_special = token_id in tokenizer.all_special_ids
            logger.info(f"Token '{token_str}' (ID: {token_id}) is in tokenizer.all_special_ids: {is_special}")
    else:
        num_added_tokens = 0

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    return raw_datasets, tokenizer, num_added_tokens


def tokenize_datasets(raw_datasets, tokenizer, accelerator, args):
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names

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

    if args.use_special_cdr_tokens:
        def process_one_example(examples, idx):
            special_tokens_mask = [1]
            sequence = []
            for i, section in enumerate(SECTIONS):
                section_seq = list(examples[section][idx])
                if "FW" in section:
                    sequence.extend(section_seq)
                    special_tokens_mask.extend([0]*len(section_seq))
                else:
                    sequence.extend([SPECIAL_TOKENS[0]] + section_seq + [SPECIAL_TOKENS[1]])
                    special_tokens_mask.extend([1] + [0]*len(section_seq) + [1])
            special_tokens_mask.append(1)  # Add [SEP] token at the end
            if padding == "max_length":
                assert len(special_tokens_mask) <= max_seq_length, "The sequence with special tokens exceeds max_seq_length."
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
            return tokenized
    else:
        def process_one_example(examples, idx):
            sequence = []
            for section in SECTIONS:
                sequence.extend(list(examples[section][idx]))
            return " ".join(sequence)

        def tokenize_function(examples):
            collated_examples = [process_one_example(examples, idx) for idx in range(len(examples[column_names[0]]))]
            tokenized = tokenizer(
                collated_examples,
                padding=padding,
                truncation=True,
                max_length=max_seq_length,
                return_special_tokens_mask=True,
            )
            return tokenized

    with accelerator.main_process_first():
        if args.use_loop_tokens or args.use_quantized_loop_tokens or args.use_loop_tokens_whole_sequence:
            remove_columns = SECTIONS
        else:
            remove_columns = SECTIONS + ["angles"]
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



def evaluate(model, eval_dataloader, accelerator, args, epoch, train_loss, completed_steps, repo_id=None, api=None):
    model.eval()
    losses = []
    correct_masked_aa_predictions = 0
    total_masked_aa_predictions = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
            masked_token_indices = torch.where(batch["labels"] != -100)
            masked_logits = outputs.logits[masked_token_indices]
            predicted_token_ids = torch.argmax(masked_logits, dim=-1)
            masked_true_labels = batch["labels"][masked_token_indices]
            correct_masked_aa_predictions += (predicted_token_ids == masked_true_labels).sum().item()
            total_masked_aa_predictions += masked_true_labels.numel()
        loss = outputs.loss
        losses.append(accelerator.gather_for_metrics(loss.repeat(args.per_device_eval_batch_size)))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")
    
    aa_recovery = correct_masked_aa_predictions / total_masked_aa_predictions if total_masked_aa_predictions > 0 else 0.0

    logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss} aa_recovery: {aa_recovery:.4f}")

    if args.with_tracking:
        accelerator.log(
            {
                "perplexity": perplexity,
                "eval_loss": eval_loss,
                "train_loss": train_loss,
                "aa_recovery": aa_recovery,
                "epoch": epoch,
                "step": completed_steps,
            },
            step=completed_steps,
        )

    if args.push_to_hub and epoch < args.num_train_epochs - 1:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            api.upload_folder(
                commit_message=f"Training in progress epoch {epoch}",
                folder_path=args.output_dir,
                repo_id=repo_id,
                repo_type="model",
                token=args.hub_token,
            )
    return perplexity, eval_loss, aa_recovery


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

    accelerator, repo_id, api = setup_accelerator(args)
    raw_datasets, tokenizer, num_added_tokens = setup_datasets(args, accelerator)
    tokenizer.save_pretrained(
        os.path.join(args.output_dir, "tokenizer"), is_main_process=accelerator.is_main_process, save_function=accelerator.save
    )
    
    if args.config_name:
        config = BertConfig.from_pretrained(args.config_name, trust_remote_code=args.trust_remote_code)
    elif args.model_name_or_path:
        config = BertConfig.from_pretrained(args.model_name_or_path, trust_remote_code=args.trust_remote_code)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.model_name_or_path:
        if (args.use_loop_tokens or args.use_quantized_loop_tokens or args.use_loop_tokens_whole_sequence):
            with open(args.loop_token_model_config, "r") as f:
                loop_config = json.load(f)
            if os.path.exists(args.model_name_or_path):
                # load model from local path
                model = BertLoopModelForMaskedLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    loop_config=loop_config,
                    use_quantized_loop_embeddings=args.use_quantized_loop_tokens,
                    use_loop_tokens_whole_sequence=args.use_loop_tokens_whole_sequence,
                )
            else:
                model = BertLoopModelForMaskedLM(
                    config=config,
                    loop_config=loop_config,
                    use_quantized_loop_embeddings=args.use_quantized_loop_tokens,
                    use_loop_tokens_whole_sequence=args.use_loop_tokens_whole_sequence,
                )
                pretrained_model = AutoModelForMaskedLM.from_pretrained(
                    args.model_name_or_path,
                    from_tf=bool(".ckpt" in args.model_name_or_path),
                    config=config,
                    trust_remote_code=args.trust_remote_code,
                )
                model.load_state_dict(pretrained_model.state_dict(), strict=False)
                model.bert.embeddings.loop_embeddings.load_state_dict(torch.load(args.loop_token_model_weights))
            model.bert.embeddings.convert_ids_to_tokens = tokenizer.convert_ids_to_tokens # set the convert_ids_to_tokens method, needed for loop embeddings
            for param in model.bert.embeddings.loop_embeddings.parameters(): # freeze the loop tokenizer, FIXME: training it causes bugs in multigpu set up because some parameters do not receive grad
                param.requires_grad = False
        else:
            logger.info("Pretrained model with no loop embeddings")
            model = AutoModelForMaskedLM.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                trust_remote_code=args.trust_remote_code,
            )

    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config, trust_remote_code=args.trust_remote_code)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {num_params}")

    # resize because added special tokens
    if num_added_tokens > 0:
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=4)
    model.config.save_pretrained(args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save)
    
    # initialize the new tokens with the CLS token embedding - this did not seem to work so well
    # input_embeddings = model.get_input_embeddings()
    # for idx in range(len(tokenizer) - num_added_tokens, len(tokenizer)):
    #     input_embeddings.weight.data[idx] = input_embeddings.weight.data[tokenizer.cls_token_id]

    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer, accelerator, args)
    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # Conditional for small test subsets
    if args.streaming:
        for i, example in enumerate(train_dataset.take(3)):
            logger.info(f"Sample {i} of the training set: {example}.")
    else:
        if len(train_dataset) > 3:
            # Log a few random samples from the training set:
            for index in random.sample(range(len(train_dataset)), 3):
                logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Data collator
    # This one will take care of randomly masking the tokens.
    data_collator = DataCollatorForLanguageModelingNoSpecialTokens(tokenizer=tokenizer, mlm_probability=args.mlm_probability)

    # DataLoaders creation:
    if args.streaming:
        shuffled_train_dataset = train_dataset.shuffle(
            seed=args.seed, # Use a seed for reproducibility
            buffer_size=10_000 # Choose a suitable buffer size
        )
        train_dataloader = DataLoader(
            shuffled_train_dataset,
            shuffle=False,  # <--- MUST BE FALSE for IterableDataset
            collate_fn=data_collator,
            batch_size=args.per_device_train_batch_size
        )
        len_train_dataset = pd.read_parquet(args.train_file).shape[0]
        len_train_loader = math.ceil(len_train_dataset / (args.per_device_train_batch_size * accelerator.num_processes))
    else:
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
        )
        len_train_dataset = len(train_dataset)
        len_train_loader = math.ceil(len(train_dataloader) / accelerator.num_processes)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm_loops.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be
    # shorter in multiprocess)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len_train_loader / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len_train_loader / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if args.with_tracking:
        experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        if args.report_to == "wandb":
            accelerator.init_trackers(
                project_name="IglooLM/IglooALM",
                config=experiment_config,
                init_kwargs={"wandb": {"dir": "./wandb/", "name": args.run_name}},
            )

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len_train_dataset}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        accelerator.print(f"Resumed from checkpoint: {args.resume_from_checkpoint}")
        accelerator.load_state(args.resume_from_checkpoint, strict=False)
        # Extract `epoch_{i}` or `step_{i}`
        if os.path.exists(os.path.join(args.resume_from_checkpoint, "step.txt")):
            with open(os.path.join(args.resume_from_checkpoint, "step.txt"), "r") as f:
                starting_epoch = int(f.read().strip()) + 1
                resume_step = None
                completed_steps = starting_epoch * num_update_steps_per_epoch
        elif os.path.exists(os.path.join(args.resume_from_checkpoint, "epoch.txt")):
            with open(os.path.join(args.resume_from_checkpoint, "epoch.txt"), "r") as f:
                resume_step = int(f.read().strip()) * args.gradient_accumulation_steps
                starting_epoch = resume_step // len_train_loader
                completed_steps = resume_step // args.gradient_accumulation_steps
                resume_step -= starting_epoch * len_train_loader

    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    best_eval_loss = float("inf")

    for epoch in range(starting_epoch, args.num_train_epochs):
        if args.streaming:
            train_dataloader.dataset.set_epoch(epoch)

        model.train()
        total_loss = 0.0
        if args.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            # We skip the first `n` batches in the dataloader when resuming from a checkpoint
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                # We keep track of the loss at each epoch
                if args.with_tracking:
                    total_loss += loss.detach().float()
                    accelerator.log(
                        {"train_loss": loss.detach().float()},
                        step=completed_steps,
                    )
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0 and accelerator.sync_gradients:
                    output_dir = "last_step" # f"step_{completed_steps}"
                    if args.output_dir is not None:
                        output_dir = os.path.join(args.output_dir, output_dir)
                    accelerator.save_state(output_dir)
                    with open(os.path.join(output_dir, "step.txt"), "w") as f:
                        f.write(str(completed_steps))
                
                    train_loss = total_loss / len_train_loader
                    perplexity, eval_loss, aa_recovery = evaluate(
                        model, eval_dataloader, accelerator, args,
                        epoch, train_loss, completed_steps,
                        repo_id=repo_id, api=api
                    )

                    if eval_loss < best_eval_loss:
                        output_dir = "best_step"
                        if args.output_dir is not None:
                            output_dir = os.path.join(args.output_dir, output_dir)
                        accelerator.save_state(output_dir)
                        logger.info(f"Saving best model for step {completed_steps} with eval_loss {eval_loss} at {output_dir}")
                        with open(os.path.join(output_dir, "best_step.txt"), "w") as f:
                            f.write(str(completed_steps))
                    best_eval_loss = min(best_eval_loss, eval_loss)

            if completed_steps >= args.max_train_steps:
                break
        
        if args.checkpointing_steps == "epoch":
            train_loss = total_loss / len_train_loader
            perplexity, eval_loss, aa_recovery = evaluate(
                model, eval_dataloader, accelerator, args,
                epoch, train_loss, completed_steps,
                repo_id=repo_id, api=api
            )
            output_dir = "last_epoch" # f"epoch_{epoch}"
            if args.output_dir is not None:
                output_dir = os.path.join(args.output_dir, output_dir)
            accelerator.save_state(output_dir)
            with open(os.path.join(output_dir, "epoch.txt"), "w") as f:
                f.write(str(epoch))

            if eval_loss < best_eval_loss:
                output_dir = "best_epoch"
                if args.output_dir is not None:
                    output_dir = os.path.join(args.output_dir, output_dir)
                accelerator.save_state(output_dir)
                logger.info(f"Saving best model for epoch {epoch} with eval_loss {eval_loss} at {output_dir}")
                with open(os.path.join(output_dir, "best_epoch.txt"), "w") as f:
                    f.write(str(epoch))


    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            if args.push_to_hub:
                api.upload_folder(
                    commit_message="End of training",
                    folder_path=args.output_dir,
                    repo_id=repo_id,
                    repo_type="model",
                    token=args.hub_token,
                )
            with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
                json.dump({"perplexity": perplexity, "eval_loss": eval_loss, "aa_recovery": aa_recovery}, f)

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()