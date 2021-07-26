# coding=utf-8
# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT). (Modifications for BERTAC)
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""

""" Modified from examples/run_glue.py in the original HuggingFace Transformers.
This script is for building vocabularies for pretrained CNNs:
 - Other TLM settings except for ALBERT and RoBERTa have been commented out or deleted for simplicity
 - The main fuction is load_examples()
"""


import argparse
import glob
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForSequenceClassification,
    AlbertTokenizer,
#    BertConfig,
#    BertForSequenceClassification,
#    BertTokenizer,
#    DistilBertConfig,
#    DistilBertForSequenceClassification,
#    DistilBertTokenizer,
#    FlaubertConfig,
#    FlaubertForSequenceClassification,
#    FlaubertTokenizer,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
#    XLMConfig,
#    XLMForSequenceClassification,
#    XLMRobertaConfig,
#    XLMRobertaForSequenceClassification,
#    XLMRobertaTokenizer,
#    XLMTokenizer,
#    XLNetConfig,
#    XLNetForSequenceClassification,
#    XLNetTokenizer,
#    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            #BertConfig,
            #XLNetConfig,
            #XLMConfig,
            #RobertaConfig,
            #DistilBertConfig,
            AlbertConfig,
            #XLMRobertaConfig,
            #FlaubertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
    #"bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    #"xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    #"xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    #"distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    #"xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
    #"flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def load_examples(args, filename, tokenizer):
    processor = processors[args.task_name]()
    logger.info("load examples in {}".format(filename))
    if filename == 'train.tsv':
        examples = processor.get_train_examples(args.data_dir)
    if filename == 'dev.tsv' or filename == 'dev_matched.tsv':
        examples = processor.get_dev_examples(args.data_dir)
    if filename == 'dev_mismatched.tsv':
        examples = processor.get_dev_mm_examples(args.data_dir)
    if filename == 'test.tsv':
        examples = processor.get_test_examples(args.data_dir)
    if filename == 'test_mismatched.tsv':
        examples = processor.get_test_mm_examples(args.data_dir) 

    all_tokens = []
    for (i, example) in enumerate(examples):
        #example.text_a: original text1
        #example.text_b: original text2 (if exists)
        sub_tokens = tokenizer.tokenize_for_cnn(example.text_a) 
        for sub_token in sub_tokens:
            all_tokens.append(sub_token)

        if example.text_b is not None: # if the task is for sentence pairs.
           sub_tokens = tokenizer.tokenize_for_cnn(example.text_b)
           for sub_token in sub_tokens:
               all_tokens.append(sub_token)
    return all_tokens

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    parser.add_argument(
        "--ostem",
        default=None,
        type=str,
        required=True,
        help="The output vocab file's stem",
    )

    # Other parameters
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=None,
    )

    all_tokens = []
    train_file = "train.tsv"
    dev_file = ['dev_matched.tsv', 'dev_mismatched.tsv'] if args.task_name == 'mnli' else ['dev.tsv']
    for i in range(len(dev_file)):
       if os.path.isfile(os.path.join(args.data_dir, dev_file[i])):
          logger.info("{} is processed".format(dev_file[i]))
          eval_tokens = load_examples(args, dev_file[i], tokenizer)
          all_tokens += eval_tokens

    if os.path.isfile(os.path.join(args.data_dir, train_file)):
       logger.info("{} is processed".format(train_file))
       train_tokens = load_examples(args, train_file, tokenizer)
       all_tokens += train_tokens

    bt_stem = "ot"
    output_vocab_file_bin = os.path.join(args.output_dir, "{}_{}_vocab_file.bin".format(args.ostem, bt_stem))
    logger.info("Save bin file for vocab at {}".format(output_vocab_file_bin))
    torch.save({"tokens": all_tokens}, output_vocab_file_bin)

if __name__ == "__main__":
    main()
