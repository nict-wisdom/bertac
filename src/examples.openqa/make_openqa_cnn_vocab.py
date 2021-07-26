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
""" Finetuning the library models for question-answering on SQuAD (DistilBERT, Bert, XLM, XLNet)."""

""" Modified from examples/run_squad.py in the original Huggingface Transformers.
This script is for building vocabularies for pretrained CNNs.
 - Other TLM settings except for ALBERT and RoBERTa have been commented out or deleted for simplicity
 - The main fuction is load_examples()
"""


import argparse
import glob
import logging
import os
import io
import sys
import random
import timeit

import numpy as np
import torch
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    #BertConfig,
    #BertForQuestionAnswering,
    #BertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    #get_linear_schedule_with_warmup,
    #squad_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.openqa import OpenQAResult, OpenQAV1Processor, OpenQAV2Processor


logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (AlbertConfig,RobertaConfig, )),
    (),
)

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
}



# Modified by Jong-Hoon Oh
def load_examples(args, filename, tokenizer, evaluate=False, output_examples=False):
    processor = OpenQAV2Processor()
    # Required parameters
    if evaluate:
        logger.info("load examples in {}".format(filename))
        examples = processor.get_dev_examples(args.data_dir, filename=filename)
    else:
        logger.info("load examples in {}".format(filename))
        examples = processor.get_train_examples(args.data_dir, filename=filename)

    all_tokens = []
    for (i, example) in enumerate(examples):
        for (j, token) in enumerate(example.q_tokens): 
           sub_tokens = tokenizer.tokenize_for_cnn(token)
           for sub_token in sub_tokens:
               all_tokens.append(sub_token)
        for (j, token) in enumerate(example.doc_tokens): 
           sub_tokens = tokenizer.tokenize_for_cnn(token)
           for sub_token in sub_tokens:
               all_tokens.append(sub_token)
    return all_tokens

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default="",
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument(
        "--ostem",
        default="squad",
        type=str,
        required=True,
        help="stem for output file",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )

    # Other parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        help="The input data dir. Should contain the .json files for the task."
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--dev_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
       print("no directory with such a name: {}".format(args.output_dir))
       return

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    args.model_type = args.model_type.lower()
    # Only the ALBERT model is allowed 
    assert args.model_type == 'albert' or args.model_type == 'roberta'
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=None,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=None,
    )

    all_tokens = []
    if args.train_file is not None:
       logger.info("{} is processed".format(args.train_file))
       train_tokens = load_examples(args, args.train_file, tokenizer, evaluate=False)
       all_tokens += train_tokens
    if args.predict_file is not None:
       logger.info("{} is processed".format(args.predict_file))
       eval_tokens = load_examples(args, args.predict_file, tokenizer, evaluate=True)
       all_tokens += eval_tokens
    if args.dev_file is not None:
       logger.info("{} is processed".format(args.dev_file))
       dev_tokens = load_examples(args, args.dev_file, tokenizer, evaluate=True)
       all_tokens += dev_tokens

    bt_stem = "ot" 
    output_vocab_file_bin = os.path.join(args.output_dir, "{}_{}_vocab_file.bin".format(args.ostem, bt_stem))
    logger.info("Save bin file for vocab at {}".format(output_vocab_file_bin))
    torch.save({"tokens": all_tokens}, output_vocab_file_bin)

if __name__ == "__main__":
    main()
