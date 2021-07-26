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

""" Modified from examples/run_glue.py in the original HuggingFace Transformers for caching feature files before training/testing.
Other model settings except for ALBERT and RoBERTa have been commented out or deleted for simplicity.
The following libraries/functions/classes have been added/modified:
  - Libraries: torchtext, cnn_utils, and train_utils are additionaly imported
  - Functions/classes: 
     class TTDataset(torchtext.data.Dataset):
     def load_cnn_model_and_vocab():
     def load_and_cache_examples(): the main body of caching
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

## added by Jong-Hoon Oh
import torchtext
import cnn_utils
import train_utils

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

## added by Jong-Hoon Oh
class TTDataset(torchtext.data.Dataset):
    '''Dummy Dataset for build_vocab'''
    def __init__(self, words, fields):
        data_fields = [('text', fields['text'])]
        ex = (words,)
        examples = [torchtext.data.Example.fromlist(ex, data_fields)]
        super(TTDataset, self).__init__(examples, data_fields)

## added by Jong-Hoon Oh
def load_cnn_model_and_vocab(args, cnn_file, words):
    assert args.emb_file and args.min_freq

    fields = cnn_utils.get_fields()
    # build vocabularies from words (idx of input)
    train_utils.build_vocab(args, fields, TTDataset(words, fields), [])
    # load pre-trained generator model
    vocab = fields['text'].vocab
    model, pre_fields = train_utils.load_cnn_model(args, cnn_file, fields)

    return model, pre_fields['text'].vocab.stoi


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


# modified by Jong-Hoon Oh
# - Converting input examples to cached examples
# - cnn_stoi: vocab.stoi for cnn models
def load_and_cache_examples(args, task, filename, tokenizer, cnn_stoi, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    fstem =  list(filter(None,filename.split("/"))).pop() 
    fstem =  fstem.split(".")[0]
    data_type = args.task_name

    feat_dir = args.feat_dir if args.feat_dir is not None else "."
    cached_features_file = os.path.join(
        args.feat_dir,
        data_type,
        "cached_{}_{}_{}_{}_{}".format(
            fstem,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.cnn_stem,
            list(filter(None, args.cnn_model.split("_"))).pop(),
            str(args.max_seq_length),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s with fstem %s", args.data_dir, fstem)
        logger.info("FSTEM: {}".format(fstem))
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        if fstem == 'train':
           examples = processor.get_train_examples(args.data_dir) 
        elif fstem in ["dev", "dev_matched"]:
           examples = processor.get_dev_examples(args.data_dir) 
        elif fstem == 'dev_mismatched':
           examples = processor.get_dev_mm_examples(args.data_dir) 
        elif fstem == 'test_mismatched':
           examples = processor.get_test_mm_examples(args.data_dir) 
        elif fstem in ["test", "test_matched"]:
           examples = processor.get_test_examples(args.data_dir) 

        features = convert_examples_to_features(
            examples,
            tokenizer,
            cnn_stoi=cnn_stoi,
            label_list=label_list,
            max_length=args.max_seq_length,
            output_mode=output_mode,
            pad_on_left=bool(args.model_type in ["xlnet"]),  # pad on the left for xlnet
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

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
        default="cola",
        type=str,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
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
        "--feat_dir",
        default="",
        type=str,
        help="Where do you want to store the processed data whose features were extracted from the input data",
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
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    # added by Jong-Hoon Oh
    parser.add_argument(
        "--prep_vocab_file",
        default=None,
        type=str,
        required=True,
        help="The preprocessed_vocab_file. see make_glue_cnn_vocab.py",
    )
    parser.add_argument(
        "--emb_file",
        default=None,
        type=str,
        required=True,
        help="The embedding vector file used for CNN",
    )
    parser.add_argument(
        "--cnn_model",
        default=None,
        type=str,
        required=True,
        help="The CNN model file name",
    )
    parser.add_argument(
        "--cnn_stem",
        default="enwiki",
        type=str,
        help="file stem for CNN models for caching",
    )
    parser.add_argument(
        "--min_freq",
        default=5,
        type=int,
        help="min freq. for unknown words",
    )
    parser.add_argument(
        "--emb_dim",
        default=None,
        type=int,
        help="dim for representation of fastText",
    )


    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    args = parser.parse_args()
    assert args.prep_vocab_file is not None
    assert args.cnn_model is not None
    assert args.emb_dim is not None
    assert args.emb_file is not None

    if (not os.path.exists(args.prep_vocab_file)):
        raise ValueError(
            "prep_vocab_file ({}) does not exist. Check the --prep_vocab_file option.".format( args.prep_vocab_file) )
    
    if (not os.path.exists(args.cnn_model)):
        raise ValueError(
            "cnn_model ({}) does not exist. Check the --cnn_model option.".format( args.cnn_model) )

    if (not os.path.exists(args.emb_file)):
        raise ValueError(
            "emb_file ({}) does not exist. Check the --emb_file option.".format( args.emb_file) )

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
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
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

    # added by Jong-Hoon Oh
    # prep_vocab_file: see run_preprocessor.sh
    prep_tokens = torch.load(args.prep_vocab_file)
    all_tokens = prep_tokens['tokens']

    # Here, loading CNN models
    cnn_model, cnn_stoi = load_cnn_model_and_vocab(args, args.cnn_model, all_tokens)
    cnn_dim = len(cnn_model.args.filter_widths) * cnn_model.args.filter_size
    args.cnn_dim = cnn_dim 

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    # CONFIG 
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=None,
    )
    config.num_of_TIERs = 3 
    config.cnn_dim = args.cnn_dim
    config.emb_dim = args.emb_dim
    config.cnn_model = args.cnn_model
    config.cnn_stem = args.cnn_stem

    # TOKENIZER
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=None,
    )
    # MODEL
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=None,
    )

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    cnn_model.to(args.device)

    train_file = "train.tsv"
    dev_file = ["dev_matched.tsv","dev_mismatched.tsv"] if args.task_name == "mnli" or args.task_name == "mnlim" else ["dev.tsv", ] 
    #test_file = ["test_matched.tsv","test_mismatched.tsv"] if args.task_name == "mnli" or args.task_name == "mnlim" else ["test.tsv", ] 
    ###################
    ####  MAIN
    ###################
    logger.info("==== {} ====".format(train_file))
    logger.info("==== {} ====".format(dev_file))
    for i in range(len(dev_file)):
       if os.path.isfile(os.path.join(args.data_dir, dev_file[i])):
          logger.info("==== {} ====".format(dev_file[i]))
          load_and_cache_examples(args, args.task_name, dev_file[i], tokenizer, cnn_stoi, evaluate=True)
    if os.path.isfile(os.path.join(args.data_dir, train_file)):
       logger.info("==== {} ====".format(train_file))
       load_and_cache_examples(args, args.task_name, train_file, tokenizer, cnn_stoi, evaluate=False)

if __name__ == "__main__":
    main()
