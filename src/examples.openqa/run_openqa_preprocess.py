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

""" Modified from examples/run_squad.py in the original Huggingface Transformers for caching feature files before training/testing.
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
import logging
import os
import random
import timeit

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    AlbertConfig,
    AlbertForQuestionAnswering,
    AlbertTokenizer,
    #BertConfig,
    #BertForQuestionAnswering,
    #BertForSequenceClassification,
    #BertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    openqa_convert_examples_to_features,
)
from transformers.data.metrics.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)
from transformers.data.processors.openqa import OpenQAResult, OpenQAV1Processor, OpenQAV2Processor


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

## added by Jong-Hoon Oh
import torchtext
import cnn_utils
import train_utils

NUM_PARALLEL_EXEC_UNITS = 4
os.environ['OMP_NUM_THREADS'] = str(NUM_PARALLEL_EXEC_UNITS)
os.environ["KMP_AFFINITY"] = "granularity=fine,verbose,compact,1,0"
os.environ['KMP_WARNINGS'] = 'off'

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (AlbertConfig, RobertaConfig,)),
    (),
)

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionAnswering, RobertaTokenizer),
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
    train_utils.build_vocab(args, fields, TTDataset(words, fields), [])
    vocab = fields['text'].vocab
    model, pre_fields = train_utils.load_cnn_model(args, cnn_file, fields)

    return model, pre_fields['text'].vocab.stoi



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()

# modified by Jong-Hoon Oh
# DATA PROCESSING PART
# - Converting input examples to cached examples
# - cnn_stoi: vocab.stoi for the cnn model
def load_and_cache_examples(args, filename, tokenizer, cnn_stoi, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    bert_token_str = "ot0"
    input_dir = args.feat_dir if args.feat_dir else "."
    fstem =  list(filter(None,filename.split("/"))).pop()
    fstem =  fstem.split(".")[0]
    fstem =  fstem
    cached_file = "cached_{}_{}_{}_{}_{}_{}".format(
            fstem,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.cnn_stem,
            list(filter(None, args.cnn_model.split("_"))).pop(),
            bert_token_str,
            str(args.max_seq_length),
        )
    # split the input data into data, positive_data, feature, and example
    dset_dir = input_dir + '/dset'
    pdset_dir = input_dir + '/pdset'
    feat_dir = input_dir + '/feat'
    exset_dir = input_dir + '/exset'
    cached_dset_file = os.path.join(dset_dir,cached_file)
    cached_feat_file = os.path.join(feat_dir,cached_file)
    cached_pdset_file = os.path.join(pdset_dir,cached_file)
    cached_exset_file = os.path.join(exset_dir,cached_file)
    if evaluate:
       logger.info("Specified cached file %s for dev or predict files", cached_dset_file)
    else:
       logger.info("Specified cached file %s for train files", cached_dset_file)

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_dset_file) and not args.overwrite_cache:
        logger.info("Feature files already exist: %s", cached_dset_file)
    else:
        logger.info("Creating features from dataset file at %s", input_dir) # input_dir="." by defaults

        # if no predict file for evaluation or no train file for training
        if not args.data_dir and ((evaluate and not args.predict_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQuAD.")

            tfds_examples = tfds.load("openqa")
            examples = OpenQAV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            # The main part of data processing in our OpenQA experiments
            processor = OpenQAV1Processor()

            if evaluate:
                # initializer
                examples = processor.get_dev_examples(args.data_dir, filename=filename)
            else:
                # initializer
                examples = processor.get_train_examples(args.data_dir, filename=filename)

        features, dataset, possible_dataset = openqa_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            cnn_stoi=cnn_stoi,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            return_dataset="pt", # "pt" represents 'pytorch dataset'
            threads=args.threads,
        )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_dset_file)
            if evaluate:
               logger.info("dataset:{}".format(len(dataset)))
               torch.save({"dataset": dataset}, cached_dset_file)
               logger.info("features")
               torch.save({"features": features}, cached_feat_file)
               logger.info("examples")
               torch.save({"examples": examples}, cached_exset_file)
            else:
               logger.info("dataset:{}".format(len(dataset)))
               torch.save({"dataset": dataset}, cached_dset_file)
               logger.info("possible_dataset:{}".format(len(possible_dataset)))
               torch.save({"possible_dataset": possible_dataset}, cached_pdset_file)

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

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
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--prep_vocab_file",
        default=None,
        type=str,
        help="preprocessed_vocab_file with the train/dev/predict file. see make_openqa_cnn_vocab.py",
    )
    parser.add_argument(
        "--emb_file",
        default=None,
        type=str,
        help="The embedding vector file used for cnn",
    )
    parser.add_argument(
        "--cnn_model",
        default=None,
        type=str,
        help="The cnn model file name",
    )
    parser.add_argument(
        "--cnn_stem",
        default="enwiki",
        type=str,
        help="stem for cnn models for caching (different vocab.stoi for each model)",
    )
    parser.add_argument(
        "--min_freq",
        default=5,
        type=int,
        help="min freq. for unknown words",
    )
    parser.add_argument(
        "--emb_dim",
        default=300,
        type=int,
        help="dim for representation of fastText",
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
        "--predict_file",
        default=None,
        type=str,
        help="The input evaluation file. If a data dir is specified, will look for the file there"
        + "If no data dir or train/predict files are specified, will run with tensorflow_datasets.",
    )
    parser.add_argument(
        "--dev_file",
        default=None,
        type=str,
        help="The input development file. If a data dir is specified, will look for the file there"
        + "If no data dir or devel files are specified, will run with tensorflow_datasets.",
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
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--feat_dir",
        default="",
        type=str,
        help="Where do you want to store the processed data whose features were extracted from the input data",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
        "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
        "be truncated to this length.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument(
        "--max_answer_length",
        default=30,
        type=int,
        help="The maximum length of an answer that can be generated. This is needed because the start "
        "and end predictions are not conditioned on one another.",
    )
    parser.add_argument(
        "--verbose_logging",
        action="store_true",
        help="If true, all of the warnings related to data processing will be printed. "
        "A number of warnings are expected for a normal SQuAD evaluation.",
    )
    parser.add_argument(
        "--lang_id",
        default=0,
        type=int,
        help="language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)",
    )

    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus")
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
    parser.add_argument("--server_ip", type=str, default="", help="Can be used for distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="Can be used for distant debugging.")

    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()

    assert args.prep_vocab_file is not None
    assert args.cnn_model is not None
    assert args.cnn_stem is not None
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

    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

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

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        # The barrier starts
        torch.distributed.barrier()

    # added by Jong-Hoon Oh
    #  - Load cnn model and pre-processed vocab.
    #  - prep_vocab_file: see vocab/
    prep_tokens = torch.load(args.prep_vocab_file)
    all_tokens = prep_tokens['tokens']
    cnn_model, cnn_stoi = load_cnn_model_and_vocab(args, args.cnn_model, all_tokens)
    cnn_dim = len(cnn_model.args.filter_widths) * cnn_model.args.filter_size
    args.cnn_dim = cnn_dim

    args.model_type = args.model_type.lower()
    # "albert": (AlbertConfig, AlbertForQuestionAnswering, AlbertTokenizer),
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.num_of_TIERs = 3
    config.cnn_dim = args.cnn_dim
    config.emb_dim = args.emb_dim
    config.cnn_model = args.cnn_model
    config.cnn_stem = args.cnn_stem
    # tokenizer_class: AlbertTokenizer
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # model_class: AlbertForQuestionAnswering
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path), # ckpt: tensorflow file, pt: pytorch file
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    ###########

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will download model & vocab
        # The barrier ends
        torch.distributed.barrier()

    model.to(args.device)
    cnn_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Before we do anything with models, we want to ensure that we get fp16 execution of torch.einsum if args.fp16 is set.
    # Otherwise it'll default to "promote" mode, and we'll get fp32 operations. Note that running `--fp16_opt_level="O2"` will
    # remove the need for this code, but it is still valid.
    if args.fp16:
        try:
            import apex

            apex.amp.register_half_function(torch, "einsum")
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    if args.train_file is not None:
       load_and_cache_examples(args, args.train_file, tokenizer, cnn_stoi, evaluate=False, output_examples=False)
    if args.predict_file is not None:
       load_and_cache_examples(args, args.predict_file, tokenizer, cnn_stoi, evaluate=True, output_examples=True)
    if args.dev_file is not None:
       load_and_cache_examples(args, args.dev_file, tokenizer, cnn_stoi, evaluate=True, output_examples=True)

if __name__ == "__main__":
    main()
