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

""" Modified from examples/run_squad.py in the original Huggingface Transformers for training/testing a passage selector.
Other model settings except for ALBERT and RoBERTa have been commented out or deleted for simplicity.
The following libraries/functions/classes have been added/modified:
  - Libraries: torchtext, cnn_utils, and train_utils are additionaly imported
  - Functions/classes:
     class TTDataset(torchtext.data.Dataset): 
     def load_and_cache_examples(): just loads cached examples
	     (does not cache examples here but loads them already cached by 'run_openqa_preprocess.py')
     def load_cnn_model_and_vocab(): loads the pretrained cnn and its vocab.
     def train(): trains a model
     def evaluate(): evaluates a trained model
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
    AlbertForSequenceClassification,
    AlbertTokenizer,
    #BertConfig,
    #BertForQuestionAnswering,
    #BertForSequenceClassification,
    #BertTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
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

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (tuple(conf.pretrained_config_archive_map.keys()) for conf in (AlbertConfig,RobertaConfig,)),
    (),
)

MODEL_CLASSES = {
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
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
# - cnn_model should be given as an argument
def train(args, cnn_model, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    ctrain_dataset = train_dataset
    train_sampler = RandomSampler(ctrain_dataset) if args.local_rank == -1 else DistributedSampler(ctrain_dataset)
    train_dataloader = DataLoader(ctrain_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    revised_warmup_steps = int(t_total * args.warmup_ratio)
    logger.info("WARMUP STEPS: {}".format(revised_warmup_steps))

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=revised_warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(ctrain_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Warmup = %d", revised_warmup_steps)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    # Added here for reproductibility
    set_seed(args)

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)

            inputs = {
                "input_ids": batch[0],
                "input_qids": batch[1], # added by Jong-Hoon Oh
                "input_pids": batch[2], # added by Jong-Hoon Oh
                "attention_mask": batch[3],
                "token_type_ids": batch[4],
                "labels": batch[7],
                "Gh": cnn_model, # added by Jong-Hoon Oh
                "num_of_TIERs": args.num_of_TIERs, # added by Jong-Hoon Oh
            }

            outputs = model(**inputs)
            # model outputs are always tuple in transformers (see doc)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                # Log metrics
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    if args.local_rank == -1 and args.evaluate_during_training:  
                    # Only evaluate when single GPU otherwise metrics may not average well
                        results, _, _ , _ = evaluate(args, cnn_model, model, tokenizer, prefix=str(global_step))

                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # Save model checkpoint
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0 and args.save_checkpoints:
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, "module") else model
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def compute_metrics(preds, labels):
    return {"acc": simple_accuracy(preds, labels)}

# modified by Jong-Hoon Oh
# - cnn_model should be given as an argument
def evaluate(args, cnn_model, model, tokenizer, prefix="test"):
    # modified by Jong-Hoon Oh
    # - it simply gets all the data, examples, and features from cache
    dataset, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    running_output = []
    running_y_true = []
    example_ids = []

    start_time = timeit.default_timer()
    results = {}
    
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "input_qids": batch[1], # added by Jong-Hoon Oh
                "input_pids": batch[2], # added by Jong-Hoon Oh
                "attention_mask": batch[3],
                "token_type_ids": batch[4],
                "labels": batch[6],
                "Gh": cnn_model, # added by Jong-Hoon Oh
                "num_of_TIERs": args.num_of_TIERs, # added by Jong-Hoon Oh
            }


            example_indices = batch[5] # all_example_index

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

        running_output.append(logits.detach().to('cpu'))
        running_y_true.append(inputs["labels"].detach().to('cpu'))

        if preds is None:
           preds = logits.detach().cpu().numpy()
           out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
           preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
           out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        # features used here
        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            qas_id = eval_feature.qas_id
            example_ids.append(qas_id)

    running_output = torch.nn.functional.softmax(torch.cat(running_output).to(torch.float32), dim=1)
    running_y_true = torch.cat(running_y_true)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = compute_metrics(preds, out_label_ids)
    results.update(result)

    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(result[key]))

    evalTime = timeit.default_timer() - start_time
    logger.info("  Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))
    return results, running_output, running_y_true, example_ids

# modified by Jong-Hoon Oh
# - Assuming that there are cached example files pre-processed by 'run_openqa_preprocess.py'
# - cnn_stoi: vocab.stoi for the CNN model
def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Load data features from cache
    #  - args.feat_dir => for cached feature file
    bert_token_str = "ot0"
    input_dir = args.feat_dir if args.feat_dir else "."
    dset_dir = input_dir + '/dset'
    pdset_dir = input_dir + '/pdset'
    feat_dir = input_dir + '/feat'
    exset_dir = input_dir + '/exset'
    fstem =  list(filter(None,args.predict_file.split("/"))).pop() if evaluate else list(filter(None, args.train_file.split("/"))).pop()
    fstem =  fstem.split(".")[0]

    # modified by Jong-Hoon Oh
    #  - for less memory use, we split the dataset into four parts (dset, pdset, feat, exset)
    bert_model_name = None
    blist = list(filter(None, args.model_name_or_path.split("/")))
    for i in range(len(blist)):
       if 'roberta-' in blist[i] or 'bert-' in blist[i]:
           flist = list(filter(None, blist[i].split(".")))
           for j in range(len(flist)):
               if 'roberta-' in flist[j] or 'bert-' in flist[j]:
                  bert_model_name = flist[j]
    assert bert_model_name is not None

    cached_file = "cached_{}_{}_{}_{}_{}_{}".format(
            fstem,
            bert_model_name,
            args.cnn_stem,
            list(filter(None, args.cnn_model.split("_"))).pop(),
            bert_token_str,
            str(args.max_seq_length),
        )
    cached_dset_file = os.path.join(dset_dir,cached_file)
    cached_pdset_file = os.path.join(pdset_dir,cached_file)
    cached_feat_file = os.path.join(feat_dir,cached_file)
    cached_exset_file = os.path.join(exset_dir,cached_file)
    # modified by Jong-Hoon Oh
    if evaluate:
       logger.info("Specified cached file %s for evluate", cached_file)
    else:
       logger.info("Specified cached file %s for train", cached_file)

    # Init features and dataset from cache if it exists
    # - Assuming that all the input examples are already cached 
    if os.path.exists(cached_dset_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_dset_file)
        dset = torch.load(cached_dset_file) # selector uses dset["dataset"]
        dataset = dset["dataset"]
        del dset
        if evaluate:  # evaluation needs features
           feat_set = torch.load(cached_feat_file) 
           features = feat_set["features"]
           del feat_set 
    else:
        if args.local_rank == 0 and not evaluate:
            # Make sure only the first process in distributed training process the dataset, and the others will use the cache
            torch.distributed.barrier()
        logger.info("No cached file at %s", input_dir) # input_dir="." by defaults
        if (not os.path.exists(cached_dset_file)):
            raise ValueError(
            "cached file ({}) does not exist. Run run_openqa_preprocess.py first.".format( cached_dset_file ) )
        return  

    if args.local_rank == 0 and not evaluate:
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    if output_examples:
        return dataset, features

    return dataset


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
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
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
        help="The embedding vector file used for CNN",
    )
    parser.add_argument(
        "--cnn_model",
        default=None,
        type=str,
        help="The CNN model file name",
    )
    parser.add_argument(
        "--cnn_stem",
        default="enwiki",
        type=str,
        help="stem for CNN models for caching",
    )
    parser.add_argument(
        "--min_freq",
        default=5,
        type=int,
        help="min freq. for unknown words",
    )
    parser.add_argument(
        "--num_of_TIERs",
        default=1,
        type=int,
        help="number of TIER layers",
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
    parser.add_argument("--pred_dev", action="store_true", help="If true, the predict_file is assumed as development data.")
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
        "--version_2_with_negative",
        action="store_true",
        help="If true, the SQuAD examples contain some that do not have an answer.",
    )
    parser.add_argument(
        "--null_score_diff_threshold",
        type=float,
        default=0.0,
        help="If null_score - best_non_null is greater than the threshold predict null.",
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
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio.")
    parser.add_argument(
        "--n_best_size",
        default=20,
        type=int,
        help="The total number of n-best predictions to generate in the nbest_predictions.json output file.",
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

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_checkpoints", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
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

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

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

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will download model & vocab
        # The barrier starts
        torch.distributed.barrier()

    # added by Jong-Hoon Oh
    # - Loading adv model and pre-processed vocab. 
    prep_tokens = torch.load(args.prep_vocab_file)
    all_tokens = prep_tokens['tokens']
    cnn_model, _ = load_cnn_model_and_vocab(args, args.cnn_model, all_tokens)
    cnn_dim = len(cnn_model.args.filter_widths) * cnn_model.args.filter_size
    args.cnn_dim = cnn_dim

    # deleting tokens that are never used later
    del all_tokens
    del prep_tokens

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.num_of_TIERs = args.num_of_TIERs
    config.cnn_dim = args.cnn_dim
    config.emb_dim = args.emb_dim
    config.cnn_model = args.cnn_model
    config.cnn_stem = args.cnn_stem
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path), # ckpt: tensorflow file, pt: pytorch file
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

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

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False)
        global_step, tr_loss = train(args, cnn_model, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Save the trained model and the tokenizer
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        # Take care of distributed/parallel training
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned, where they are in args.output_dir
        model = model_class.from_pretrained(args.output_dir)  
        tokenizer = tokenizer_class.from_pretrained(args.output_dir,
              do_lower_case=args.do_lower_case,
        )
        model.to(args.device)

    # Evaluation - we can ask to evaluate all the checkpoints (sub-directories) in a directory
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = [args.output_dir]
            if args.eval_all_checkpoints:
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)) 
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
        else:
            # Testing 
            checkpoints = [args.model_name_or_path]
            if args.eval_all_checkpoints:
                logger.info("EVAL_ALL_CHECHPOINTS")
                checkpoints = list(
                    os.path.dirname(c)
                    for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)) 
                )
                logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce model loading logs
            logger.info("Loading checkpoint %d for evaluation", len(checkpoints))

        logger.info("Evaluate the following checkpoints: %s", checkpoints)

        for checkpoint in checkpoints:
            # Reload the model
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            model = model_class.from_pretrained(checkpoint)  # , force_download=True)
            model.to(args.device)
            logger.info("Loading check point and eval: [%s]",global_step)
            if args.fp16:
                try:
                    from apex import amp
                except ImportError:
                    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        
                model = amp.initialize(model, opt_level=args.fp16_opt_level)

            # Evaluate
            if global_step == "":
               global_step  = "dev" if args.pred_dev else "test"
            result, outputs, y_true, eids = evaluate(args, cnn_model, model, tokenizer, prefix=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)
            # storing the results into the output file
            output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(global_step))
            with open(output_prediction_file, "w") as writer:
               for i in range(len(outputs)):
                  writer.write("%s\t%f\t%d\n" % (eids[i], outputs[i][1], y_true[i]))

    logger.info("Results: {}".format(results))

    return results

if __name__ == "__main__":
    main()
