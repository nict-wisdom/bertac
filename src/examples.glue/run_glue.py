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
from __future__ import print_function
import argparse
import glob
import json
import logging
import os
import sys
import io
import random

""" Modified from examples/run_glue.py in the original HuggingFace Transformers.
Other model settings except for ALBERT and RoBERTa have been commented out or deleted for simplicity.
The following libraries/functions/classes have been added/modified:
  - Libraries: torchtext, cnn_utils, and train_utils are additionaly imported
  - Functions/classes: (see '## added by Jong-Hoon Oh')
     class TTDataset(torchtext.data.Dataset):
     def load_and_cache_examples(): just loads cached examples
	     (do not cache examples here but loads them already cached by 'run_openqa_preprocess.py')
     def load_cnn_model_and_vocab(): loads the pretrained cnn and its vocab.
     def train(): train a model
     def evaluate(): evaluate a trained model
"""

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
    AlbertForSequenceClassification4SingleSent,
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
    RobertaForSequenceClassification4SingleSent,
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
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# added by Jong-Hoon Oh
import torchtext
import cnn_utils
import train_utils

logger = logging.getLogger(__name__)

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (
            #BertConfig,
#            XLNetConfig,
#            XLMConfig,
#            RobertaConfig,
#            DistilBertConfig,
            AlbertConfig,
#            XLMRobertaConfig,
#            FlaubertConfig,
        )
    ),
    (),
)

MODEL_CLASSES = {
#    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
#    "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
#    "xlm": (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaForSequenceClassification4SingleSent, RobertaTokenizer),
#    "distilbert": (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer),
    "albert": (AlbertConfig, AlbertForSequenceClassification, AlbertForSequenceClassification4SingleSent, AlbertTokenizer),
#    "xlmroberta": (XLMRobertaConfig, XLMRobertaForSequenceClassification, XLMRobertaTokenizer),
#    "flaubert": (FlaubertConfig, FlaubertForSequenceClassification, FlaubertTokenizer),
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


def train(args, train_dataset, model, cnn_model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # applying warmup_ratio
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
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
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
    logger.info("  Total warmup steps = %d", revised_warmup_steps)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            if args.task_name == 'cola' or args.task_name == 'sst-2': # single-sentence task
               #TensorDataset(all_input_ids, all_input_qids, all_attention_mask, all_token_type_ids, all_labels)
               inputs = {"input_ids": batch[0], "input_qids": batch[1],
                      "attention_mask": batch[2], "labels": batch[4], "Gh": cnn_model, "num_of_TIERs": args.num_of_TIERs}
               if args.model_type != "distilbert":
                   inputs["token_type_ids"] = (
                       batch[3] if args.model_type in ["bert", "xlnet", "albert"] else None
                   )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
            else: # sentence-pair tasks
               #TensorDataset(all_input_ids, all_input_qids, all_input_pids, all_attention_mask, all_token_type_ids, all_labels)
               inputs = {"input_ids": batch[0], "input_qids": batch[1], "input_pids": batch[2],
                      "attention_mask": batch[3], "labels": batch[5], "Gh": cnn_model, "num_of_TIERs": args.num_of_TIERs}
               if args.model_type != "distilbert":
                   inputs["token_type_ids"] = (
                       batch[4] if args.model_type in ["bert", "xlnet", "albert"] else None
                   )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
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

                #==============
                # Please, make sure to call `optimizer.step()` before `lr_scheduler.step()`
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                #==============
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, cnn_model, tokenizer, cnn_stoi)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["loss"] = loss_scalar
                    logging_loss = tr_loss

                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0 and args.save_checkpoints:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
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


def evaluate(args, model, cnn_model, tokenizer, cnn_stoi, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("{}".format(args.task_name), "{}-mm".format(args.task_name)) if args.task_name == "mnli" or args.task_name == "mnlim" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir) if args.task_name == "mnli" or args.task_name == "mnlim" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        test_file = "dev.tsv"
        if eval_task == 'mnli':
           test_file = "dev_matched.tsv"
        elif eval_task == 'mnli-mm':
           test_file = "dev_mismatched.tsv"
        eval_dataset, features = load_and_cache_examples(args, eval_task, test_file, tokenizer, cnn_stoi, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        running_output = []
        running_y_true = []
        example_ids = []
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                if args.task_name == 'cola' or args.task_name == 'sst-2': # single-sentence task
                   #TensorDataset(all_input_ids, all_input_qids, all_attention_mask, all_token_type_ids, all_labels)
                   inputs = {"input_ids": batch[0], "input_qids": batch[1],
                          "attention_mask": batch[2], "labels": batch[4], "Gh": cnn_model, "num_of_TIERs": args.num_of_TIERs}
                   if args.model_type != "distilbert":
                       inputs["token_type_ids"] = (
                           batch[3] if args.model_type in ["bert", "xlnet", "albert"] else None
                       )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                   example_indices = batch[5] # all_example_index
                else: # sentence-pair tasks
                   #TensorDataset(all_input_ids, all_input_qids, all_input_pids, all_attention_mask, all_token_type_ids, all_labels)
                   inputs = {"input_ids": batch[0], "input_qids": batch[1], "input_pids": batch[2],
                          "attention_mask": batch[3], "labels": batch[5], "Gh": cnn_model, "num_of_TIERs": args.num_of_TIERs}
                   if args.model_type != "distilbert":
                       inputs["token_type_ids"] = (
                           batch[4] if args.model_type in ["bert", "xlnet", "albert"] else None
                       )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                   example_indices = batch[6] # all_example_index
                outputs = model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1


            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

            # Store the logits results to running output
            running_output.append(logits.detach().to('cpu'))
            running_y_true.append(inputs["labels"].detach().to('cpu'))
            # features used here
            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                uid = eval_feature.uid
                example_ids.append(uid)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            preds = np.argmax(preds, axis=1)
            running_output = torch.nn.functional.softmax(torch.cat(running_output).to(torch.float32), dim=1)
        elif args.output_mode == "regression":
            preds = np.squeeze(preds)
            running_output = torch.cat(running_output).to(torch.float32)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)

        running_y_true = torch.cat(running_y_true)
        output_prediction_file = os.path.join(args.output_dir, "predictions_{}.out".format(eval_task))
        with open(output_prediction_file, "w") as writer:
            for i in range(len(running_output)):
                # number of labels == 1:
                output_str = "{}".format(running_output[i][0])
                if args.num_labels == 2:
                   output_str = "{} {}".format(running_output[i][0], running_output[i][1])
                elif args.num_labels == 3:
                   output_str = "{} {} {}".format(running_output[i][0], running_output[i][1], running_output[i][2])
                writer.write("{}\t{}\t{}\n".format(example_ids[i], output_str, running_y_true[i]))

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results_{}.txt".format(eval_task))
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    return results

def load_and_cache_examples(args, task, filename, tokenizer, cnn_stoi, evaluate=False): 
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    fstem =  list(filter(None,filename.split("/"))).pop() 
    fstem =  fstem.split(".")[0]
    data_type = task

    bert_model_name = None
    blist = list(filter(None, args.model_name_or_path.split("/")))
    for i in range(len(blist)):
       if 'roberta-' in blist[i] or 'bert-' in blist[i]:
           bert_model_name = blist[i]
    assert bert_model_name is not None

    feat_dir = args.feat_dir if args.feat_dir is not None else "."
    features_file = "cached_{}_{}_{}_{}_{}".format(
            fstem,
            bert_model_name,
            args.cnn_stem,
            list(filter(None, args.cnn_model.split("_"))).pop(),
            str(args.max_seq_length),
        )
    cached_features_file = os.path.join(
        args.feat_dir,
        data_type,
        features_file,
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
        logger.info("Successfully loading features")
        new_features = []
        unique_id = 1000000000
        example_index = 0
        # HERE ASSIGN EXAMPLE IDS
        for example_feature in tqdm(features, total=len(features), desc="add example index and unique id"):
            if not example_feature:
                continue
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
            example_index += 1
        features = new_features
        del new_features
    else:
        logger.info("cached_features_files is not existing: %s", cached_features_file)
        assert  os.path.exists(cached_features_file)
        return

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    logger.info("Converting features to Tensors")
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_qids = torch.tensor([f.input_qids for f in features], dtype=torch.long)
    for f in features:
        assert f.input_ids is not None
        assert f.input_qids is not None
        assert f.attention_mask is not None
        assert f.token_type_ids is not None
        assert f.label is not None

    if not (task == 'cola' or task == 'sst-2'): # sentence-pair tasks
       all_input_pids = torch.tensor([f.input_pids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression": # STS-B
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    if evaluate:
       all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
       if task == 'cola' or task == 'sst-2': # single-sentence tasks
          logger.info("Single-sentence task (evaluation): {}".format(task))
          dataset = TensorDataset(all_input_ids, all_input_qids, all_attention_mask, all_token_type_ids, all_labels, all_example_index)
       else: # sentence-pair tasks
          logger.info("Sentence-pair task (evaluation): {}".format(task))
          dataset = TensorDataset(all_input_ids, all_input_qids, all_input_pids, all_attention_mask, all_token_type_ids, all_labels, all_example_index)
       return dataset, features
    else: # training
       if task == 'cola' or task == 'sst-2': # single-sentence tasks
          logger.info("Single-sentence task (training): {}".format(task))
          dataset = TensorDataset(all_input_ids, all_input_qids, all_attention_mask, all_token_type_ids, all_labels)
       else: # sentence-pair tasks
          logger.info("Sentence-pair task (training): {}".format(task))
          dataset = TensorDataset(all_input_ids, all_input_qids, all_input_pids, all_attention_mask, all_token_type_ids, all_labels)
       return dataset

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
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum query sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    # added by Jong-Hoon Oh (use '--warmup_ratio' instead of '--warmup_steps')
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
    parser.add_argument("--save_checkpoints", action="store_true", help="SAVE Checkpoint?")
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    # arguments for BERTAC 
    parser.add_argument(
        "--prep_vocab_file",
        default=None,
        type=str,
        required=True,
        help="preprocessed_vocab_file with the train and predict file. see run_proprecess.py (it should be run before running this script)",
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
        help="The file name of CNN model",
    )
    parser.add_argument(
        "--cnn_stem",
        default="enwiki",
        type=str,
        required=True,
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
        help="number of TIER Layers",
    )
    parser.add_argument(
        "--emb_dim",
        default=None,
        type=int,
        help="dim for representation of fastText",
    )

    # fp16 
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
    parser.add_argument("--threads", type=int, default=1, help="multiple threads for converting example to features")
    args = parser.parse_args()
    assert args.prep_vocab_file is not None
    assert args.cnn_model is not None
    assert args.cnn_stem is not None
    assert args.emb_dim is not None
    assert args.emb_file is not None
    if args.fp16:
       assert args.fp16_opt_level == 'O0' or args.fp16_opt_level == 'O1' or args.fp16_opt_level == 'O2' or args.fp16_opt_level == 'O3' 

    if (not os.path.exists(args.prep_vocab_file)):
        raise ValueError(
            "prep_vocab_file ({}) does not exist. Check the --prep_vocab_file option.".format( args.prep_vocab_file) )
    
    if (not os.path.exists(args.cnn_model)):
        raise ValueError(
            "cnn_model ({}) does not exist. Check the --cnn_model option.".format( args.cnn_model) )

    if (not os.path.exists(args.emb_file)):
        raise ValueError(
            "emb_file ({}) does not exist. Check the --emb_file option.".format( args.emb_file) )

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

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    args.num_labels = num_labels
    logger.info("{}: {}".format(args.task_name, args.num_labels))

    # added by Jong-Hoon Oh
    # prep_vocab_file: see make_glue_cnn_vocab.py
    prep_tokens = torch.load(args.prep_vocab_file)
    all_tokens = prep_tokens['tokens']

    # Load pretrained CNNs
    cnn_model, cnn_stoi = load_cnn_model_and_vocab(args, args.cnn_model, all_tokens)
    cnn_dim = len(cnn_model.args.filter_widths) * cnn_model.args.filter_size
    args.cnn_dim = cnn_dim

    # Load pretrained TLM and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class4ms, model_class4ss, tokenizer_class = MODEL_CLASSES[args.model_type]
    # CONFIG 
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    config.num_of_TIERs = args.num_of_TIERs
    config.cnn_dim = args.cnn_dim
    config.emb_dim = args.emb_dim
    config.cnn_model = args.cnn_model
    config.cnn_stem = args.cnn_stem
    logger.info("num_labels of {}: {}".format(args.task_name, num_labels))

    # TOKENIZER
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    # MODEL
    if args.task_name == 'cola' or args.task_name == 'sst-2':
        model_class = model_class4ss # model for single-sentence tasks
    else:
        model_class = model_class4ms # model for sentence-pair tasks

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

    # Training
    if args.do_train:
       train_file = "train.tsv"
       # load_and_cache_examples simply gets all the data from cache
       train_dataset = load_and_cache_examples(args, args.task_name, train_file, tokenizer, cnn_stoi, evaluate=False)
       global_step, tr_loss = train(args, train_dataset, model, cnn_model, tokenizer)
       logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            result = evaluate(args, model, cnn_model, tokenizer, cnn_stoi, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
