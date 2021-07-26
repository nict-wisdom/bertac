#!/usr/bin/env python3
# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.

import logging
import torch
import cnn_utils
from Gmodel import Generator

logger = logging.getLogger()

def load_cnn_model(args, cnn_file, fields):
    logger.info("Loading cnn model parameters from '{}'".format(
        cnn_file))
    params = torch.load(
        cnn_file, map_location=lambda storage, loc: storage)

    model = Generator(params['args'])
    model.load_state_dict(params['G_model_state_dict'])
    pre_fields = cnn_utils.build_fields(params['field_vocab'])


    if args.emb_file:
        vocab = pre_fields['text'].vocab
        words = fields['text'].vocab.itos
        tgt_words = cnn_utils.extend_model_vocab(args, model, 0, vocab, words)
        args.cnn_vocab_size = len(vocab)
        cnn_utils.set_emb(args.emb_file, model, 0, vocab, tgt_words)

    for p in model.parameters():
       p.requires_grad = False

    return model, pre_fields


def build_vocab(args, fields, train_data, val_data):
    logger.info("Building vocab (min_freq={})...".format(args.min_freq))
    fields['text'].build_vocab(train_data, val_data, min_freq=args.min_freq)
    
    return


