#!/usr/bin/env python3
# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.

import logging
import os
import torch
import torch.nn as nn
import torchtext
from collections import defaultdict
from tqdm import tqdm
import torch.optim as optim

logger = logging.getLogger(__name__)
optimizer = {
    'adam': optim.Adam,
    'rmsprop': optim.RMSprop,
}


def get_fields():
    fields = {}
    fields['label'] = torchtext.data.Field(sequential=False, batch_first=True)
    fields['text'] = torchtext.data.Field(batch_first=True)
    return fields

def build_fields(vocab):
    fields = get_fields()
    for k, v in dict(vocab).items():
        v.stoi = defaultdict(lambda: 0, v.stoi)
        fields[k].vocab = v
    return fields

def set_emb(filename, model, eidx, vocab, tgt_words):
    ext = os.path.splitext(filename)[1][1:]
    if ext == 'gz':
        raise RuntimeError("Got a compressed file '{}', expected a plain text "
                           "file".format(filename))

    tgt_words = set([w for w in tgt_words if w in vocab.stoi])
    if len(tgt_words) == 0:
        logger.warning("No target words to set")
        return

    logger.info("Loading pre-trained embeddings from '{}' for {} words".format(
        filename, len(tgt_words)))
    with open(filename, 'rb') as f:
        lines = [line for line in f]

    dim, cnt = None, 0
    emb = model.embs[eidx].weight.data
    for line in tqdm(lines, total=len(lines)):
        try:
            # Python 2
            w, entries = line.rstrip().split(' ', 1)
            w = w.decode('utf-8', errors='ignore')
        except TypeError:
            # Python 3
            line = line.decode('utf-8', errors='ignore')
            w, entries = line.rstrip().split(' ', 1)
        if w in tgt_words:
            entries = entries.split(' ')
            if dim is None and len(entries) > 1:
                dim = len(entries)
                assert dim == emb.size(1)
            elif len(entries) == 1:
                logger.info("Skip word '{}' with 1-dimensional "
                            "vector {}".format(w, entries))
                continue
            elif dim != len(entries):
                raise RuntimeError(
                    "Vector for word '{}' has '{}' dimensions, expected "
                    "'{}' dimensions".format(w, len(entries), dim))
            v = torch.Tensor([float(x) for x in entries])
            emb[vocab.stoi[w]].copy_(v)
            cnt += 1

    assert cnt > 0
    logging.info("Found {:.2f}% ({}/{})".format(
        100.0 * cnt / len(tgt_words), cnt, len(tgt_words)))


def extend_model_vocab(args, model, eidx, vocab, words):
    logger.info("Extending model vocab...")
    emb_words = set()
    with open(args.emb_file, 'rb') as f:
        for line in f:
            try:
                # Python 2
                w, entries = line.rstrip().split(' ', 1)
                w = w.decode('utf-8', errors='ignore')
            except TypeError:
                # Python 3
                line = line.decode('utf-8', errors='ignore')
                w, entries = line.rstrip().split(' ', 1)
            emb_words.add(w)

    rwords = [w for w in words if w in emb_words]

    logger.info("{} words are in pre-trained embeddings".format(len(rwords)))

    tgt_words = set()
    for w in rwords:
        if w not in vocab.stoi: # if the word is not in vocab
            vocab.itos.append(w)
            vocab.stoi[w] = len(vocab.itos) - 1
            tgt_words.add(w)

    if len(tgt_words) > 0:
        logger.info("{} words are new and added to model vocab".format(
            len(tgt_words)))
        if eidx == 0:
           model.args.vocab_size = len(vocab)
           cur_vocab_size = model.args.vocab_size
           cur_emb_dim = model.args.emb_dim
           cur_padding_idx = model.args.padding_idx

        old_emb = model.embs[eidx % 2].weight.data
        model.embs[eidx % 2] = nn.Embedding(cur_vocab_size,
                                     cur_emb_dim,
                                     cur_padding_idx)
        model.embs[eidx % 2].weight.data[:old_emb.size(0)] = old_emb
    return tgt_words

def get_emb_params(args):
    vocab_sizes = [args.vocab_size]
    emb_dims = [args.emb_dim]
    padding_idcs = [args.padding_idx]
    emb_params = list(zip(vocab_sizes, emb_dims, padding_idcs))
    return emb_params
