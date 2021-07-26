# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.
#!/usr/bin/env python3

import io
import argparse
import sys
import csv
import os
import spacy
import re

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

def ReplaceSpace(input_str):
    r = re.sub('[ ]+', ' ', input_str)
    r = re.sub('^[ ]', '', r)
    r = re.sub('[ ]$', '', r)
    return r

def CoLA_np(args):
    nlp = spacy.load("en_core_web_md")
    input_lines = _read_tsv(args.input_file)
    examples = []
    for (i, line) in enumerate(input_lines):
       if len(line) < 4: # head
          continue
       line[3] = ReplaceSpace(line[3])
       sent1 = nlp(line[3])
       np1 = []
       sent1_tokens = []
       for token in sent1:
           sent1_tokens.append(token.text)
       sent1_str = " ".join(sent1_tokens)
       print("{}\tTOKENS\t1\t{}".format(i,sent1_str))
       for np_idx in sent1.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np1_str = " ".join(tmp_np)
           print("{}\tNP\t1\t{}".format(i,np1_str))
       print("END")

def MNLI_np(args):
    nlp = spacy.load("en_core_web_md")
    input_lines = _read_tsv(args.input_file)
    examples = []
    for (i, line) in enumerate(input_lines):
       if i == 0: # head
          continue
       line[8] = ReplaceSpace(line[8])
       line[9] = ReplaceSpace(line[9])
       sent1 = nlp(line[8])
       sent2 = nlp(line[9])
       np1 = []
       np2 = []
       sent1_tokens = []
       sent2_tokens = []
       for token in sent1:
           sent1_tokens.append(token.text)
       for token in sent2:
           sent2_tokens.append(token.text)
       sent1_str = " ".join(sent1_tokens)
       sent2_str = " ".join(sent2_tokens)
       print("{}\tTOKENS\t1\t{}".format(line[0],sent1_str))
       print("{}\tTOKENS\t2\t{}".format(line[0],sent2_str))
       for np_idx in sent1.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np1_str = " ".join(tmp_np)
           print("{}\tNP\t1\t{}".format(line[0],np1_str))
       for np_idx in sent2.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np2_str = " ".join(tmp_np)
           print("{}\tNP\t2\t{}".format(line[0],np2_str))
       print("END")

def MRPC_np(args):
    nlp = spacy.load("en_core_web_md")
    input_lines = _read_tsv(args.input_file)
    examples = []
    for (i, line) in enumerate(input_lines):
       if i == 0: # head
          continue
       sent1 = nlp(line[3])
       sent2 = nlp(line[4])
       np1 = []
       np2 = []
       sent1_tokens = []
       sent2_tokens = []
       for token in sent1:
           sent1_tokens.append(token.text)
       for token in sent2:
           sent2_tokens.append(token.text)
       sent1_str = " ".join(sent1_tokens)
       sent2_str = " ".join(sent2_tokens)
       print("{}\tTOKENS\t1\t{}".format(i,sent1_str))
       print("{}\tTOKENS\t2\t{}".format(i,sent2_str))
       for np_idx in sent1.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np1_str = " ".join(tmp_np)
           print("{}\tNP\t1\t{}".format(i,np1_str))
       for np_idx in sent2.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np2_str = " ".join(tmp_np)
           print("{}\tNP\t2\t{}".format(i,np2_str))
       print("END")

def QNLI_np(args):
    nlp = spacy.load("en_core_web_md")
    input_lines = _read_tsv(args.input_file)
    examples = []
    for (i, line) in enumerate(input_lines):
       if i == 0: # head
          continue
       if len(line) < 2:
          continue
       line[1] = ReplaceSpace(line[1])
       line[2] = ReplaceSpace(line[2])
       sent1 = nlp(line[1])
       sent2 = nlp(line[2])
       np1 = []
       np2 = []
       sent1_tokens = []
       sent2_tokens = []
       for token in sent1:
           sent1_tokens.append(token.text)
       for token in sent2:
           sent2_tokens.append(token.text)
       sent1_str = " ".join(sent1_tokens)
       sent2_str = " ".join(sent2_tokens)
       print("{}\tTOKENS\t1\t{}".format(line[0],sent1_str))
       print("{}\tTOKENS\t2\t{}".format(line[0],sent2_str))
       for np_idx in sent1.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np1_str = " ".join(tmp_np)
           print("{}\tNP\t1\t{}".format(line[0],np1_str))
       for np_idx in sent2.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np2_str = " ".join(tmp_np)
           print("{}\tNP\t2\t{}".format(line[0],np2_str))
       print("END")

def QQP_np(args):
    nlp = spacy.load("en_core_web_md")
    input_lines = _read_tsv(args.input_file)
    examples = []
    for (i, line) in enumerate(input_lines):
       if i == 0: # head
          continue
       if len(line) < 6: # head
          continue
       line[3] = ReplaceSpace(line[3])
       line[4] = ReplaceSpace(line[4])
       sent1 = nlp(line[3])
       sent2 = nlp(line[4])
       #print('NER:', nlp.ner(line[3]))
       np1 = []
       np2 = []
       sent1_tokens = []
       sent2_tokens = []
       for token in sent1:
           sent1_tokens.append(token.text)
       for token in sent2:
           sent2_tokens.append(token.text)
       sent1_str = " ".join(sent1_tokens)
       sent2_str = " ".join(sent2_tokens)
       print("{}\tTOKENS\t1\t{}".format(line[0],sent1_str))
       print("{}\tTOKENS\t2\t{}".format(line[0],sent2_str))
       for np_idx in sent1.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np1_str = " ".join(tmp_np)
           print("{}\tNP\t1\t{}".format(line[0],np1_str))
       for np_idx in sent2.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np2_str = " ".join(tmp_np)
           print("{}\tNP\t2\t{}".format(line[0],np2_str))
       print("END")

def RTE_np(args):
    nlp = spacy.load("en_core_web_md")
    input_lines = _read_tsv(args.input_file)
    examples = []
    for (i, line) in enumerate(input_lines):
       if i == 0: # head
          continue
       sent1 = nlp(line[1])
       sent2 = nlp(line[2])
       np1 = []
       np2 = []
       sent1_tokens = []
       sent2_tokens = []
       for token in sent1:
           sent1_tokens.append(token.text)
       for token in sent2:
           sent2_tokens.append(token.text)
       sent1_str = " ".join(sent1_tokens)
       sent2_str = " ".join(sent2_tokens)
       print("{}\tTOKENS\t1\t{}".format(line[0],sent1_str))
       print("{}\tTOKENS\t2\t{}".format(line[0],sent2_str))
       for np_idx in sent1.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np1_str = " ".join(tmp_np)
           print("{}\tNP\t1\t{}".format(line[0],np1_str))
       for np_idx in sent2.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np2_str = " ".join(tmp_np)
           print("{}\tNP\t2\t{}".format(line[0],np2_str))
       print("END")

def SST2_np(args):
    nlp = spacy.load("en_core_web_md")
    input_lines = _read_tsv(args.input_file)
    examples = []
    for (i, line) in enumerate(input_lines):
       if i == 0: # head
          continue
       if len(line) < 2: # head
          continue
       line[0] = ReplaceSpace(line[0])
       sent1 = nlp(line[0])
       np1 = []
       sent1_tokens = []
       for token in sent1:
           sent1_tokens.append(token.text)
       sent1_str = " ".join(sent1_tokens)
       print("{}\tTOKENS\t1\t{}".format(i,sent1_str))
       for np_idx in sent1.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np1_str = " ".join(tmp_np)
           print("{}\tNP\t1\t{}".format(i,np1_str))
       print("END")

def STSB_np(args):
    nlp = spacy.load("en_core_web_md")
    input_lines = _read_tsv(args.input_file)
    examples = []
    for (i, line) in enumerate(input_lines):
       if i == 0: # head
          continue
       sent1 = nlp(line[7])
       sent2 = nlp(line[8])
       np1 = []
       np2 = []
       sent1_tokens = []
       sent2_tokens = []
       for token in sent1:
           sent1_tokens.append(token.text)
       for token in sent2:
           sent2_tokens.append(token.text)
       sent1_str = " ".join(sent1_tokens)
       sent2_str = " ".join(sent2_tokens)
       print("{}\tTOKENS\t1\t{}".format(line[0],sent1_str))
       print("{}\tTOKENS\t2\t{}".format(line[0],sent2_str))
       for np_idx in sent1.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np1_str = " ".join(tmp_np)
           print("{}\tNP\t1\t{}".format(line[0],np1_str))
       for np_idx in sent2.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np2_str = " ".join(tmp_np)
           print("{}\tNP\t2\t{}".format(line[0],np2_str))
       print("END")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        required=True,
        help="The input file.",
    )
    parser.add_argument(
        "--task",
        default=None,
        type=str,
        required=True,
        help="GLUE task name.",
    )
    args = parser.parse_args()

    if (not os.path.exists(args.input_file)):
        raise ValueError(
            "input_file ({}) does not exist. Check the --input_file option.".format( args.input_file ) )

    assert args.task in ['CoLA', 'MNLI', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B']
    if args.task == 'CoLA':
       CoLA_np(args)
    if args.task == 'MNLI':
       MNLI_np(args)
    if args.task == 'MRPC':
       MRPC_np(args)
    if args.task == 'QNLI':
       QNLI_np(args)
    if args.task == 'QQP':
       QQP_np(args)
    if args.task == 'RTE':
       RTE_np(args)
    if args.task == 'SST-2':
       SST2_np(args)
    if args.task == 'STS-B':
       STSB_np(args)

if __name__ == "__main__":
    main()

