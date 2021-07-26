#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Modified from main.py of OpenQA (https://github.com/thunlp/OpenQA/blob/master/main.py)"""

import argparse
import torch
import numpy as np
import json
import os
import sys
import csv
import subprocess
import logging

import regex as re

if sys.version_info < (3, 5):
    raise RuntimeError('Only supports Python 3.5 or higher.')

import unicodedata

logger = logging.getLogger()

def add_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """
    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--dataset', type=str, default="searchqa",
                         help='Dataset: searchqa, quasart or unftriviaqa')
    runtime.add_argument('--base_dir', type=str, default=".",
                         help='base_dir of the pre-processing')

def _read_tsv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        return list(csv.reader(f, delimiter="\t", quotechar=quotechar))


def load_data_with_doc(args, filename, ofilename):
    """Load examples from preprocessed file.
    One example per line in a tsv format.
    """
    # Load JSON lines
    res = []
    keys = set()
    step =0
    num_docs = 100 # the maximum number of passages for each question 
    input_lines = _read_tsv(filename)
    examples = []
    with open(ofilename, 'w', encoding='utf8') as fout:
       for (i, line) in enumerate(input_lines):
          if len(line) < 2: # END
             assert line[0] == 'END'
             json.dump(examples, fout)
             fout.write("\n")
             examples = []
             continue

          assert len(line) == 6
          id1 = line[0]
          id2 = line[1]
          q = line[2].split(' ')
          d = line[3].split(' ')
          npq = line[4].split(' ')
          npd = line[5].split(' ')
          each_ex = {'id': [str(id1),int(id2)], 'question': q, 'document': d , 'npm_question': npq, 'npm_document': npd}
          examples.append(each_ex)

def main(args):
    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    dataset = args.dataset #'quasart'#'searchqa'
    dset = ['train', 'dev', 'test']
    for fstem in dset:
        filename = "{}/NPMasking/{}/{}.npm".format(args.base_dir,dataset,fstem) 
        ofilename = "{}/NPMasking/{}/{}.npm.tmp".format(args.base_dir,dataset,fstem) 
        if os.path.isfile(filename):
           load_data_with_doc(args, filename, ofilename)

if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Script for converting TSV to JSON',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_args(parser)
    args = parser.parse_args()

    # Run!
    main(args)
