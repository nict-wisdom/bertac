#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Modified from main.py of OpenQA (https://github.com/thunlp/OpenQA/blob/master/main.py)"""

import argparse
import torch
import numpy as np
import json
import os
import sys
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
    runtime.add_argument('--dataset', type=str, default="quasart",
                         help='dataset: searchqa or quasart')
    runtime.add_argument('--base_dir', type=str, default=".",
                         help='base_dir of the pre-processing')
    runtime.add_argument('--datatype', type=str, default="qd",
                         help='datatype: qd (question-document) or qa (question-answer)')

def load_data_with_doc(args, filename, ofilename):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    Here, one example means 'question, list of documents, ids
    """
    # Load JSON lines
    num_docs = 100 # the maximum number of passages for each question 
    with open(ofilename, 'w', encoding='utf8') as fout:
        with open(filename, encoding='utf8') as f:
            for line in f:
                ex = json.loads(line) 
                if args.datatype == 'qd':
                   try:
                       question = " ".join(ex[0]['question'])
                   except:
                       continue
                   for i in range(len(ex)):
                       # ignore documents with less than 2 words and
                       # truncate documents with 300 words
                       if len(ex[i]['document']) > 2:
                          if len(ex[i]['document']) > 300:
                             ex[i]['document'] = ex[i]['document'][:300]
                          id_str = "{}".format(":::".join(str(value) for value in ex[i]['id']))
                          question  = " ".join(ex[i]['question'])
                          document  = " ".join(ex[i]['document'])
                          fout.write("{}\t{}\t{}\n".format(id_str, question, document))
                else:
                   try:
                       question = ex['question']
                       fout.write("{}\n".format(question))
                   except:
                       continue
   
def main(args):

    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    dataset = args.dataset #'quasart', 'searchqa'
    # input file: 
    #  - ../download/[quasart|searchqa]/*.json for datatype == 'qd'
    #  - ../download/[quasart|searchqa]/*.txt  for datatype == 'qa'
    dset = ['train', 'dev', 'test']
    if args.datatype == 'qd':
       for fstem in dset:
           filename  = args.base_dir+"/download/"+dataset+"/"+fstem+".json" 
           ofilename = args.base_dir+"/NPMasking/"+dataset+"/"+fstem+".json.txt" 
           load_data_with_doc(args, filename, ofilename)
    else:
       for fstem in dset:
           filename  = args.base_dir +"/download/"+dataset+"/"+fstem+".txt" 
           ofilename = args.base_dir +"/NPMasking/"+dataset+"/"+fstem+".q" 
           load_data_with_doc(args, filename, ofilename)

if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'Script for converting JSON to TXT',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_args(parser)
    args = parser.parse_args()
    assert args.datatype == 'qd' or args.datatype == 'qa'

    # Run!
    main(args)
