#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Main OpenQA training and testing script."""
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


from utils import normalize


from corenlp_tokenizer import CoreNLPTokenizer
corenlp_classpath = '.'
PROCESS_TOK = None

if sys.version_info < (3, 5):
    raise RuntimeError('Only supports Python 3.5 or higher.')

import unicodedata

logger = logging.getLogger()


def add_train_args(parser):
    """Adds commandline arguments pertaining to training a model. These
    are different from the arguments dictating the model architecture.
    """

    # Runtime environment
    runtime = parser.add_argument_group('Environment')
    runtime.add_argument('--dataset', type=str, default="searchqa",
                         help='Dataset: searchqa, quasart or unftriviaqa')
    runtime.add_argument('--base_dir', type=str, default=".",
                         help='base_dir of the pre-processing')

def load_data_with_doc(args, filename):
    """Load examples from preprocessed file.
    One example per line, JSON encoded.
    Here, one example means 'question, list of documents, ids
    """
    # Load JSON lines
    res = []
    keys = set()
    step =0
    num_docs = 100 # the maximum number of passages for each question 
    with open(filename, encoding='utf8') as f:
        for line in f:
            ex = json.loads(line) 
            step+=1
            try:
                question = " ".join(ex[0]['question'])
            except:
                logger.info(step)
                logger.info(ex)
                continue

            tmp_res = []
            for i in range(len(ex)):
                # ignore documents with less than 5 words
                # truncate documents with 300 words
                if (len(ex[i]['document'])>5):
                    if len(ex[i]['document']) > 300:
                       ex[i]['document'] = ex[i]['document'][:300]
                    if len(ex[i]['npm_document']) > 300:
                       ex[i]['npm_document'] = ex[i]['npm_document'][:300]
                    tmp_res.append(ex[i])
            if (len(tmp_res)<num_docs):
                len_tmp_res = len(tmp_res)
            tmp_res = sorted(tmp_res, key=lambda x:len(x['document'])) # sort with the length
            assert(len(tmp_res)!=0)
            res.append(tmp_res)
            keys.add(question)
    return res, keys

def has_answer(args, answer, t):
    text = [w.lower() for w in t] # lower input text
    res_list = []
    for a in answer: # answers, quasart and searchqa 
        single_answer = " ".join(a).lower()
        single_answer = normalize(single_answer)
        single_answer = tokenize_text(single_answer).words()
        single_answer = [w.lower() for w in single_answer]
        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i: i + len(single_answer)]:
                res_list.append((i, i+len(single_answer)-1)) 
    if (len(res_list)>0):
        return True, res_list 
    else:
        return False, res_list

def has_answer_prep(args, answer, t):
    text = [w.lower() for w in t] # lower input text
    res_list = []
    for a in answer: # multiple answers 
        lower_text  = " ".join(text)
        orig_text  = " ".join(t)
        single_answer = " ".join(a).lower()
        single_answer = normalize(single_answer)
        single_answer = tokenize_text(single_answer).words()
        single_answer = [w.lower() for w in single_answer]
        orig_answer = " ".join(single_answer)
        word2char = []
        cur = 0
        for i in range(len(text)):
            word2char.append(cur)
            cur += len(text[i]) +1 

        for i in range(0, len(text) - len(single_answer) + 1):
            if single_answer == text[i: i + len(single_answer)]:
                start_position = word2char[i]
                end_position = start_position + len(orig_answer)
                extracted_answer = lower_text[start_position:end_position]
                extracted_answer_orig = orig_text[start_position:end_position]
                if(extracted_answer == orig_answer):
                   res_list.append((extracted_answer_orig, i, i+len(single_answer)-1, start_position, end_position)) 
                else:
                   print("the extracted/orig answer are different: [{}], [{}]".format(orig_answer, extracted_answer))

    if (len(res_list)>0):
        return False, res_list 
    else:
        res_list.append(("null",-1,-1,-1,-1))
        return True, res_list

# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------

def read_data(filename, args):
    res = []
    for line in open(filename, encoding='utf8'):
        data = json.loads(line)
        #print("ANSWERS: {}".format(data['answers']))
        answer = [tokenize_text(a).words() for a in data['answers']]
        question = " ".join(tokenize_text(data['question']).words())
        res.append({"answer":answer, "question":question})

        has_answer(args, answer, [])    # update TOKENIZER_CACHE
    return res
    

TOKENIZER_CACHE = {}
def tokenize_text(text):
    global PROCESS_TOK
    if text not in TOKENIZER_CACHE:
        assert PROCESS_TOK is not None
        TOKENIZER_CACHE[text] = PROCESS_TOK.tokenize(text)
    return TOKENIZER_CACHE[text]

def answer_data(args, docs, exs, tag):
    assert len(docs) ==  len(exs)
    pairs = []
    data_logs = []
    for i in range(len(docs)): # x docs per question
        HasAnswer_list = []
        for idx_doc in range(len(docs[i])):
            HasAnswer = []
            question = docs[i][idx_doc%len(docs[i])]["question"]
            document = docs[i][idx_doc%len(docs[i])]["document"]
            npm_question = docs[i][idx_doc%len(docs[i])]["npm_question"]
            npm_document = docs[i][idx_doc%len(docs[i])]["npm_document"]
            answer = exs[i]['answer']
            answer_position = has_answer_prep(args, answer, document)

            tmp_str = tag + "\t" + str(i) + "\t" + str(idx_doc) + "\t" + str(answer_position[0]) 
            data_logs.append(tmp_str)
            summary = {}
            summary['context'] = " ".join(document)
            summary['npm_context'] = " ".join(npm_document)
            all_answers = []
            for j in range(len(answer_position[1])):
               each_answer = {'text': answer_position[1][j][0], 'answer_start': answer_position[1][j][3]}
               all_answers.append(each_answer)
            summary['qas'] = [{'answers': all_answers, 'id': [i, idx_doc], 'question': " ".join(question), 'npm_question': " ".join(npm_question),'is_impossible': answer_position[0]}]
            paragraph = []
            paragraph.append(summary)
            instance = {"title":'null', "paragraphs": paragraph}
            pairs.append(instance)

    data = {"data": pairs}
    return data, data_logs

def main(args):
    # --------------------------------------------------------------------------
    # TOK 
    # corenlp v3.8: following OpenQA's main.py, we use corenlp v3.8.
    corenlp_classpath = "{}/download/stanford-corenlp-full-2017-06-09/*".format(args.base_dir)
    global PROCESS_TOK
    PROCESS_TOK = CoreNLPTokenizer(classpath=corenlp_classpath)

    # DATA
    logger.info('-' * 100)
    logger.info('Load data files')
    dataset = args.dataset #'quasart'#'searchqa'#'toy3'
    dset = ['train', 'dev', 'test']
    for fstem in dset:
       fname_docs = "{}/NPMasking/{}/{}.npm.tmp".format(args.base_dir, dataset, fstem)
       fname_answers = "{}/download/{}/{}.txt".format(args.base_dir, dataset, fstem)
       if not os.path.isfile(fname_docs):
           logger.info("{} is not existing\n".format(fname_docs))
       if not os.path.isfile(fname_answers):
           logger.info("{} is not existing\n".format(fname_answers))
       # read the documents
       docs, questions = load_data_with_doc(args, fname_docs) 
       logger.info(len(docs))
       # read the answers
       exs_with_doc = read_data(fname_answers, args) 
       logger.info('Num %s examples = %d' % (fstem,len(exs_with_doc)))
       # validation
       logger.info('Validating')
       odir = "{}/NPMasking/{}".format(args.base_dir,dataset)
       filename_out_log = "{}/{}_{}.npm.json.log".format(odir,dataset,fstem)
       v_data, v_log = answer_data(args, docs, exs_with_doc, fstem) # match answer with doc
       with open(filename_out_log, 'w', encoding='utf8') as f:
           for i in range(len(v_log)):
              f.write(v_log[i]+"\n")
       # store the results in a JSON format
       filename_out = "{}/{}_{}.npm.json".format(odir,dataset,fstem)
       print("output_file: {}".format(filename_out))
       with open(filename_out, 'w', encoding='utf8') as f:
	       json.dump(v_data, f)

    PROCESS_TOK = None

if __name__ == '__main__':
    # Parse cmdline args and setup environment
    parser = argparse.ArgumentParser(
        'converting *.npm.tmp to JSON file in the SQuAD format',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add_train_args(parser)
    args = parser.parse_args()

    # Run!
    main(args)
