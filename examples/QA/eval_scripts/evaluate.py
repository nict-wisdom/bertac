import unicodedata
import argparse
import torch
import numpy as np
import json
import os
import sys
import subprocess
import logging

from eval_utils import Timer, AverageMeter, metric_max_over_ground_truths, exact_match_score, f1_score
from utils import normalize

if sys.version_info < (3, 5):
    raise RuntimeError('Only supports Python 3.5 or higher.')

def load_pred(args):
    res = [] 
    for line in open(args.pred, encoding='utf8'):
        data = json.loads(line)
        answer = data['answer']
        res.append({"answer":answer})
    return res


def evaluation(args, predictions):
    eval_time = Timer()
    f1 = AverageMeter()
    exact_match = AverageMeter()
    idx = 0
    pred_stem = list(filter(None, args.pred.split("/"))).pop()
    gt_stem = list(filter(None, args.gt.split("/"))).pop()
    gt_stem2  = gt_stem.split(".")[0]
    for line in open(args.gt, encoding='utf8'):
        data = json.loads(line)
        answer = data['answers'] # orignal
        ground_truths = []
        prediction = predictions[idx]['answer']
    
        for a in answer:
            ground_truths.append(a)
        tmp_em = metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        tmp_f1 = metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        exact_match.update(tmp_em)
        f1.update(tmp_f1)
        idx += 1
    print("{}.{}: EM = {} F1 = {} | idx:{} | eval_time:{}".format(gt_stem2, pred_stem, exact_match.avg * 100, f1.avg * 100, idx, eval_time.time() ))

def main():

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--gt",
        default=None,
        type=str,
        required=True,
        help="ground truth",
    )
    parser.add_argument(
        "--pred",
        default=None,
        type=str,
        required=True,
        help="prediction data",
    )
    args = parser.parse_args()

    assert args.gt is not None
    assert args.pred is not None

    if (not os.path.exists(args.pred)):
        raise ValueError(
            "pred file ({}) does not exist.".format( args.pred ) )
    
    if (not os.path.exists(args.gt)):
        raise ValueError(
            "ground truth file ({}) does not exist.".format( args.gt ) )

    prediction = load_pred(args)
    evaluation(args, prediction)

if __name__ == "__main__":
    main()
