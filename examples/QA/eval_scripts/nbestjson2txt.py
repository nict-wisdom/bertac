#!/usr/bin/env python3
# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.

import argparse
import json
import glob
import logging
import os
import random
import timeit

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="input_file",
    )
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        required=True,
        help="input_file",
    )
    parser.add_argument(
        "--output_file",
        default=None,
        type=str,
        required=True,
        help="input_file",
    )
    args = parser.parse_args()
    fname = os.path.join(args.data_dir, args.input_file)
    print("fname: {}".format(fname))
    with open(
         os.path.join(args.data_dir, args.input_file), "r", encoding="utf-8"
    ) as reader:
      ex = json.load(reader)

    ofname = os.path.join(args.data_dir, args.output_file)
    print("ofname: {}".format(ofname))
    with open(ofname, "w", encoding="utf-8") as writer:
      for idx, ids in enumerate(ex):
          topn = len(ex[ids]) if len(ex[ids]) < 2 else 2
          for n in range(topn):
              if n <= 2:
                 ostr = ids + '\t' + str(n) + '\t' + ex[ids][n]['text'] + '\t' + str(ex[ids][n]['probability'])
                 writer.write(ostr + '\n')

if __name__ == "__main__":
    main()
