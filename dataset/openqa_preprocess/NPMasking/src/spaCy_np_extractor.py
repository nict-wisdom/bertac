# Copyright (c) 2021-present, Data-driven Intelligent System Research Center (DIRECT), National Institute of Information and Communications Technology (NICT).
# All rights reserved.
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

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--input_file",
        default=None,
        type=str,
        required=True,
        help="The input file.",
    )
    args = parser.parse_args()

    if (not os.path.exists(args.input_file)):
        raise ValueError(
            "input_file ({}) does not exist. Check the --input_file option.".format( args.input_file ) )

    nlp = spacy.load("en_core_web_md")
    input_lines = _read_tsv(args.input_file)
    examples = []
    for (i, line) in enumerate(input_lines):
       if len(line) < 2: # head
          continue
       cid = line[0]
       line[1] = ReplaceSpace(line[1])
       line[2] = ReplaceSpace(line[2])
       sent1 = nlp(line[1])
       sent2 = nlp(line[2])
       np1 = []
       sent1_tokens = []
       for token in sent1:
           sent1_tokens.append(token.text)
       sent1_str = " ".join(sent1_tokens)
       print("{}\tTOKENS\t1\t{}".format(cid,sent1_str))
       for np_idx in sent1.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np1_str = " ".join(tmp_np)
           print("{}\tNP\t1\t{}".format(cid,np1_str))
       np2 = []
       sent2_tokens = []
       for token in sent2:
           sent2_tokens.append(token.text)
       sent2_str = " ".join(sent2_tokens)
       print("{}\tTOKENS\t2\t{}".format(cid,sent2_str))
       for np_idx in sent2.noun_chunks:
           tmp_np = []
           for t in np_idx:
               tmp_np.append("{} {}".format(t.text, t.pos_))
           np2_str = " ".join(tmp_np)
           print("{}\tNP\t2\t{}".format(cid,np2_str))
       print("END")

if __name__ == "__main__":
    main()

