#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2021 Nagoya University (Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from io import open
import json
import logging
import sys

from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description="Merge source and target data.json files into one json file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--face-json", type=str, help="Json file with the face features"
    )
    parser.add_argument(
        "--fbank-json",
        type=str,
        help="Json file with the fbank features",
    )
    parser.add_argument(
        "--num_utts", default=-1, type=int, help="Number of utterances (take from head)"
    )
    parser.add_argument("--verbose", "-V", default=1, type=int, help="Verbose option")
    parser.add_argument(
        "--out",
        "-O",
        type=str,
        help="The output filename. " "If omitted, then output to sys.stdout",
    )
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    with open(args.face_json, "rb") as f:
        face_json = json.load(f)["utts"]
    with open(args.fbank_json, "rb") as f:
        fbank_json = json.load(f)["utts"]

    count = 0
    data = {"utts": {}}
    # (dirty) loop through input only because in/out should have same files
    for utt_id, v in face_json.items():

        entry = {"input": face_json[utt_id]["input"]}

        entry["output"] = fbank_json[utt_id]["input"]
        entry["output"][0]["name"] = "target1"

        data["utts"][utt_id] = entry
        count += 1
        if args.num_utts > 0 and count >= args.num_utts:
            break

    if args.out is None:
        out = sys.stdout
    else:
        out = open(args.out, "w", encoding="utf-8")

    json.dump(
        data,
        out,
        indent=4,
        ensure_ascii=False,
        separators=(",", ": "),
    )
