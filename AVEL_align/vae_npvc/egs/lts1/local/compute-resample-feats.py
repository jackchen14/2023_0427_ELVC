#!/usr/bin/env python3

# Copyright 2021 Academia Sinica (Pin-Jui Ku, Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
from distutils.util import strtobool
import logging

import kaldiio
import numpy
import resampy

from PIL import Image

from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_readers import file_reader_helper
from espnet.utils.cli_writers import file_writer_helper
from espnet2.utils.types import int_or_none


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute resampling features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interp-mode", type=str, help="Interpolation mode", default="bilinear"
    )
    parser.add_argument(
        "--write-num-frames", type=str, help="Specify wspecifer for utt2num_frames"
    )
    parser.add_argument(
        "--filetype",
        type=str,
        default="mat",
        choices=["mat", "hdf5"],
        help="Specify the file format for output. "
        '"mat" is the matrix format in kaldi',
    )
    parser.add_argument(
        "--compress", type=strtobool, default=False, help="Save in compressed format"
    )
    parser.add_argument(
        "--compression-method",
        type=int,
        default=2,
        help="Specify the method(if mat) or " "gzip-level(if hdf5)",
    )
    parser.add_argument("--verbose", "-V", default=1, type=int, help="Verbose option")
    parser.add_argument("target_utt2num_frames", type=str, help="Target utt2num_frames file")
    parser.add_argument("rspecifier", type=str, help="Input specifier")
    parser.add_argument("wspecifier", type=str, help="Output specifier")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    interp_mode = getattr(Image, args.interp_mode.upper())

    target_utt2num_frames = [l.strip().split() for l in open(args.target_utt2num_frames)]
    target_utt2num_frames = dict([[utt,int(nframes)] for utt,nframes in target_utt2num_frames])
    
    # Read, extract and write.
    with file_reader_helper(args.rspecifier, args.filetype) as reader, \
        file_writer_helper(
            args.face_wspecifier,
            filetype=args.filetype,
            write_num_frames=args.write_num_frames,
            compress=args.compress,
            compression_method=args.compression_method,
        ) as writer:

        # Loop start.
        for utt_id, feat_in in reader:
            logging.info("Resampling features of {}...".format(utt_id))

            target_nframes = target_utt2num_frames[utt_id]

            for i in range(feat_in.shape[1]):
                feat_dim = numpy.round(feat_in[:,i].reshape(-1,1)).astype(numpy.uint8)
                feat_out = numpy.array(
                    Image.fromarray(feat_dim).resize(
                        (feat_dim.shape[1],target_nframes), interp_mode 
                    ),
                    dtype=np.float
                )

            # write resampled feature
            writer[utt_id] = feat_out


if __name__ == "__main__":
    main()
