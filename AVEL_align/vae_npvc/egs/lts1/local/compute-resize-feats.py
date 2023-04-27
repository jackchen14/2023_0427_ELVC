#!/usr/bin/env python3

# Copyright 2021 Academia Sinica (Pin-Jui Ku, Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import argparse
from distutils.util import strtobool
import logging

import cv2
import kaldiio
import numpy
import resampy
from PIL import Image

from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_writers import file_writer_helper
from espnet2.utils.types import int_or_none


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute face feature from video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target_utt2num_frames", type=str, default=None, help="Target utt2num_frames file"
    )
    parser.add_argument(
        "--interp_mode", type=str, help="Interpolation mode", default="bilinear"
    )
    parser.add_argument("--lip_width", type=int, help="Width of the output lip frame")
    parser.add_argument("--lip_height", type=int, help="Height of the output lip frame")
    parser.add_argument("--target_width", type=int, help="Width of the output target frame")
    parser.add_argument("--target_height", type=int, help="Height of the output target frame")
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
    parser.add_argument(
        "--normalize",
        type=bool,
        default=True,
        help="Normalizes image data to scale in [-1,1]",
    )
    parser.add_argument("rspecifier", type=str, help="video scp file")
    parser.add_argument("wspecifier", type=str, help="Write face specifier")
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

    # Check target utt2num_frames
    interp_mode = getattr(Image, args.interp_mode.upper())
    if args.target_utt2num_frames is not None and os.path.exists(args.target_utt2num_frames):
        target_utt2num_frames = [l.strip().split() for l in open(args.target_utt2num_frames)]
        target_utt2num_frames = dict([[utt,int(nframes)] for utt,nframes in target_utt2num_frames])
    else:
        target_utt2num_frames = None
 
    # Build lip extractor object, and defice desired lip size.
    target_size = (args.target_width, args.target_height)

    # Read, extract and write.
    with kaldiio.ReadHelper(args.rspecifier) as video_scp, \
        file_writer_helper(
            args.face_wspecifier,
            filetype=args.filetype,
            write_num_frames=args.write_num_frames,
            compress=args.compress,
            compression_method=args.compression_method,
        ) as f_writer:

        # Loop start.
        for utt_id, lip_features in video_scp:
            logging.info("Resizing lip features of {}...".format(utt_id))

            if args.normalize:
                lip_features = (lip_features + 1) * 128

            lip_features = lip_features.reshape((-1,lip_height,lip_width))
            lip_features = np.array([
                cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)
                for frame in lip_features
            ])

            if args.normalize:
                lip_features = lip_features / 128 - 1

            # write lip feature
            f_writer[utt_id] = lip_features

def temporal_resize(img, target_length, interp_mode):
    target_dim = img.shape[1]
    img_new = numpy.zeros((target_length, target_dim), dtype=numpy.float)
    for i in range(target_dim):
        tmp = numpy.round(img[:,i]).astype(numpy.uint8).reshape((-1,1))
        img_new[:,i] = numpy.array(
            Image.fromarray(tmp).resize(
                (1, target_length), interp_mode 
            ),
            dtype=numpy.float
        ).reshape(-1)
    return img_new


if __name__ == "__main__":
    main()
