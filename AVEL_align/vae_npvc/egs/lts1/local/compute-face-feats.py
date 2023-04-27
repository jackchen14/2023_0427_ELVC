#!/usr/bin/env python3

# Copyright 2021 Academia Sinica (Pin-Jui Ku, Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import argparse
from distutils.util import strtobool
import logging

import kaldiio
import numpy
import resampy
from PIL import Image

from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_writers import file_writer_helper
from espnet2.utils.types import int_or_none

from video_transform import Lip_Extractor, load_video


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute face feature from video",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fps", type=int_or_none, help="Video Sampling frequency")
    parser.add_argument(
        "--target_utt2num_frames", type=str, default=None, help="Target utt2num_frames file"
    )
    parser.add_argument(
        "--interp_mode", type=str, help="Interpolation mode", default="bilinear"
    )
    parser.add_argument("--lip_width", type=int, help="Width of the output lip frame")
    parser.add_argument("--lip_height", type=int, help="Width of the output lip frame")
    parser.add_argument(
        "--shape_predictor_path", type=str, help="the pretrained lip predictor path"
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
    parser.add_argument(
        "--normalize",
        type=bool,
        default=True,
        help="Normalizes image data to scale in [-1,1]",
    )
    parser.add_argument("rspecifier", type=str, help="video scp file")
    parser.add_argument("face_wspecifier", type=str, help="Write face specifier")
    parser.add_argument("info_file", type=str, help="file containing extraction information")
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
    lip_extractor = Lip_Extractor(args.shape_predictor_path)
    lip_size = (args.lip_width, args.lip_height)

    # Read, extract and write.
    with open(args.rspecifier, "r") as video_scp, \
        open(args.info_file, "w") as f_info, \
        file_writer_helper(
            args.face_wspecifier,
            filetype=args.filetype,
            write_num_frames=args.write_num_frames,
            compress=args.compress,
            compression_method=args.compression_method,
        ) as f_writer:

        # Loop start.
        for line in video_scp.read().splitlines():
            utt_id, video_path = line.split(" ")
            rate, frames = load_video(video_path)
            if args.fps is not None and rate != args.fps:
                raise Exception(
                    "The video sampling rate ({}) is different with the config ({}) !".format(
                        rate, args.fps
                    )
                )

            logging.info("Extracting lip features of {}...".format(utt_id))
            lip_frames = lip_extractor.catch_lip(frames, lip_size)
            lip_features = numpy.array([frame['lip_frame'] for frame in lip_frames])
            t, h, w = lip_features.shape
            lip_features = numpy.reshape(lip_features, (t, h * w))

            # Resample
            if target_utt2num_frames is not None:
                target_nframes = target_utt2num_frames[utt_id]
                lip_features = temporal_resize(lip_features, target_nframes, interp_mode)

            if args.normalize:
                lip_features = (lip_features / 128) - 1

            # write lip feature
            f_writer[utt_id] = lip_features

            # write information
            f_info.write(f"{utt_id}\n")
            for frame in lip_frames:
                f_info.write(f"{frame['detection']} {frame['bbox'][0]} {frame['bbox'][1]} {frame['bbox'][2]} {frame['bbox'][3]}\n")

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
