#!/usr/bin/env python3

# Copyright 2021 Academia Sinica (Pin-Jui Ku, Wen-Chin Huang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import cv2
import dlib
from tqdm import tqdm
import numpy
from imutils import face_utils

from scipy.interpolate import interp1d

def crop_frame(frame, x, y, w, h):
    b, r = frame.shape
    b_pad = 0 if b >= y + h else y + h - b
    r_pad = 0 if r >= x + w else x + w - r
    if b_pad > 0 or r_pad > 0:
        frame = cv2.copyMakeBorder(frame, 0, b_pad,0, r_pad, cv2.BORDER_CONSTANT, value=0)
    return frame[y : y + h, x : x + w]

def load_video(filepath):
    video_stream = cv2.VideoCapture(filepath)
    rate = video_stream.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        frames.append(frame)

    return rate, frames

def linear_interpolation(starting_frame, ending_frame, missing_frames, size):
    num_frames = len(missing_frames)
    t = (-1, num_frames)
    for j in range(4):
        y = (starting_frame["bbox"][j], ending_frame["bbox"][j])
        f = interp1d(t, y)
        values = f([_ for _ in range(num_frames)])
        for i, value in enumerate(values):
            missing_frames[i]["bbox"][j] = int(value)

    for i in range(len(missing_frames)):
        x, y, w, h = missing_frames[i]["bbox"]
        lip_frame = crop_frame(missing_frames[i]["grey_frame"], x, y, w, h)
        lip_frame = cv2.resize(lip_frame, size, interpolation=cv2.INTER_AREA)
        missing_frames[i]["lip_frame"] = lip_frame

    return missing_frames


class Lip_Extractor:
    def __init__(self, shape_predictor_path):
        self.face_detector = dlib.get_frontal_face_detector()
        self.lip_detector = dlib.shape_predictor(shape_predictor_path)

    def catch_lip(self, video_frames, size):
        lip_frames = []

        # First loop through the frames
        for i, video_frame in enumerate(video_frames):

            # RGB to grey scale
            gray_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2GRAY)

            # Actual face detection
            faces = self.face_detector(gray_frame, 1)

            if len(faces) == 0:
                content = {
                    "detection": False,
                    "grey_frame": gray_frame,
                    "bbox": [None, None, None, None],
                }
            else:
                face = faces[0]
                shape = self.lip_detector(gray_frame, face)
                shape = face_utils.shape_to_np(shape)
                lip_indexs = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
                jaw_indexs = face_utils.FACIAL_LANDMARKS_IDXS["jaw"]

                lip_shape = shape[lip_indexs[0] : lip_indexs[1]]
                jaw_shape = shape[jaw_indexs[0] : jaw_indexs[1]]
                nose_shape = shape[30:31]

                (x, y, w, h) = cv2.boundingRect(numpy.array([lip_shape]))
                # x, y = min(jaw_shape[:,0]), min(nose_shape[:,1])
                # w, h = max(jaw_shape[:,0]) - x, max(jaw_shape[:,1]) - y
                x = min(jaw_shape[:,0])
                w = max(jaw_shape[:,0]) - x
                y = y + h // 2 - w // 4
                h = w // 2

                lip_frame = crop_frame(gray_frame, x, y, w, h)
                lip_frame = cv2.resize(lip_frame, size, interpolation=cv2.INTER_AREA)

                content = {
                    "detection": True,
                    "grey_frame": gray_frame,
                    "bbox": (x, y, w, h),
                    "lip_frame": lip_frame,
                }

            lip_frames.append(content)
        
        # Second loop to do linear interpolation
        new_lip_frames = []
        i = 0
        while i < len(lip_frames):
            content = lip_frames[i]

            if lip_frames[i]["detection"]:
                new_lip_frames.append(lip_frames[i])
                i += 1

            else:
                missing_frames = []

                # no detected frames before
                if i == 0:

                    # find next detected frame
                    while not lip_frames[i]["detection"]:
                        missing_frames.append(lip_frames[i])
                        i += 1

                    # Now lip_frames[i] is a successfully detected frame
                    ending_frame = lip_frames[i]

                    # beacuse no detected frames before, use end frame as starting frame
                    starting_frame = ending_frame

                else:
                    # the last frame is definitely a good frame
                    starting_frame = lip_frames[i-1]

                    # find next detected frame
                    while i < len(lip_frames) and not lip_frames[i]["detection"]:
                        missing_frames.append(lip_frames[i])
                        i += 1
                
                    # if reaching end, meaning no detected frame later anymore
                    if i == len(lip_frames):
                        ending_frame = starting_frame

                    # normal situation
                    else:
                        # Now lip_frames[i] is a successfully detected frame
                        ending_frame = lip_frames[i]
                    
                # perform linear interpolation
                interpolated_frames = linear_interpolation(
                    starting_frame, ending_frame, missing_frames, size)
                new_lip_frames += interpolated_frames

        return lip_frames


