# Functions for feature extractions

import av
import cv2
import numpy as np
import math
import os
from pathlib import Path
from skimage import img_as_ubyte
from skimage.feature import local_binary_pattern, hog
from skimage.filters import roberts, sobel
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage.color import rgb2lab, rgb2gray

ROOT = Path(__file__).resolve().parents[3]

def img_feature_extraction (img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))

    LAB_img = rgb2lab(img)
    A_img = LAB_img[:,:,1]
    A_feat = A_img.mean()

    B_img = LAB_img[:,:,2]
    B_feat = B_img.mean()

    gray = rgb2gray(img)
    gray = img_as_ubyte(gray)

    ent = entropy(gray, disk(3))
    ent_mean = ent.mean()
    ent_std = ent.std()

    rob = roberts(gray)
    rob_mean = rob.mean()

    sob = sobel(gray)
    sob_mean = sob.mean()

    #Gabor 1
    kernel1 = cv2.getGaborKernel((9, 9), 3, np.pi/4, np.pi, 0.5, 0, ktype=cv2.CV_32F)    
    gabor1 = (cv2.filter2D(gray, cv2.CV_8UC3, kernel1)).mean()
    
    #Gabor 2
    kernel2 = cv2.getGaborKernel((9, 9), 3, np.pi/2, np.pi/4, 0.9, 0, ktype=cv2.CV_32F)    
    gabor2 = (cv2.filter2D(gray, cv2.CV_8UC3, kernel2)).mean()

    #Gabor 3
    kernel3 = cv2.getGaborKernel((9, 9), 5, np.pi/2, np.pi/2, 0.1, 0, ktype=cv2.CV_32F)    
    gabor3 = (cv2.filter2D(gray, cv2.CV_8UC3, kernel3)).mean()

    custom_features = np.array([A_feat, B_feat, ent_mean, ent_std, rob_mean, 
                                sob_mean, gabor1, gabor2, gabor3], np.float32)

    custom_features /= (np.linalg.norm(custom_features) + 1e-6)
    return custom_features

def get_image_metadata(image_path):
    image_path = ROOT/image_path 
    img = cv2.imread(image_path)

    filename = os.path.basename(image_path)

    size = img.nbytes

    _, ext = os.path.splitext(filename)
    format = ext.lower().replace('.', '')

    height, width, channels = img.shape
    return filename, size, height, width, channels, format

def extract_keyframes(video_path):
    video_path = ROOT/video_path
    keyframes = []

    with av.open(video_path) as container:
        container.streams.video[0].thread_type = "AUTO"
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONREF"

        for frame in container.decode(stream):
            if frame.key_frame:
                img = frame.to_ndarray(format="bgr24")
                img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
                timestamp = float(frame.pts*stream.time_base)
                keyframes.append((timestamp, img))

    return keyframes

def get_video_metadata(video_path):
    video_path = ROOT/video_path
    try:
        with av.open(video_path) as container:
            filename = os.path.basename(video_path)
            duration = float(container.duration / av.time_base) if container.duration else 0.
            file_size = container.size

            _, ext = os.path.splitext(filename)
            format = ext.lower().replace('.', '')

            stream = container.streams.video[0]
            fps = float(stream.average_rate)

            return filename, file_size, duration, fps, format
    except IndexError:
        print("No video stream found in the file")



