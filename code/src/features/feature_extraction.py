import av
import cv2
import numpy as np
import math
import os



def img_to_hu_moments(rgb_image):
    # Convert rgb image to grayscale
    img = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    # Find image edges
    edges = cv2.Canny(img, 200, 400)

    # Extract image countours
    contours, _ = cv2.findContours(edges,
                               cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return np.zeros(7, dtype=np.float32)
    
    contours = max(contours, key=cv2.contourArea)
    
    # Calculate Hu Moments
    moments = cv2.moments(contours)
    huMoments = cv2.HuMoments(moments).flatten()

    # Log transform to bring hu moments in the same range
    huMoments = -np.sign(huMoments) * np.log10(np.abs(huMoments))

    # Return normalize Hu Moment
    return huMoments/(np.linalg.norm(huMoments) + 1e-8)

def img_to_hist(rgb_image, bins=8):
    rgb = rgb_image
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    red = cv2.calcHist(
        [rgb], [0], None, [bins], [0, 256]
    )

    green = cv2.calcHist(
        [rgb], [1], None, [bins], [0, 256]
    )
    
    blue = cv2.calcHist(
        [rgb], [2], None, [bins], [0, 256]
    )

    hue = cv2.calcHist(
        [hsv], [0], None, [bins], [0, 256]
    )

    saturation = cv2.calcHist(
        [hsv], [1], None, [bins], [0, 256]
    )

    value = cv2.calcHist(
        [hsv], [2], None, [bins], [0, 256]
    )

    vector = np.concatenate([red, green, blue, hue, saturation, value], axis=0)
    vector = vector.reshape(-1)

    return vector / np.linalg.norm(vector)

def img_feature_extraction (img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    w_hist = math.sqrt(7/48)
    w_hu = 1.

    hu_moment = img_to_hu_moments(img)
    hist = img_to_hist(img)

    hist *= w_hist
    hu_moment *= w_hu

    vector = np.concatenate([hist, hu_moment])

    vector = vector / np.linalg.norm(vector)
    
    return np.array(vector, dtype=np.float32)

def get_image_metadata(image_path): 
    img = cv2.imread(image_path)

    filename = os.path.basename(image_path)

    size = img.nbytes

    _, ext = os.path.splitext(filename)
    format = ext.lower().replace('.', '')

    height, width, channels = img.shape
    return filename, size, height, width, channels, format

def extract_keyframes(video_path):
    keyframes = []

    with av.open(video_path) as container:
        container.streams.video[0].thread_type = "AUTO"
        stream = container.streams.video[0]
        stream.codec_context.skip_frame = "NONREF"

        for frame in container.decode(stream):
            if frame.key_frame:
                img = frame.to_ndarray(format="bgr24")
                img = cv2.resize(img, (160, 120), interpolation=cv2.INTER_AREA)
                timestamp = float(frame.pts*stream.time_base)
                keyframes.append((timestamp, img))

    return keyframes

def get_video_metadata(video_path):
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



