import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import normalize

def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')

    return dict

def get_image(image_bytes):
    image = np.zeros(shape=(32, 32, 3))

    reds = image_bytes[:1024]
    greens = image_bytes[1024:2048]
    blues = image_bytes[2048:3072]

    for i in range(32):
        for j in range(32):
            image[i, j, 0] = reds[i*32 + j]
            image[i, j, 1] = greens[i*32 + j]
            image[i, j, 2] = blues[i*32 + j]

    return image.astype(np.uint8)

def get_hsv(rgb_images:list):
    
    hsv_images = [cv2.cvtColor(img, cv2.COLOR_RGB2HSV) for img in rgb_images]

    return hsv_images

def build_histogram(id, rgb_images, hsv_images, bins=8):
    
    plt.imshow(rgb_images[id])

    rgb = rgb_images[id]
    hsv = hsv_images[id]

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

    fig, axs = plt.subplots(2, 3, figsize=(15,4))
    axs[0, 0].hist(red, bins=bins, color='r')
    axs[0, 1].hist(green, bins=bins, color='g')
    axs[0, 2].hist(blue, bins=bins, color='b')
    axs[1, 0].hist(hue, bins=bins, color='y')
    axs[1, 1].hist(saturation, bins=bins, color='m')
    axs[1, 2].hist(value, bins=bins, color='c')

    plt.show()

def get_vector(rgb, hsv ,bins=8):
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
    
    return vector

def split_per_category(dataset_dict : dict, n : int = 9999):
    labels = dataset_dict[b'labels']
    data = dataset_dict[b'data']

    new_labels = []
    new_data = []

    counter = {}

    for i in range (len(labels)):
        if labels[i] not in counter:
            counter[labels[i]] = 0
        
        if counter[labels[i]] < n:
            counter[labels[i]] += 1
            new_labels.append(labels[i])
            new_data.append(data[i])

    return new_labels, new_data


def images2vectors(rgb_images:list, hsv_images:list, bins=8):
    images_vector = [get_vector(rgb_images[i], hsv_images[i], bins=bins) for i in range(len(rgb_images))]
    images_vector = np.vstack(images_vector)

    return normalize(images_vector, 'l2')

def get_dataset(file:str):
    dataset = unpickle(file=file)

    labels, images_byte = split_per_category(dataset)

    rgb_images = [get_image(img) for img in images_byte]

    return labels, rgb_images

def preprocess_data(labels, rgb_images, bins=8):
    hsv_images = get_hsv(rgb_images)

    image_vector = images2vectors(rgb_images, hsv_images, bins=bins)

    return image_vector




