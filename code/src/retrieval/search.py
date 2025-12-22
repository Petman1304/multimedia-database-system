from features.color_histogram import *
from features.feature_extraction import *
import sqlite3
import math
import numpy as np

def image_similarity_search(db : sqlite3.Connection, query, top_k=5):
    cursor = db.cursor()
    try:
        vector_db = cursor.execute(
            "SELECT media_id, image_vector FROM image_features"
        )
        vector_db = [(idx, np.frombuffer(feat, np.float32))for idx, feat in vector_db.fetchall()]
    except:
        print("Error fetching database")
    
    q = query
    q_v = img_feature_extraction(q)

    distances = []

    for idx, v in vector_db:
        distances.append((idx, math.dist(q_v, v)))

    distances.sort(key=lambda x: x[1])

    return distances[:top_k]

def fetch_image_from_db(db:sqlite3.Connection, search_result:list):
    cursor = db.cursor()
    images = []

    for id, dist in search_result:
        cursor.execute(
            "SELECT filepath FROM media WHERE media_id = ?",
            (id,)
        )
        path = cursor.fetchone()[0]
        img = cv2.imread(path)
        images.append((id, dist, img))

    return images

def search(idx, labels, rgb_images, bins=8 ,top_k=5, method = 'euclidean'):
    assert method in ['euclidean', 'manhattan', 'chi-square']

    vector_db = preprocess_data(labels, rgb_images, bins=bins)
    
    query_vector = vector_db[idx]
    distances = []

    for _, vector in enumerate(vector_db):
        if method == 'euclidean':
            distances.append(euclidean_dist(query_vector, vector))
        elif method == 'manhattan':
            distances.append(manhattan_dist(query_vector, vector))
        else:
            distances.append(chi2_dist(query_vector, vector))

    top_idx = np.argpartition(distances, top_k)[:top_k]

    res_labels = [labels[id] for id in top_idx]

    print(f"Query label : {labels[idx]}")
    print(f"Result labels : {res_labels}")

    plt.imshow(rgb_images[idx])
    
    fig, axs = plt.subplots(1, top_k, )

    for i in range(top_k):
        axs[i].imshow(rgb_images[top_idx[i]])

    plt.show 

    return res_labels, top_idx

def euclidean_dist(base, target):
    return math.dist(base, target)

def manhattan_dist(base, target):
    return np.sum(np.abs(base - target))

def chi2_dist(base, target):
    chi = 0.5 * np.sum([((a - b) ** 2) / (a + b) for (a, b) in zip(base, target)])
    return chi
