from features.color_histogram import *
from features.feature_extraction import *
from pathlib import Path
import sqlite3
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

class Retriever:
    def __init__(self, db:sqlite3.Connection):
        self.db = db
        self.cursor = db.cursor()

    def image_similarity_search(self, query, top_k=5):
        try:
            vector_db = self.cursor.execute(
                "SELECT media_id, image_vector FROM image_features"
            )
            vector_db = [(idx, np.frombuffer(feat, np.float32))for idx, feat in vector_db.fetchall()]
        except:
            print("Error fetching database")
        
        q = query
        q_v = img_feature_extraction(q)
        print("Query norm:", np.linalg.norm(q_v))

        for idx, v in vector_db[:5]:
            v = np.frombuffer(v, np.float32)
            print(idx, np.linalg.norm(v))

        distances = []

        for idx, v in vector_db:
            distances.append((idx, self.cosine_similarity(q_v, v)))

        distances.sort(key=lambda x: x[1], reverse=True)

        return distances[:top_k]

    def fetch_image_from_db(self, search_result, base_dir):
        cursor = self.cursor
        images = []

        for id, dist in search_result:
            cursor.execute(
                "SELECT filepath FROM media WHERE media_id = ?",
                (id,)
            )
            path = cursor.fetchone()[0]
            print(path)
            path = os.path.normpath(os.path.join(base_dir, path))
            img = cv2.imread(path)
            images.append((id, dist, img))

        return images

    def euclidean_dist(self, base, target):
        return math.dist(base, target)

    def manhattan_dist(self, base, target):
        return np.sum(np.abs(base - target))

    def chi2_dist(self, base, target):
        chi = 0.5 * np.sum([((a - b) ** 2) / (a + b) for (a, b) in zip(base, target)])
        return chi
    
    def cosine_similarity(self, base, target):
        return np.dot(base, target) / ((np.linalg.norm(base)*np.linalg.norm(target)) + 1e-6)
