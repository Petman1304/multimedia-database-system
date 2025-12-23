from util.feature_extraction import *
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import sqlite3
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

class Retriever:
    def __init__(self, db:sqlite3.Connection, base_dir):
        self.db = db
        self.cursor = db.cursor()
        self.base_dir = base_dir
        self.image_knn = self.build_image_knn()

    def build_image_knn(self):
        self.cursor.execute(
            """
SELECT image_features.media_id, image_features.image_vector
FROM image_features
"""
        )

        vecs = self.cursor.fetchall()
        img_v = []
        m_id = []

        for id, v in vecs:
            m_id.append(id)
            img_v.append(np.frombuffer(v, dtype=np.float32))

        knn = NearestNeighbors(
            n_neighbors=13,
        )

        knn.fit(img_v)

        return knn 


    def image_similarity_search(self, query, search_method="Euclidean Distance", top_k=5):
        try:
            vector_db = self.cursor.execute(
                "SELECT media_id, image_vector FROM image_features"
            )
            vector_db = [(idx, np.frombuffer(feat, np.float32))for idx, feat in vector_db.fetchall()]
        except:
            print("Error fetching database")
        
        q = query
        q_v = img_feature_extraction(q)


        distances = []
        
        if search_method == "Euclidean Distance":
            for idx, v in vector_db:
                distances.append((idx, 1 - self.euclidean_dist(q_v, v)))

        elif search_method == "Cosine Similarity":
            for idx, v in vector_db:
                distances.append((idx, self.cosine_similarity(q_v, v)))
        
        else:
            knn = self.image_knn

            m_id = []
            for idx, v in vector_db:
                m_id.append(idx)

            distances = self.knn_search(knn, m_id, q_v, 5)

        
        distances.sort(key=lambda x: x[1], reverse=True)

        return distances[:top_k]

    def fetch_image_from_db(self, search_result):
        cursor = self.cursor
        images = []

        for id, dist in search_result:
            cursor.execute(
                """
                SELECT 
                    media.filename, media.filepath, media.size,
                    image.label,
                    image_metadata.width, image_metadata.height, image_metadata.channels, image_metadata.extension
                FROM 
                    media 
                INNER JOIN 
                    image ON media.media_id = image.media_id
                INNER JOIN
                    image_metadata ON media.media_id = image_metadata.media_id
                WHERE 
                    media.media_id = ?
                """,
                (id, )
            )

            row = cursor.fetchone()
            path = row[1]
            path = os.path.join(self.base_dir, path.replace("\\", "/"))
            img = cv2.imread(path)

            fn = row[0]
            size = row[2]
            w = row[4]
            h = row[5]
            ch = row[6]
            ext = row[7]

            label = row[3]

            metadata = (fn, size, w, h, ch, ext)


            images.append((id, dist, img, metadata, label))
        return images
    
    def image_metadata_filter(self, max_size, min_w, max_w, min_h, max_h, ext):
        db_q = """
SELECT media.media_id
FROM media
INNER JOIN image_metadata ON media.media_id = image_metadata.media_id
WHERE 1=1
"""     
        params = []
        if ext:
            e = ",".join("?" for _ in ext)
            db_q += f"AND image_metadata.extension IN ({e})"

        params.extend(ext)

        db_q += "AND image_metadata.width BETWEEN ? AND ?"
        params.extend([min_w, max_w])
        db_q += "AND image_metadata.height BETWEEN ? AND ?"
        params.extend([min_h, max_h])
        db_q += "AND media.size <= ?"
        params.append(max_size)

        self.cursor.execute(db_q, params)
        result = self.cursor.fetchall()
        return result
    
    def video_metadata_filter(self, max_size, min_fps, max_fps, min_dur, max_dur, ext):
        db_q = """
SELECT media.media_id
FROM media
INNER JOIN video_metadata ON media.media_id = video_metadata.media_id
WHERE 1=1
"""     
        params = []
        if ext:
            e = ",".join("?" for _ in ext)
            db_q += f"AND image_metadata.extension IN ({e})"

        params.extend(ext)

        db_q += "AND image_metadata.width BETWEEN ? AND ?"
        params.extend([min_w, max_w])
        db_q += "AND image_metadata.height BETWEEN ? AND ?"
        params.extend([min_h, max_h])
        db_q += "AND media.size <= ?"
        params.append(max_size)

        self.cursor.execute(db_q, params)
        result = self.cursor.fetchall()
        return result


    def euclidean_dist(self, base, target):
        return math.dist(base, target)

    def manhattan_dist(self, base, target):
        return np.sum(np.abs(base - target))

    def chi2_dist(self, base, target):
        chi = 0.5 * np.sum([((a - b) ** 2) / (a + b) for (a, b) in zip(base, target)])
        return chi
    
    def cosine_similarity(self, base, target):
        return np.dot(base, target)
    
    def knn_search(self, knn, m_id, query, top_k):
        query = query.reshape(1, -1)
        dist, idx = knn.kneighbors(query, n_neighbors=top_k)

        return ((m_id[i], dist) for i, dist in zip(idx[0], dist[0]))
