from features.color_histogram import *
from features.feature_extraction import *
from pathlib import Path
import sqlite3
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[3]

class Retriever:
    def __init__(self, db:sqlite3.Connection, base_dir):
        self.db = db
        self.cursor = db.cursor()
        self.base_dir = base_dir

    def image_similarity_search(self, query, search_method="Euclidean Distance"):
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
        
        if search_method == "Euclidean Distance":
            for idx, v in vector_db:
                distances.append((idx, 1 - self.euclidean_dist(q_v, v)))

        elif search_method == "Cosine Similarity":
            for idx, v in vector_db:
                distances.append((idx, self.cosine_similarity(q_v, v)))

        
        distances.sort(key=lambda x: x[1], reverse=True)

        return distances

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
        
        return self.cursor.fetchall()


    def euclidean_dist(self, base, target):
        return math.dist(base, target)

    def manhattan_dist(self, base, target):
        return np.sum(np.abs(base - target))

    def chi2_dist(self, base, target):
        chi = 0.5 * np.sum([((a - b) ** 2) / (a + b) for (a, b) in zip(base, target)])
        return chi
    
    def cosine_similarity(self, base, target):
        return np.dot(base, target)
