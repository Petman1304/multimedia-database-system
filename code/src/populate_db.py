from features.feature_extraction import *
from pathlib import Path
import glob
import sqlite3
import os

ROOT = Path(__file__).resolve().parents[2]

def insert_images_data(filename, filepath, size,  label, width, height, channels, extension, vector, type='image'):
    vector = vector.tobytes()
    try:
        cursor.execute(
            "INSERT INTO media (filename, filepath, size, type) VALUES (?, ?, ?, ?)",
            (filename, filepath, size, type)
        )

        media_id = cursor.lastrowid

        cursor.execute(
            "INSERT INTO image (media_id, label) VALUES (?, ?)",
            (media_id, label)
        )

        cursor.execute(
            "INSERT INTO image_metadata (media_id, width, height, channels, extension) VALUES (?, ?, ?, ?, ?)",
            (media_id, width, height, channels, extension)
        )

        cursor.execute(
            "INSERT INTO image_features (media_id, image_vector) VALUES (?, ?)",
            (media_id, vector)
        )
    except :
        print("Errir inserting data to tables")

db = sqlite3.connect("../../database/multimedia.db")
db.execute("PRAGMA foreign_keys = ON")

cursor = db.cursor()

ddl = """

CREATE TABLE IF NOT EXISTS media(
    media_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    filepath    TEXT NOT NULL,
    size        INTEGER NOT NULL,
    type  TEXT CHECK(type IN ('image', 'video')) NOT NULL
);

CREATE TABLE IF NOT EXISTS image (
    media_id INTEGER PRIMARY KEY,
    label TEXT,
    FOREIGN KEY (media_id) REFERENCES media(media_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS image_metadata (
    media_id INTEGER PRIMARY KEY,
    width INTEGER,
    height INTEGER,
    channels INTEGER,
    extension TEXT,
    FOREIGN KEY (media_id) REFERENCES image(media_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS image_features (
    media_id INTEGER PRIMARY KEY,
    image_vector BLOB NOT NULL,
    FOREIGN KEY (media_id) REFERENCES image(media_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS video (
    media_id INTEGER PRIMARY KEY,
    label TEXT,
    FOREIGN KEY (media_id) REFERENCES media(media_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS video_metadata (
    media_id INTEGER PRIMARY KEY,
    duration REAL,
    fps INTEGER,
    extension TEXT,
    FOREIGN KEY (media_id) REFERENCES video(media_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS video_keyframes (
    keyframe_id INTEGER PRIMARY KEY AUTOINCREMENT,
    media_id INTEGER NOT NULL,
    frame BLOB,
    timestamp REAL,
    FOREIGN KEY (media_id) REFERENCES video(media_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS keyframe_features (
    keyframe_id INTEGER PRIMARY KEY, 
    keyframe_vector BLOB NOT NULL,
    FOREIGN KEY (keyframe_id) REFERENCES video_keyframes(keyframe_id) ON DELETE CASCADE
);
"""

cursor.executescript(ddl)

img_label = []
img_path = []

for image_label in os.listdir(ROOT / "database" / "image_file"):
    label_path = os.path.join("database", "image_file", image_label)

    # if not os.path.isdir(label_path):
    #     continue

    for fname in os.listdir(ROOT/label_path):
        path = os.path.join(label_path, fname)
        img_label.append(image_label)
        img_path.append(path)

# img_label = []
# with open("../../database/image_file/labels.txt", "r") as f:
#     img_label = [line.strip() for line in f.readlines()]

# img_path = [image for image in glob.glob(os.path.join(r"..\..\database\image_file", "*.png"))]

print(f"- Starting populating database with {len(img_path)} image data...")

i = 0

for img in img_path:
    bgr_img = cv2.imread(ROOT/img)

    filename, size, height, width, channels, format = get_image_metadata(img)

    img_vector = img_feature_extraction(bgr_img)

    label = img_label[i]
    i = i + 1 

    insert_images_data(filename, img, size, label, width, height, channels, format, img_vector)

print(f"- Finished populating database with {len(img_path)} image data...")

videos = []

for vid_label in os.listdir(r"..\..\database\video_file"):
    label_path = os.path.join("database", "video_file", vid_label)

    # if not os.path.isdir(label_path):
    #     continue

    for fname in os.listdir(ROOT/label_path):
        video_path = os.path.join(label_path, fname)
        videos.append((video_path, vid_label))

print(f"- Starting populating database with {len(videos)} video data...")
for vid_path, label in videos:
    filename, file_size, duration, fps, format = get_video_metadata(vid_path)
    
    cursor.execute(
        "INSERT INTO media (filename, filepath, size, type) VALUES (?, ?, ?, ?)",
        (filename, vid_path, file_size, 'video')
    )

    media_id = cursor.lastrowid

    cursor.execute(
        "INSERT INTO video (media_id, label) VALUES (?, ?)",
        (media_id, label)
    )

    cursor.execute(
        "INSERT INTO video_metadata (media_id, duration, fps, extension) VALUES (?, ?, ?, ?)",
        (media_id, duration, fps, format)
    )
    
    keyframes = extract_keyframes(vid_path)
    for timestamp, frame in keyframes:

        _, buffer = cv2.imencode(".png", frame)
        frame_blob = buffer.tobytes()
        frame_vector = img_feature_extraction(frame)
        fv_blob = frame_vector.tobytes()

        cursor.execute(
            "INSERT INTO video_keyframes (media_id, frame, timestamp) VALUES (?, ?, ?)",
            (media_id, frame_blob, timestamp)
        )

        keyframe_id = cursor.lastrowid

        cursor.execute(
            "INSERT INTO keyframe_features (keyframe_id, keyframe_vector) VALUES (?, ?)",
            (keyframe_id, fv_blob)
        )
print(f"- Finished populating database with {len(videos)} video data...")

db.commit()
db.close()


    

