import os
import sys
import cv2
import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
import joblib

# 設定ファイルインポート
sys.path.append(str(Path(__file__).resolve().parent.parent))
from resource.info import resource_info

from module.image_detection import detect_face
from module.extract_feature import extract_face_feature

def main():
    model_path = resource_info.CURRENT_MODEL_PATH
    image_dir_path = resource_info.TRAIN_IMAGE_DIR_PATH
    image_names = resource_info.TRAIN_IMAGE_NAMES
    person_info = resource_info.PERSON_NAMES

    # 顔写真データの登録
    face_data = []
    labels = []
    for (person_id, person_name), image_name in zip(person_info.items(), image_names):

        # 各人物の顔写真を1枚ずつ読み込む
        img = cv2.imread(os.path.join(image_dir_path, image_name))
        
        # 顔領域を検出し、特徴量を抽出する
        face = detect_face(img)
        face_feature = extract_face_feature(face)
        
        # 特徴量と人物IDを登録
        face_data.append(face_feature)
        labels.append(person_id)

    # 顔特徴のデータセットを作成
    X = np.array(face_data)

    # 顔特徴の近傍探索モデルを学習
    model = NearestNeighbors(n_neighbors=1)
    model.fit(X)

    # 学習済みモデルを保存
    joblib.dump(model, model_path)

if __name__ == '__main__':
    main()