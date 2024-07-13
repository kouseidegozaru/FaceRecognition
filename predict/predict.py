import os
import sys
import cv2
from pathlib import Path
import joblib

# 設定ファイルインポート
sys.path.append(str(Path(__file__).resolve().parent.parent))
from resource.info import resource_info

from module.image_detection import detect_face
from module.extract_feature import extract_face_feature

def predict(image_path,model_path):

    #学習済みモデルの読み込み
    model = joblib.load(model_path)

    # 顔写真を入力して、人物を特定する
    unknown_img = cv2.imread(image_path)
    unknown_face = detect_face(unknown_img)
    if unknown_face is None:
        print('顔画像が見つかりませんでした')
        return None,None
    unknown_feature = extract_face_feature(unknown_face)

    # 近傍探索を行い、最も近い特徴量を持つ人物IDを取得
    distances, indices = model.kneighbors([unknown_feature])
    return distances, indices


def main():

    model_path = resource_info.CURRENT_MODEL_PATH
    person_names = resource_info.PERSON_NAMES
    test_image_dir_path = resource_info.TEST_IMAGE_DIR_PATH
    test_image_names = resource_info.TEST_IMAGE_NAMES

    for image_name in test_image_names:
        image_path = os.path.join(test_image_dir_path,image_name)
        distances, indices = predict(image_path,model_path)
        if distances is not None:
            # 最も近い特徴量を持つ人物IDを表示
            print(f'Name: {person_names[indices[0][0]]}')
            print(f'Distances: {distances}')
            print('--------------------------------')


    

if __name__ == '__main__':
    main()