import cv2
from sklearn.decomposition import PCA

def extract_face_feature(face_image):
    """
    入力された顔領域画像から特徴量を抽出して返す
    """
    # 画像の有無
    if face_image is None:
        raise("Error: Image is None")
    # 顔領域画像をリサイズ
    resized = cv2.resize(face_image, (64, 64))
    
    # 画像を1次元の特徴ベクトルに変換
    feature_vector = resized.flatten()
    
    # 主成分分析 (PCA) を使って特徴量を圧縮
    pca = PCA(n_components=1)
    compressed_feature = pca.fit_transform([feature_vector])[0]
    
    return compressed_feature
