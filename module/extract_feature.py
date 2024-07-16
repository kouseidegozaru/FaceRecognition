import face_recognition

def extract_face_feature(face_image):
    """
    入力された顔領域画像から特徴量を抽出して返す
    """
    # 画像の有無
    if face_image is None:
        raise("Error: Image is None")
    
    # 顔画像から128次元の特徴ベクトルを抽出
    face_encoding = face_recognition.face_encodings(face_image)
    
    if len(face_encoding) > 0:
        # 最初の顔の特徴ベクトルを返す
        return face_encoding[0]
    else:
        return None
