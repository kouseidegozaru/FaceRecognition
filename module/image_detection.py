import face_recognition
import numpy as np

def detect_face(image):
    """
    入力画像から顔領域を検出し、その領域を切り出して返す
    """
    # 画像の有無
    if image is None:
        raise ValueError("Error: Image is None")
    
    # 顔領域を検出
    face_locations = face_recognition.face_locations(image)
    
    # 検出された顔領域の中から、最大の領域を取得
    if len(face_locations) > 0:
        top, right, bottom, left = max(face_locations, key=lambda r: (r[2]-r[0])*(r[1]-r[3]))
        face_image = image[top:bottom, left:right]
        return np.array(face_image)  # numpy配列として返す
    else:
        return None