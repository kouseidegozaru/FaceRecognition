import cv2

def detect_face(image):
    """
    入力画像から顔領域を検出し、その領域を切り出して返す
    """
    # 画像の有無
    if image is None:
        raise("Error: Image is None")
    
    # OpenCV の顔検出器を取得
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # 入力画像をグレースケールに変換
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 顔領域を検出
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # 検出された顔領域の中から、最大の領域を取得
    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda r: r[2]*r[3])
        return image[y:y+h, x:x+w]
    else:
        return None
