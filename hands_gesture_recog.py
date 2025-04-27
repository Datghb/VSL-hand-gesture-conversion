import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np
import pickle

# Hàm xử lý ảnh
def image_processed(hand_img, hands_model):
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
    img_flip = cv2.flip(img_rgb, 1)

    output = hands_model.process(img_flip)

    try:
        data = output.multi_hand_landmarks[0]
        data = str(data).strip().split('\n')
        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        clean = []
        for i in data:
            if i not in garbage:
                i = i.strip()
                clean.append(float(i[2:]))
        return clean
    except:
        return np.zeros([1, 63], dtype=int)[0]

# Load model
with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)

# Khởi tạo camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Khởi tạo Mediapipe Hands để vẽ landmarks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame = cv2.flip(frame, 1)  # Lật frame trước
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Nếu có bàn tay thì vẽ landmarks
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # Tiếp tục xử lý để dự đoán
    data = image_processed(frame, hands)
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1, 63))
    print(y_pred)

    # Ghi kết quả lên frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 100)
    fontScale = 3
    color = (255, 0, 0)
    thickness = 5
    frame = cv2.putText(frame, str(y_pred[0]), org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
