import cv2
import mediapipe as mp
import numpy as np
import pickle


# Hàm xử lý ảnh đầu vào
def extract_landmarks(frame, hands_model):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(img_rgb)

    try:
        data = []
        # Nếu có từ 1 tay trở lên
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    data.extend([lm.x, lm.y, lm.z])

        # Nếu chỉ có 1 tay, thêm 63 giá trị 0 vào để đủ 126 đặc trưng
        if len(results.multi_hand_landmarks) == 1:
            data.extend([0] * 63)  # Thêm 63 số 0 cho tay thứ 2 (không có)

        # Nếu có 2 tay, thì không cần thêm giá trị 0
        elif len(results.multi_hand_landmarks) > 2:
            # Nếu có nhiều hơn 2 tay, chỉ lấy 2 tay đầu tiên
            data = data[:126]  # Lấy đủ 126 đặc trưng cho 2 tay

        return data
    except:
        return [0] * 126  # Nếu không nhận diện được tay nào, trả về 126 giá trị 0


# Load model và LabelEncoder
with open('model.pkl', 'rb') as f:
    model, label_encoder = pickle.load(f)

# Khởi tạo camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Khởi tạo mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Nhận diện tối đa 2 tay
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting ...")
        break

    frame = cv2.flip(frame, 1)
    landmarks = extract_landmarks(frame, hands)

    # Nếu có tay thì vẽ landmarks và dự đoán
    if np.any(landmarks):
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        prediction = model.predict(np.array(landmarks).reshape(1, -1))[0]
        label = label_encoder.inverse_transform([prediction])[0]

        # Hiển thị kết quả
        cv2.putText(frame, f'Prediction: {label.upper()}', (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    cv2.imshow('Hand Sign Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
