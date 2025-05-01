import numpy as np
import cv2 as cv
from pathlib import Path
import mediapipe as mp


def get_image():
    Class = 'sorry'
    Path('DATASET/' + Class).mkdir(parents=True, exist_ok=True)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    i = 0
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2  # Nhận diện tối đa 2 tay
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame = cv.flip(frame, 1)
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                # Tạo ảnh trống để chứa cả hai tay
                h, w, _ = frame.shape
                full_hand_image = np.zeros((h, w, 3), dtype=np.uint8)

                for j, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Lấy tọa độ bounding box của tay
                    x_min = w
                    y_min = h
                    x_max = y_max = 0

                    for lm in hand_landmarks.landmark:
                        x, y = int(lm.x * w), int(lm.y * h)
                        x_min = min(x, x_min)
                        y_min = min(y, y_min)
                        x_max = max(x, x_max)
                        y_max = max(y, y_max)

                    # Thêm padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(w, x_max + padding)
                    y_max = min(h, y_max + padding)

                    # Cắt ảnh tay
                    hand_img = frame[y_min:y_max, x_min:x_max]

                    # Vẽ ảnh tay vào ảnh trống
                    full_hand_image[y_min:y_max, x_min:x_max] = hand_img

                    # Vẽ landmarks lên ảnh gốc
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                # Lưu ảnh chứa cả 2 bàn tay vào thư mục
                if full_hand_image.size > 0:
                    i += 1
                    if i % 5 == 0:
                        cv.imwrite(f'DATASET/{Class}/{i}_both_hands.png', full_hand_image)

            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q') or i > 500:
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    get_image()
