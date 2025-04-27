import numpy as np
import cv2 as cv
from pathlib import Path
import mediapipe as mp


def get_image():
    Class = 'c'
    Path('DATASET/' + Class).mkdir(parents=True, exist_ok=True)
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    i = 0
    with mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
    ) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Lật ảnh
            frame = cv.flip(frame, 1)

            # Mediapipe xử lý ảnh
            image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            frame = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            # Vẽ landmarks nếu phát hiện tay
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style()
                    )

            i += 1
            if i % 5 == 0:
                cv.imwrite('DATASET/' + Class + '/' + str(i) + '.png', frame)

            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q') or i > 500:
                break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    get_image()
