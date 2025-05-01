import cv2
import mediapipe as mp
import pandas as pd
import os
import numpy as np


def image_processed(file_path):
    hand_img = cv2.imread(file_path)
    if hand_img is None:
        return np.zeros([1, 126], dtype=int)[0]  # Trả về 126 số 0 nếu không thể đọc ảnh

    # Chuyển ảnh sang RGB (do Mediapipe yêu cầu)
    img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)

    # Không cần flip vì ảnh đã crop tay sẵn rồi
    mp_hands = mp.solutions.hands

    # Chạy Mediapipe ở chế độ ảnh tĩnh, nhận diện tối đa 2 tay
    hands = mp_hands.Hands(static_image_mode=True,
                           max_num_hands=2,  # Nhận diện tối đa 2 tay
                           min_detection_confidence=0.7)

    output = hands.process(img_rgb)
    hands.close()

    try:
        multi_landmarks = output.multi_hand_landmarks
        landmark_list = []

        for hand_landmarks in multi_landmarks:
            for lm in hand_landmarks.landmark:
                landmark_list.extend([lm.x, lm.y, lm.z])

        # Nếu chỉ có 1 tay, thêm 63 số 0 vào để đảm bảo đủ 126 cột
        if len(multi_landmarks) == 1:
            landmark_list.extend([0] * 63)  # Thêm 63 cột nếu chỉ có 1 tay

        # Đảm bảo luôn có 126 đặc trưng (2 tay x 63 đặc trưng)
        return landmark_list[:126]  # Nếu có nhiều hơn 126, chỉ lấy 126 cột đầu tiên

    except:
        return [0] * 126  # Nếu không nhận diện được tay, trả về mảng toàn số 0


def make_csv():
    mypath = 'DATASET'
    with open('dataset.csv', 'a') as file_name:
        for each_folder in os.listdir(mypath):
            if each_folder.startswith('._'):
                continue

            for each_image in os.listdir(os.path.join(mypath, each_folder)):
                if each_image.startswith('._'):
                    continue

                label = each_folder
                file_path = os.path.join(mypath, each_folder, each_image)

                data = image_processed(file_path)

                # Kiểm tra nếu tất cả giá trị trong landmark là 0 thì bỏ qua dòng này
                if all(value == 0 for value in data):
                    continue  # Bỏ qua nếu không có tay trong ảnh

                try:
                    line = ",".join(map(str, data)) + f",{label}\n"
                except:
                    line = "0," * 126 + "None\n"  # Lưu dữ liệu khi có lỗi

                file_name.write(line)

    print("✅ CSV đã được tạo xong!")


if __name__ == "__main__":
    make_csv()
