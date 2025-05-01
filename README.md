Hand Gesture Recognition with MediaPipe
This project uses MediaPipe and a machine learning model to recognize hand gestures in real-time via webcam.

📁 Project Structure
get_data.py: Collects gesture data (images) for training.

get_image.py: Captures static images from webcam or video.

hands_gesture_recog.py: Runs real-time hand gesture recognition using the trained model.

model.pkl: Saved machine learning model.

requirements.txt: Lists all dependencies to run this project.

test_mediapipe.py: Tests if MediaPipe and webcam setup work correctly.

train.ipynb: Jupyter Notebook for training the hand gesture model.

.gitignore: Specifies files to ignore in version control.

README.md: This file.

🚀 Quick Start
Install dependencies:
pip install -r requirements.txt
Collect training data:
python get_data.py
Train the model:

Open train.ipynb and run all cells in order.
Run gesture recognition:
python hands_gesture_recog.py

📹 Demo Video
👉 You can add your demo video link or embed below:
[Insert YouTube or local video link here]

📦 Requirements
Python 3.7+
OpenCV
MediaPipe
scikit-learn
numpy


💡 Notes
Use test_mediapipe.py to ensure your webcam and MediaPipe setup are working.

The trained model is saved as model.pkl and can be reused without retraining.
