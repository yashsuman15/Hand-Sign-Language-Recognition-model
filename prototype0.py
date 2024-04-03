import pickle
import cv2
import mediapipe as mp
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore


class HandSignRecognitionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Sign Recognition")
        # self.setMinimumSize(1000, 600)
        self.setMaximumSize(1800, 900)

        # background image
        self.background_image = QtGui.QPixmap('interface images/5199419.jpg')
        self.background_label = QtWidgets.QLabel(self)
        self.background_label.setPixmap(self.background_image)
        self.background_label.setGeometry(0, 0, self.background_image.width(), self.background_image.height())

        # custom stylesheets for widgets
        self.setStyleSheet("""
                    QLabel {
                        font-size: 18px;
                        color: white;
                        background-color: rgba(0, 0, 0, 0.7);
                        padding: 10px;
                        border-radius: 5px;
                    }
                    QPushButton {
                        font-size: 16px;
                        color: white;
                        background-color: #007BFF;
                        padding: 8px 16px;
                        border: 2px solid #0056b3;
                        border-radius: 5px;
                    }
                    QPushButton:hover {
                        background-color: #0056b3;
                    }
                """)

        # load the model
        self.model_dict = pickle.load(open('models/model5.p', 'rb'))
        self.model = self.model_dict['model']

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(max_num_hands=1, static_image_mode=True, min_detection_confidence=0.3)

        self.labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                            19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

        # hand sign image
        self.image_path = 'image/image.jpg'
        self.image = cv2.imread(self.image_path)
        self.image = cv2.resize(self.image, (600, 600))

        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Add heading
        self.heading_label = QtWidgets.QLabel("Hand Sign Detection and Recognition")
        self.main_layout.addWidget(self.heading_label, alignment=QtCore.Qt.AlignHCenter)

        # Update styles for labels
        self.heading_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #FFD700; padding: 10px;")

        self.top_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.top_layout)

        self.static_image_label = QtWidgets.QLabel()
        self.static_image_label.setPixmap(QtGui.QPixmap(self.image_path))
        self.top_layout.addWidget(self.static_image_label)
        # Resize the label to change the dimensions of the displayed image
        self.static_image_label.resize(640, 480)

        # Adjust layout spacing and alignment
        self.top_layout.setSpacing(20)
        self.top_layout.setAlignment(QtCore.Qt.AlignHCenter)

        self.vertical_layout = QtWidgets.QVBoxLayout()

        # webcam feed widget
        self.webcam_label = QtWidgets.QLabel()
        self.webcam_label.setFixedSize(640, 480)  # Set a fixed size for the video feed box
        self.webcam_label.setStyleSheet("border: 2px solid black;")
        self.vertical_layout.addWidget(self.webcam_label, alignment=QtCore.Qt.AlignCenter)

        self.prediction_label = QtWidgets.QLabel("Prediction: ")
        self.prediction_label.setStyleSheet(
                                        "font-size: 24px; font-weight: bold; color: white; "
                                        "background-color: rgba(255, 69, 0, 0.5); padding: 10px; border-radius: 5px;")
        self.vertical_layout.addWidget(self.prediction_label)

        self.top_layout.addLayout(self.vertical_layout)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.button_layout)

        self.open_webcam_button = QtWidgets.QPushButton("Open Webcam")
        self.open_webcam_button.clicked.connect(self.start_webcam)
        self.button_layout.addWidget(self.open_webcam_button)

        self.close_webcam_button = QtWidgets.QPushButton("Close Webcam")
        self.close_webcam_button.clicked.connect(self.close_webcam)
        self.close_webcam_button.setEnabled(False)
        self.button_layout.addWidget(self.close_webcam_button)

        # Add icons to button
        self.open_webcam_button.setIcon(QtGui.QIcon('interface images/webcam-30-64.png'))
        self.close_webcam_button.setIcon(QtGui.QIcon('interface images/close.png'))

        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_webcam_feed)

        # webcam label when closed
        self.default_webcam_label_text = "Webcam is closed"
        self.webcam_label.setText(self.default_webcam_label_text)
        self.webcam_label.setStyleSheet("border: 2px solid black; font-size: 20px; font-weight: bold; padding: 10px; color: red")

        # Adjust layout spacing and alignment
        self.top_layout.setSpacing(20)
        self.button_layout.setAlignment(QtCore.Qt.AlignHCenter)



    def start_webcam(self):
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.open_webcam_button.setEnabled(False)
            self.close_webcam_button.setEnabled(True)
            self.webcam_label.setText("")
            self.webcam_label.setStyleSheet("border: 2px solid black;")

    def close_webcam(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.timer.stop()
            self.open_webcam_button.setEnabled(True)
            self.close_webcam_button.setEnabled(False)
            self.webcam_label.setText(self.default_webcam_label_text)
            self.webcam_label.setStyleSheet("border: 2px solid black; font-size: 20px; color: red; font-weight: bold; padding: 10px")
            self.prediction_label.setText("Prediction: ")

    def update_webcam_feed(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)

            data_aux = []
            x_ = []
            y_ = []

            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                for hand_landmarks in results.multi_hand_landmarks:
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        x_.append(x)
                        y_.append(y)

                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                if self.model:
                    prediction = self.model.predict([np.asarray(data_aux)])
                    confidence = np.max(self.model.predict_proba([np.asarray(data_aux)])) * 100
                    predicted_character = self.labels_dict[int(prediction[0])]
                    self.prediction_label.setText(f"Prediction: {predicted_character}               (Confidence: {confidence:.2f}%)")
            else:
                hand_detected = False
                self.prediction_label.setText("Prediction: Hands not detected")

            qt_img = self.convert_cv_to_qt(frame)
            self.webcam_label.setPixmap(QtGui.QPixmap.fromImage(qt_img))
            self.webcam_label.setStyleSheet("border: 2px solid black;")

    def convert_cv_to_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return convert_to_Qt_format


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = HandSignRecognitionApp()
    window.show()
    app.exec_()
#%%
