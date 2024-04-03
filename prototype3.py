import pickle
import cv2
import mediapipe as mp
import numpy as np
from PyQt5 import QtWidgets, QtGui, QtCore


# main application window
class HandSignRecognitionApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand Sign Detection and Recognition")  # Set the window title
        self.setWindowIcon(QtGui.QIcon('interface images/hand-recognition.png'))
        self.setMaximumSize(1800, 900)  # Set the maximum size of the window

        # Load and set the background image
        self.background_image = QtGui.QPixmap('interface images/5199419.jpg')
        self.background_label = QtWidgets.QLabel(self)
        self.background_label.setPixmap(self.background_image)
        self.background_label.setGeometry(0, 0, self.background_image.width(), self.background_image.height())

        # Set custom stylesheets for widgets
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

        # Load the pre-trained model
        self.model_dict = pickle.load(open('models/model6.p', 'rb'))
        self.model = self.model_dict['model']

        # Initialize MediaPipe Hands solution
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(max_num_hands=1, static_image_mode=True, min_detection_confidence=0.5)

        # Define a dictionary to map prediction labels to characters
        self.labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
                            10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
                            19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

        # Load and resize the hand sign image
        self.image_path = 'image/image.jpg'
        self.image = cv2.imread(self.image_path)
        self.image = cv2.resize(self.image, (600, 600))

        # Set up the main layout and central widget
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QVBoxLayout()
        self.central_widget.setLayout(self.main_layout)

        # Add a heading label
        self.heading_label = QtWidgets.QLabel("Hand Sign Detection and Recognition")
        self.heading_label.setStyleSheet("font-size: 34px; font-weight: bold; color: #FFD700; padding: 10px;")
        self.main_layout.addWidget(self.heading_label, alignment=QtCore.Qt.AlignHCenter)

        self.content_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.content_layout)

        # Left layout for static image
        self.left_layout = QtWidgets.QVBoxLayout()
        self.content_layout.addLayout(self.left_layout)

        self.static_image_label = QtWidgets.QLabel()
        self.static_image_label.setPixmap(QtGui.QPixmap(self.image_path))
        self.left_layout.addWidget(self.static_image_label)
        self.static_image_label.resize(640, 480)

        # Center layout for webcam feed
        self.center_layout = QtWidgets.QVBoxLayout()
        self.content_layout.addLayout(self.center_layout)

        self.webcam_label = QtWidgets.QLabel()
        self.webcam_label.setFixedSize(640, 540)
        self.webcam_label.setStyleSheet("border: 2px solid blue;")
        self.center_layout.addWidget(self.webcam_label, alignment=QtCore.Qt.AlignCenter)

        # Right layout for landmarks and prediction
        self.right_layout = QtWidgets.QVBoxLayout()
        self.content_layout.addLayout(self.right_layout)

        self.landmarks_label = QtWidgets.QLabel()
        self.landmarks_label.setFixedSize(640, 480)
        self.landmarks_label.setStyleSheet("border: 2px solid black;")
        self.right_layout.addWidget(self.landmarks_label, alignment=QtCore.Qt.AlignCenter)

        self.prediction_label = QtWidgets.QLabel("Prediction: ")
        self.prediction_label.setStyleSheet(
            "font-size: 30px; font-weight: bold; color: white; "
            "background-color: rgba(255, 69, 0, 0.5); padding: 20px; border-radius: 10px;")
        self.right_layout.addWidget(self.prediction_label)

        # Add prediction box and buttons
        self.prediction_box = QtWidgets.QTextEdit()
        self.prediction_box.setStyleSheet("font-size: 26px; background-color: white; border: 1px solid gray;")
        self.right_layout.addWidget(self.prediction_box)

        self.button_layout = QtWidgets.QHBoxLayout()
        self.right_layout.addLayout(self.button_layout)

        self.clear_box_button = QtWidgets.QPushButton("Clear Box")
        self.clear_box_button.setStyleSheet("background-color: red; color: white;")  # Change the color of the button
        self.clear_box_button.clicked.connect(self.clear_prediction_box)
        self.button_layout.addWidget(self.clear_box_button)

        self.remove_last_button = QtWidgets.QPushButton("Remove Last")
        self.remove_last_button.clicked.connect(self.remove_last_prediction)
        self.button_layout.addWidget(self.remove_last_button)

        # Add buttons for opening and closing the webcam
        self.webcam_button_layout = QtWidgets.QHBoxLayout()
        self.main_layout.addLayout(self.webcam_button_layout)

        self.open_webcam_button = QtWidgets.QPushButton("Open Webcam")
        self.open_webcam_button.clicked.connect(self.start_webcam)
        self.webcam_button_layout.addWidget(self.open_webcam_button)

        self.close_webcam_button = QtWidgets.QPushButton("Close Webcam")
        self.close_webcam_button.clicked.connect(self.close_webcam)
        self.close_webcam_button.setEnabled(False)
        self.webcam_button_layout.addWidget(self.close_webcam_button)

        self.open_webcam_button.setIcon(QtGui.QIcon('interface images/webcam-30-64.png'))
        self.close_webcam_button.setIcon(QtGui.QIcon('interface images/close.png'))

        # Initialize webcam and timer
        self.cap = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_webcam_feed)

        self.default_webcam_label_text = "Webcam is closed"
        self.webcam_label.setText(self.default_webcam_label_text)
        self.webcam_label.setStyleSheet(
            "border: 2px solid black; font-size: 40px; "
            "font-weight: bold; padding: 10px; color: red; font-family: Error;")
        self.webcam_label.setAlignment(QtCore.Qt.AlignCenter)

        self.webcam_button_layout.setAlignment(QtCore.Qt.AlignHCenter)

        # Variable for tracking concatenation
        self.concatenated_prediction = False

    def start_webcam(self):
        # Open the webcam and start the timer to update the feed
        if not self.cap or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
            self.timer.start(30)
            self.open_webcam_button.setEnabled(False)
            self.close_webcam_button.setEnabled(True)
            self.webcam_label.setText("")
            self.webcam_label.setStyleSheet("border: 2px solid black;")

    def close_webcam(self):
        # Close the webcam and stop the timer
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.timer.stop()
            self.open_webcam_button.setEnabled(True)
            self.close_webcam_button.setEnabled(False)
            self.webcam_label.setText(self.default_webcam_label_text)
            self.webcam_label.setStyleSheet(
                "border: 2px solid black; font-size: 40px;"
                " color: red; font-weight: bold; padding: 10px; font-family: Error;")
            self.prediction_label.setText("Prediction: ")

    def update_webcam_feed(self):
        # Update the webcam feed and perform hand sign recognition
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            frame = cv2.flip(frame, 1)

            data_aux = []
            x_ = []
            y_ = []

            H, W, _ = frame.shape

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.hands.process(frame_rgb)

            landmarks_image = np.zeros_like(frame_rgb)

            if results.multi_hand_landmarks:
                hand_detected = True
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                    self.mp_drawing.draw_landmarks(
                        landmarks_image,
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

                    # Change prediction label color based on confidence
                    if confidence >= 65:
                        self.prediction_label.setStyleSheet(
                            "font-size: 30px; font-weight: bold; color: white; "
                            "background-color: rgba(0, 255, 0, 0.5); padding: 20px; border-radius: 10px;")
                    else:
                        self.prediction_label.setStyleSheet(
                            "font-size: 30px; font-weight: bold; color: white; "
                            "background-color: rgba(255, 69, 0, 0.5); padding: 20px; border-radius: 10px;")

                    self.prediction_label.setText(f"Prediction: {predicted_character} (Confidence: {confidence:.2f}%)")

                    self.concatenate_prediction(predicted_character, confidence)
            else:
                hand_detected = False
                self.prediction_label.setText("Prediction: Hands not detected")

            # Convert OpenCV image to Qt image for display
            qt_img = self.convert_cv_to_qt(frame)
            self.webcam_label.setPixmap(QtGui.QPixmap.fromImage(qt_img))
            self.webcam_label.setStyleSheet("border: 2px solid black;")

            landmarks_qt_img = self.convert_cv_to_qt(landmarks_image)
            self.landmarks_label.setPixmap(QtGui.QPixmap.fromImage(landmarks_qt_img))
            self.landmarks_label.setStyleSheet("border: 2px solid black;")

    def convert_cv_to_qt(self, cv_img):
        # Convert OpenCV image to Qt image format
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        return convert_to_Qt_format

    def concatenate_prediction(self, predicted_character, confidence):
        # Concatenate the prediction if the confidence is above 65% and the prediction was not previously concatenated
        current_text = self.prediction_box.toPlainText()
        req_confidence = 65
        if confidence >= req_confidence and not self.concatenated_prediction:
            if predicted_character != current_text[:-1]:
                self.prediction_box.setText(self.prediction_box.toPlainText() + predicted_character)
                self.concatenated_prediction = True
        elif confidence < req_confidence:
            self.concatenated_prediction = False

    def clear_prediction_box(self):
        # Clear the prediction box
        self.prediction_box.clear()
        self.concatenated_prediction = False

    def remove_last_prediction(self):
        # Remove the last concatenated value from the prediction box
        current_text = self.prediction_box.toPlainText()
        if current_text:
            self.prediction_box.setText(current_text[:-1])
            if not current_text[:-1]:
                self.concatenated_prediction = False


if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = HandSignRecognitionApp()
    window.show()
    app.exec_()
