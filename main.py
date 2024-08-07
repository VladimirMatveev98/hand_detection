import cv2
import time

import mediapipe as mp

capture = cv2.VideoCapture(0)
hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1,
                                 min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5)

mpDraw = mp.solutions.drawing_utils

while True:
    status, image = capture.read()
    result = hands.process(image)
    if result.multi_hand_landmarks:
        for id, coords in enumerate(result.multi_hand_landmarks[0].landmark):
            h, w, _ = image.shape
            x, y = int(coords.x * w), int(coords.y * h)
            cv2.circle(image, (x, y), 3, (0, 255, 0))
            print(id)
            print(coords)
            if id == 8:
                cv2.circle(image, (x, y), 15, (0, 255, 0))

        mpDraw.draw_landmarks(image, result.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("head_tracking", image)
    print(status)
    cv2.waitKey(1)
    time.sleep(0.1)
