import os
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt
#画像読み込み
mp_hands = mp.solutions.hands

DIRECTORY = ".\picture\hand2.jpg"
hand_img = cv2.imread(DIRECTORY)
height, widen = hand_img.shape[:2]

result_img = cv2.resize(hand_img,(int(widen*3), int(height*3)))

hands = mp.solutions.hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
)

results = hands.process(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
results.multi_handedness

landmark_list = ['WRIST', 'THUMP_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP', 'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP', 'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
for hand_landmarks in results.multi_hand_landmarks:
    for i in range(21):
        print(landmark_list[i])
        print(hand_landmarks.landmark[i])

annotated_image = result_img.copy()
for hand_landmarks in results.multi_hand_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
            mp.solutions.drawing_styles.get_default_hand_connections_style()
            )
result_img = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

#画像表示
cv2.imshow("CarrotCake",result_img)
cv2.waitKey(0)
