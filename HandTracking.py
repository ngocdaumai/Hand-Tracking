import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDrawing = mp.solutions.drawing_utils

previousTime = 0
currentTime = 0
while True:
    _, image = capture.read()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(imageRGB)
    check_hand_appearance = results.multi_hand_landmarks
    # print(check_hand_appearance)
    if check_hand_appearance:
        for handLms in check_hand_appearance:
            mpDrawing.draw_landmarks(image, handLms, mpHands.HAND_CONNECTIONS)
            for id, lm in enumerate(handLms.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if id == 8:
                    cv2.circle(image, (cx, cy), 5, (255, 0, 225), cv2.FILLED)
    currentTime = time.time()
    fps = 1 / (currentTime - previousTime)
    previousTime = currentTime

    cv2.putText(image, str(int(fps)), (20, 50), cv2.FONT_HERSHEY_PLAIN, 2,
                (0, 0, 255), 2)
    cv2.imshow("Image", image)
    if cv2.waitKey(1)==ord("q"):
        break