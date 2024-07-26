import cv2
import mediapipe as mp
import time


class handDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 2, (255, 0, 255), cv2.FILLED)
        return lmList


def countFingers(lmList):
    tipIds = [4, 8, 12, 16, 20]  # Indices for the fingertips
    fingers = []

    # Thumb
    if lmList[3][2] > lmList[4][2]:  # y-coordinate comparison for thumb
        fingers.append(1)  # Thumb is open
    else:
        fingers.append(0)  # Thumb is closed

    # 4 Fingers
    for id in range(1, 5):
        if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:  # y-coordinate comparison
            fingers.append(1)
        else:
            fingers.append(0)

    return sum(fingers)


def main():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = cv2.flip(img, 1)  # Mirror the image
        img = detector.findHands(img)
        hands = detector.results.multi_hand_landmarks
        totalFingers = 0

        if hands:
            for handLms in hands:
                lmList = detector.findPosition(img, handNo=hands.index(handLms))
                if len(lmList) > 0:
                    totalFingers += countFingers(lmList)

        # Define the box parameters
        box_size = 140
        rect_x1, rect_y1 = 0, img.shape[0] - box_size
        rect_x2, rect_y2 = box_size, img.shape[0]

        # Draw a light red rectangle
        cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 255), 2)  # Light red outline

        # Prepare the text
        text = str(totalFingers)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        font_thickness = 2
        color = (255, 0, 0)  # Blue text

        # Find text size to center it
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size

        # Calculate text position to center it in the rectangle
        text_x = rect_x1 + (rect_x2 - rect_x1 - text_w) // 2
        text_y = rect_y1 + (rect_y2 - rect_y1 + text_h) // 2

        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, font_thickness)

        # Display FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (18, 78), cv2.FONT_HERSHEY_SIMPLEX, 3, (136, 34, 186), 1)

        # Show the image
        cv2.imshow("Image", img)

        # Exit if 'Esc' key is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
