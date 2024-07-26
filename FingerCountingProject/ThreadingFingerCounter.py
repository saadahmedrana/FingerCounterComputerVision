import cv2
import mediapipe as mp
import time
import threading


class HandDetector:
    def __init__(self, mode=False, maxHands=4, detectionCon=0.5, trackCon=0.5):
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
                    cv2.circle(img, (cx, cy), 2, (255, 0, 0), cv2.FILLED)
        return lmList


class VideoCaptureThread(threading.Thread):
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
        self.cap = cv2.VideoCapture(0)
        self.frame = None
        self.running = True

    def run(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                self.frame = cv2.flip(frame, 1)  # Mirror the image

    def stop(self):
        self.running = False
        self.cap.release()


class ImageProcessingThread(threading.Thread):
    def __init__(self, detector, video_thread):
        super().__init__()
        self.detector = detector
        self.video_thread = video_thread
        self.running = True

    def run(self):
        pTime = 0
        while self.running:
            if self.video_thread.frame is not None:
                img = self.video_thread.frame.copy()
                img = self.detector.findHands(img)
                lmList = self.detector.findPosition(img)
                totalFingers = 0

                if len(lmList) != 0:
                    hands = self.detector.results.multi_hand_landmarks
                    if hands:
                        for handLms in hands:
                            fingers = []

                            # Thumb
                            if lmList[tipIds[0]][2] < lmList[tipIds[0] - 1][2]:
                                fingers.append(1)
                            else:
                                fingers.append(0)

                            # 4 Fingers
                            for id in range(1, 5):
                                if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                                    fingers.append(1)
                                else:
                                    fingers.append(0)

                            totalFingers += sum(fingers)

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
                cv2.putText(img, str(int(fps)), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (136, 34, 186), 1)

                # Show the image
                cv2.imshow("Image", img)

            # Exit if 'Esc' key is pressed
            if cv2.waitKey(1) & 0xFF == 27:
                self.running = False

        cv2.destroyAllWindows()


def main():
    global tipIds
    tipIds = [4, 8, 12, 16, 20]  # Indices for the fingertips

    detector = HandDetector()
    video_thread = VideoCaptureThread(detector)
    video_thread.start()

    processing_thread = ImageProcessingThread(detector, video_thread)
    processing_thread.start()

    try:
        processing_thread.join()
    except KeyboardInterrupt:
        pass
    finally:
        video_thread.stop()
        video_thread.join()


if __name__ == "__main__":
    main()
