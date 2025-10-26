"""Hand Tracking Module"""
import cv2
import mediapipe as mp
import time
import math

class HandDetector: # Renamed to follow Python class naming convention (PascalCase)
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        # Initializing the Hands model with class-level attributes
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        # Landmark IDs for the tips of the fingers: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []

    def findHands(self, img, draw=True):
        # Convert to RGB, process, and store results as a class attribute
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # Draw landmarks and connections
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        xList, yList, bbox = [], [], []
        self.lmList = []
        
        # Check if any hand was detected and if the specified hand number exists
        if self.results.multi_hand_landmarks and len(self.results.multi_hand_landmarks) > handNo:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape
            
            for id, lm in enumerate(myHand.landmark):
                # Convert normalized coordinates (0 to 1) to pixel values
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmList.append([id, cx, cy])
                
                if draw and id in self.tipIds: # Only draw tip points for better performance
                    cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Calculate Bounding Box
            if xList and yList: # Check if lists are not empty
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = xmin, ymin, xmax, ymax

                if draw:
                    # Draw a rectangle around the hand
                    cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                                  (0, 255, 0), 2)

        return self.lmList, bbox

    def fingersUp(self):
        fingers = []
        if not self.lmList:
            # Return an array of 5 zeros if no hand is detected, for consistent array size
            return [0] * 5

        # --- Thumb check (based on x-coordinate) ---
        # Right Hand: Tip (4) is further right than PIV (3) -> finger is up
        # Left Hand: Tip (4) is further left than PIV (3) -> finger is up
        # A more robust check might compare the thumb tip (4) to the wrist (0) or index base (5), but 
        # for a single hand (maxHands=1), the current logic is often sufficient but can be refined.
        # For simplicity, we'll keep the original logic, assuming the hand is oriented correctly for right hand use.
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        # --- Four Fingers check (based on y-coordinate) ---
        # Tip (e.g., 8) is above the PIP joint (e.g., 6) -> finger is up
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def findDistance(self, p1, p2, img, draw=True, r=10, t=3):
        if not self.lmList:
            return 0, img, []

        try:
            x1, y1 = self.lmList[p1][1:]
            x2, y2 = self.lmList[p2][1:]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            length = math.hypot(x2 - x1, y2 - y1)
            lineInfo = [x1, y1, x2, y2, cx, cy]
            
            # The drawing is now primarily handled in the main script for better control, 
            # but we keep the option here.
            if draw:
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
                cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)

            return length, img, lineInfo
        except IndexError:
            # This can happen if p1 or p2 is outside lmList bounds
            print(f"Error: Landmark index out of bounds (p1={p1}, p2={p2})")
            return 0, img, []


# Removed the main() function from the module file, 
# as it's better placed in the main project file for clarity.