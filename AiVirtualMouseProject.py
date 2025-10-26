import cv2
import numpy as np
import HandTrackingModule as htm
import autopy
import time


W_CAM, H_CAM = 640, 480          # Webcam resolution
FRAME_REDUCTION = 100            # Border size for the active screen area
SMOOTHENING = 9                  # Higher value for smoother movement (was 7)
CLICK_THRESHOLD = 35             # Reduced distance for a click (was 40)
#########################

pTime = 0
# Initialize previous and current locations for smoother movement
plocX, plocY = 0, 0
clocX, clocY = 0, 0

# 1. Setup Camera and Detector
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W_CAM)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H_CAM)
detector = htm.HandDetector(maxHands=1, detectionCon=0.75, trackCon=0.75) # Increased confidence
wScr, hScr = autopy.screen.size()

# Main Loop
while True:
    # 2. Find hand Landmarks
    success, img = cap.read()
    if not success:
        print("Camera not detected! Exiting...")
        break
    
    # Flip the image horizontally for a more intuitive "mirror" effect
    img = cv2.flip(img, 1)

    # Hand tracking
    img = detector.findHands(img, draw=False) # Skip drawing landmarks here for speed
    lmList, bbox = detector.findPosition(img, draw=False) # Skip drawing bbox here for speed

    if lmList:
        # Get the tip of the index (8) and middle (12) fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # Check which fingers are up
        fingers = detector.fingersUp()

        # Define the active area and draw it
        cv2.rectangle(img, (FRAME_REDUCTION, FRAME_REDUCTION), 
                      (W_CAM - FRAME_REDUCTION, H_CAM - FRAME_REDUCTION),
                      (255, 0, 255), 2)
        
        # --- MOVING MODE (Index Finger Up) ---
        if fingers[1] == 1 and fingers[2] == 0:
            
            # 5. Convert Coordinates
            x3 = np.interp(x1, (FRAME_REDUCTION, W_CAM - FRAME_REDUCTION), (0, wScr))
            y3 = np.interp(y1, (FRAME_REDUCTION, H_CAM - FRAME_REDUCTION), (0, hScr))

            # 6. Smoothen Values
            # Using Exponential Moving Average (EMA) for smoother results
            clocX = plocX + (x3 - plocX) / SMOOTHENING
            clocY = plocY + (y3 - plocY) / SMOOTHENING

            # 7. Move Mouse
            try:
                # autopy coordinates must be integers
                autopy.mouse.move(int(clocX), int(clocY)) 
            except Exception as e:
                # Non-critical error, continue operation but print for debugging
                print(f"Mouse move error: {e}") 

            # Draw circle at the index finger tip and update previous location
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY

        # --- CLICKING MODE (Index and Middle Fingers Up) ---
        elif fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img, draw=False) 
            
            # Draw line and circles only in click mode
            if lineInfo:
                # Unpack lineInfo: [x1, y1, x2, y2, cx, cy]
                cv2.line(img, (lineInfo[0], lineInfo[1]), (lineInfo[2], lineInfo[3]), (255, 0, 255), 3)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 0, 255), cv2.FILLED)

                # 10. Click mouse if distance is short
                if length < CLICK_THRESHOLD:
                    # Draw a distinctive circle to indicate a successful click
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    try:
                        autopy.mouse.click()
                    except Exception as e:
                        print(f"Mouse click error: {e}") 

    # 11. Frame Rate (Performance check)
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)

    # 12. Display
    cv2.imshow("AI Virtual Mouse", img)
    
    # Check for ESC key press to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()