import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
import mediapipe as mp
import math
import av

# ==============================================================================
# 1. Hand Detector Class (Adapted from HandTrackingModule.py)
# ==============================================================================

class HandDetector:
    """Detects and tracks hand landmarks using MediaPipe."""
    def __init__(self, mode=False, maxHands=1, detectionCon=0.75, trackCon=0.75):
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
        self.tipIds = [4, 8, 12, 16, 20]
        self.lmList = []
        self.results = None

    def findHands(self, img):
        """Processes the image to find hand landmarks."""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        return img

    def findPosition(self, img, handNo=0):
        """Extracts landmark coordinates."""
        self.lmList = []
        if self.results and self.results.multi_hand_landmarks and len(self.results.multi_hand_landmarks) > handNo:
            myHand = self.results.multi_hand_landmarks[handNo]
            h, w, c = img.shape
            
            for id, lm in enumerate(myHand.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def fingersUp(self):
        """Determines which fingers are straight up (open)."""
        fingers = [0] * 5
        if not self.lmList:
            return fingers

        # Thumb (assuming right hand orientation)
        if self.lmList[self.tipIds[0]][1] > self.lmList[self.tipIds[0] - 1][1]:
             # Simple check: Tip (4) is further right than PIV (3)
            fingers[0] = 1

        # Four Fingers (Index, Middle, Ring, Pinky)
        for id in range(1, 5):
            # Tip (e.g., 8) is above the PIP joint (e.g., 6)
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id] - 2][2]:
                fingers[id] = 1

        return fingers

    def findDistance(self, p1, p2, img):
        """Calculates distance between two landmarks (p1, p2)."""
        if len(self.lmList) < max(p1, p2) + 1:
            return 0, [0, 0, 0, 0, 0, 0]

        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        length = math.hypot(x2 - x1, y2 - y1)
        lineInfo = [x1, y1, x2, y2, cx, cy]
        return length, lineInfo

# ==============================================================================
# 2. Video Processing and Logic Class
# ==============================================================================

class VirtualMouseProcessor(VideoProcessorBase):
    """Processes video frames and applies virtual mouse logic."""

    def __init__(self, smoothening, frame_reduction, click_threshold, w_cam, h_cam):
        self.detector = HandDetector(maxHands=1)
        self.smoothening = smoothening
        self.frame_reduction = frame_reduction
        self.w_cam = w_cam
        self.h_cam = h_cam
        self.plocX, self.plocY = 0, 0
        self.clocX, self.clocY = 0, 0
        self.w_scr, self.h_scr = 1920, 1080  # Fixed simulation resolution

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Called for every incoming video frame."""
        # Convert frame to OpenCV format (BGR NumPy array)
        img = frame.to_ndarray(format="bgr24")
        
        # 1. Flip the image for mirror effect
        img = cv2.flip(img, 1)

        # 2. Find hand Landmarks
        img = self.detector.findHands(img)
        lmList = self.detector.findPosition(img)

        # Draw the control boundary box
        cv2.rectangle(img, 
                      (self.frame_reduction, self.frame_reduction), 
                      (self.w_cam - self.frame_reduction, self.h_cam - self.frame_reduction),
                      (255, 0, 255), 2)
        
        # Initialize status for display
        status = "Waiting for Hand..."
        cursor_pos_text = "N/A"
        
        if lmList:
            x1, y1 = lmList[8][1:]  # Index finger tip
            x2, y2 = lmList[12][1:] # Middle finger tip
            fingers = self.detector.fingersUp()

            # --- MOVING MODE (Index Finger Up) ---
            if fingers[1] == 1 and fingers[2] == 0:
                status = "Move Mode: Controlling Cursor"
                
                # Convert Coordinates to simulated screen space
                x3 = np.interp(x1, (self.frame_reduction, self.w_cam - self.frame_reduction), (0, self.w_scr))
                y3 = np.interp(y1, (self.frame_reduction, self.h_cam - self.frame_reduction), (0, self.h_scr))

                # Smoothen Values (EMA)
                self.clocX = self.plocX + (x3 - self.plocX) / self.smoothening
                self.clocY = self.plocY + (y3 - self.plocY) / self.smoothening

                # Draw visualization elements
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                
                cursor_pos_text = f"X: {int(self.clocX)}, Y: {int(self.clocY)}"
                self.plocX, self.plocY = self.clocX, self.clocY

            # --- CLICKING MODE (Index and Middle Fingers Up) ---
            elif fingers[1] == 1 and fingers[2] == 1:
                length, lineInfo = self.detector.findDistance(8, 12, img)
                
                # Draw line and midpoint circle
                cv2.line(img, (lineInfo[0], lineInfo[1]), (lineInfo[2], lineInfo[3]), (255, 0, 255), 3)
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 10, (0, 0, 255), cv2.FILLED)
                
                status = f"Click Mode: Distance {int(length)}px"

                # Check for Click
                if length < self.click_threshold:
                    # SIMULATE CLICK VISUAL (Green circle)
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 0), cv2.FILLED)
                    status = f"**CLICK DETECTED!** Distance {int(length)}px"
            
            # Draw landmarks in the stream (optional, but helpful for demo)
            if self.detector.results.multi_hand_landmarks:
                for handLms in self.detector.results.multi_hand_landmarks:
                    self.detector.mpDraw.draw_landmarks(img, handLms, self.detector.mpHands.HAND_CONNECTIONS)

        # Add the current status text to the video frame
        cv2.putText(img, status, (10, self.h_cam - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Simulated Pos: {cursor_pos_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Convert back to WebRTC frame format
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==============================================================================
# 3. Streamlit Application Layout
# ==============================================================================

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Virtual Mouse Demo",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🖐️ AI Virtual Mouse: Interactive Demo")
st.markdown(
    """
    This Streamlit application demonstrates the real-time hand tracking and gesture detection 
    logic of the AI Virtual Mouse project. Use the sliders on the sidebar to adjust 
    performance and sensitivity in real-time!
    
    *Note: This is a visualization demo. Actual mouse control (`autopy`) is disabled for security.*
    """
)

# --- Sidebar for Interactive Settings ---
with st.sidebar:
    st.header("⚙️ Configuration Settings")
    
    # 1. Smoothing Factor (For Stability)
    SMOOTHENING = st.slider(
        "Smoothening Factor", 
        min_value=1, 
        max_value=30, 
        value=9, 
        step=1,
        help="Higher values reduce jitter but introduce lag. Lower values are faster but less stable."
    )

    # 2. Frame Reduction (For Sensitivity)
    FRAME_REDUCTION = st.slider(
        "Boundary Reduction (px)", 
        min_value=50, 
        max_value=200, 
        value=100, 
        step=10,
        help="Defines the size of the purple active area. Higher values increase sensitivity."
    )
    
    # 3. Click Threshold (For Accuracy)
    CLICK_THRESHOLD = st.slider(
        "Click Threshold Distance (px)", 
        min_value=20, 
        max_value=80, 
        value=35, 
        step=5,
        help="Distance between Index and Middle finger tips required to register a click. Lower is more accurate, higher is easier."
    )
    
    st.markdown("---")
    st.markdown("### Hand Gestures")
    st.markdown("1. **Move Mode:** Index Finger Up")
    st.markdown("2. **Click Mode:** Index Finger + Middle Finger Pinch")
    st.markdown("---")
    st.markdown("Built by: Atiq Umer")


# --- Main Application Area ---
# Set fixed camera resolution for consistent mapping
W_CAM, H_CAM = 640, 480

# Configure WebRTC to access the user's camera
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Start the WebRTC streamer with our custom processor
webrtc_streamer(
    key="virtual-mouse-demo",
    video_processor_factory=lambda: VirtualMouseProcessor(
        smoothening=SMOOTHENING, 
        frame_reduction=FRAME_REDUCTION, 
        click_threshold=CLICK_THRESHOLD,
        w_cam=W_CAM,
        h_cam=H_CAM
    ),
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_html_attrs={"style": {"width": f"{W_CAM}px", "height": f"{H_CAM}px", "margin": "0 auto"}},
)

st.markdown("---")
st.subheader("Simulated Output")
st.info(
    "Observe the status text inside the video feed and the simulated position to test the accuracy of your gestures against the configured settings."
)

st.success(
    f"To run the actual mouse control on your desktop, save your final settings (Smoothening: {SMOOTHENING}, Reduction: {FRAME_REDUCTION}, Click: {CLICK_THRESHOLD}) and apply them to your local `AiVirtualMouse.py` script."
)
