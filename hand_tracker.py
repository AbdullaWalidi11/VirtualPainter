import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

class HandTracker:
    def __init__(self, model_path='models/hand_landmarker.task'):
        """
        Initializes the MediaPipe Hand Landmarker using the modern Tasks API.
        """
        # 1. Configure the TFLite model deployment
        base_options = python.BaseOptions(model_asset_path=model_path)
        
        # 2. Set the detection parameters
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,          # We need 2 hands for the dynamic thickness feature!
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # 3. Create the landmarker object
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def find_hands(self, rgb_frame):
        """
        Passes the frame to the model and returns the detection results.
        """
        # MediaPipe requires its own specific Image format for the Tasks API
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Perform the inference
        detection_result = self.landmarker.detect(mp_image)
        
        return detection_result



    def get_positions(self, detection_result, frame_width, frame_height, hand_index=0):
        """
        Extracts pixel coordinates for all 21 landmarks of a specific hand.
        Returns a list of lists: [ [id, x, y], [id, x, y], ... ]
        """
        landmark_list = []
        
        # Check if any hands were actually detected in this frame
        if detection_result and detection_result.hand_landmarks:
            # Check if the specific hand we want (e.g., hand 0) is present
            if hand_index < len(detection_result.hand_landmarks):
                hand_landmarks = detection_result.hand_landmarks[hand_index]
                
                # Loop through all 21 points and convert to pixels
                for id, landmark in enumerate(hand_landmarks):
                    cx, cy = int(landmark.x * frame_width), int(landmark.y * frame_height)
                    landmark_list.append([id, cx, cy])
                    
        return landmark_list

    def fingers_up(self, landmark_list):
        """
        Determines which fingers are open.
        Returns a list of 5 integers (1 for open, 0 for closed).
        e.g., [0, 1, 0, 0, 0] means only the index finger is up.
        """
        fingers = []
        # The IDs for the tips of the Thumb, Index, Middle, Ring, and Pinky
        tip_ids = [4, 8, 12, 16, 20]
        
        if len(landmark_list) != 0:
            # 1. Thumb Check
            # The thumb moves horizontally, so we check the X coordinate.
            # (Note: This simple X-check works perfectly for the right hand. 
            # It will be inverted for the left hand, but handles standard use cases well).
            if landmark_list[tip_ids[0]][1] > landmark_list[tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
                
            # 2. Four Fingers Check
            # These move vertically, so we check the Y coordinate.
            for id in range(1, 5):
                # If tip Y is less than the PIP joint Y, the finger is open
                if landmark_list[tip_ids[id]][2] < landmark_list[tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
                    
        return fingers    


