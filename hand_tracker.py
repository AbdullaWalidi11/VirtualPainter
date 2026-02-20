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