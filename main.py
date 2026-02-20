import cv2
import numpy as np
from hand_tracker import HandTracker # Importing the class we just wrote!

def main():
    # 1. Initialize the Webcam
    cap = cv2.VideoCapture(0)
    
    # Briefly read the first frame to get the exact height (h) and width (w)
    success, frame = cap.read()
    if not success:
        print("Camera not found.")
        return
    h, w, c = frame.shape
    
    # 2. Initialize the Virtual Canvas
    # This creates a completely black matrix (filled with zeros) exactly the size of your webcam feed
    canvas = np.zeros((h, w, c), np.uint8)
    
    # 3. Initialize our Custom Hand Tracker
    tracker = HandTracker(model_path='models/hand_landmarker.task')
    
    print("Camera running. Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        # Flip the frame horizontally so it feels like looking in a mirror
        frame = cv2.flip(frame, 1)
        
        # MediaPipe requires RGB format, so we convert the OpenCV BGR frame
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 4. Detect hands
        # We pass the frame to our tracker. Later we will extract coordinates from 'results'
        results = tracker.find_hands(rgb_frame)
        
        # --- TODO: Finger counting and state machine logic will go here ---
        
        # For testing the canvas right now, let's draw a static blue circle in the center of the canvas
        cv2.circle(canvas, (w//2, h//2), 100, (255, 0, 0), cv2.FILLED) 
        
        # 5. Blend the Canvas onto the Video Frame
        # This is where the magic happens. We use Numpy to merge the two matrices.
        frame_with_canvas = np.where(canvas != 0, canvas, frame)
        
        # 6. Display the final merged output
        cv2.imshow('AI Virtual Painter', frame_with_canvas)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()