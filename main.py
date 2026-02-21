import cv2
import numpy as np
import math # Import math for the distance calculation
import time # Import time to get strictly increasing timestamps
from hand_tracker import HandTracker

def main():
    cap = cv2.VideoCapture(0)
    success, frame = cap.read()
    if not success:
        print("Camera not found.")
        return
    h, w, c = frame.shape
    
    canvas = np.zeros((h, w, c), np.uint8)
    tracker = HandTracker(model_path='models/hand_landmarker.task')
    
    draw_color = (0, 0, 255) 
    brush_thickness = 15
    xp, yp = 0, 0 
    
    print("AI Painter running. Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get current timestamp in milliseconds
        timestamp_ms = int(time.time() * 1000)
        results = tracker.find_hands(rgb_frame, timestamp_ms)
        
        # --- NEW LOGIC: Check how many hands are present ---
        if results and results.hand_landmarks:
            num_hands = len(results.hand_landmarks)
            
            # STATE 4: Bimanual Thickness Scaling (2 Hands Detected)
            if num_hands == 2:
                # Reset previous drawing point so we don't accidentally draw a line
                xp, yp = 0, 0 
                
                # Extract the landmark lists for BOTH hands
                hand1_list = tracker.get_positions(results, w, h, hand_index=0)
                hand2_list = tracker.get_positions(results, w, h, hand_index=1)
                
                if len(hand1_list) > 0 and len(hand2_list) > 0:
                    # Get X, Y of the index finger tips (Landmark 8) for both hands
                    _, x1, y1 = hand1_list[8]
                    _, x2, y2 = hand2_list[8]
                    
                    # Calculate the distance mathematically
                    length = math.hypot(x2 - x1, y2 - y1)
                    
                    # Map the pixel distance to a brush thickness range
                    # e.g., A distance of 50px-300px becomes a thickness of 5px-50px
                    brush_thickness = int(np.interp(length, [50, 300], [5, 50]))
                    
                    # Visual Feedback: Draw a line connecting the two fingers
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    
                    # Visual Feedback: Draw a circle in the middle showing the current thickness
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.circle(frame, (cx, cy), brush_thickness, draw_color, cv2.FILLED)
                    cv2.putText(frame, f"Thickness: {brush_thickness}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

            # Standard States: Drawing, Selecting, Erasing (1 Hand Detected)
            elif num_hands == 1:
                landmark_list = tracker.get_positions(results, w, h, hand_index=0)
                
                if len(landmark_list) != 0:
                    _, x1, y1 = landmark_list[8]
                    fingers = tracker.fingers_up(landmark_list)
                    
                    # Selection Mode (2, 3, or 4 fingers up)
                    if fingers[1] == 1 and (fingers[2] == 1 or fingers[3] == 1 or fingers[4] == 1) and fingers[0] == 0:
                        xp, yp = 0, 0 
                        if fingers == [0, 1, 1, 0, 0]:
                            draw_color = (0, 0, 255) # Two fingers
                            cv2.putText(frame, "Color: RED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
                        elif fingers == [0, 1, 1, 1, 0]:
                            draw_color = (0, 255, 0) 
                            cv2.putText(frame, "Color: GREEN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
                        elif fingers == [0, 1, 1, 1, 1]:
                            draw_color = (255, 0, 0) # Four fingers
                            cv2.putText(frame, "Color: BLUE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
                    
                    # Eraser Mode (5 fingers up)
                    elif fingers == [1, 1, 1, 1, 1]:
                        xp, yp = 0, 0
                        cv2.putText(frame, "Mode: ERASER", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        cv2.circle(frame, (x1, y1), brush_thickness + 20, (0, 0, 0), cv2.FILLED)
                        cv2.circle(canvas, (x1, y1), brush_thickness + 20, (0, 0, 0), cv2.FILLED)
                    
                    # Drawing Mode (Only Index up)
                    elif fingers == [0, 1, 0, 0, 0]:
                        cv2.circle(frame, (x1, y1), 10, draw_color, cv2.FILLED)
                        cv2.putText(frame, "Mode: DRAW", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
                        
                        if xp == 0 and yp == 0:
                            xp, yp = x1, y1
                            
                        cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                        xp, yp = x1, y1
                        
        else:
            xp, yp = 0, 0
            
        frame_with_canvas = np.where(canvas != 0, canvas, frame)
        cv2.imshow('AI Virtual Painter', frame_with_canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()