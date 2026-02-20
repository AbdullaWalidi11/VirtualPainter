import cv2
import numpy as np
from hand_tracker import HandTracker

def main():
    cap = cv2.VideoCapture(0)
    
    # Read the first frame to get dimensions
    success, frame = cap.read()
    if not success:
        print("Camera not found.")
        return
    h, w, c = frame.shape
    
    # Initialize the Virtual Canvas
    canvas = np.zeros((h, w, c), np.uint8)
    
    # Initialize Tracker
    tracker = HandTracker(model_path='models/hand_landmarker.task')
    
    # State Machine Variables
    draw_color = (0, 0, 255) # Default to Red (BGR format)
    brush_thickness = 15
    xp, yp = 0, 0 # Previous coordinates for drawing smooth lines
    
    print("AI Painter running. Press 'q' to quit.")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Detect hands
        results = tracker.find_hands(rgb_frame)
        
        # 2. Extract positions for Hand 0 (the first detected hand)
        landmark_list = tracker.get_positions(results, w, h, hand_index=0)
        
        if len(landmark_list) != 0:
            # Get the X and Y coordinates of the Index Finger Tip (Landmark 8)
            _, x1, y1 = landmark_list[8]
            
            # 3. Check which fingers are up
            fingers = tracker.fingers_up(landmark_list)
            
            # --- THE STATE MACHINE ---
            
            # STATE 1: Selection Mode / Change Colors (2, 3, or 4 fingers up)
            # We check if index is up AND any other fingers are up
            if fingers[1] == 1 and (fingers[2] == 1 or fingers[3] == 1 or fingers[4] == 1) and fingers[0] == 0:
                xp, yp = 0, 0 # Reset previous points so it doesn't draw a line when you switch modes
                
                # 2 Fingers: Red
                if fingers == [0, 1, 1, 0, 0]:
                    draw_color = (0, 0, 255)
                    cv2.putText(frame, "Color: RED", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
                # 3 Fingers: Green
                elif fingers == [0, 1, 1, 1, 0]:
                    draw_color = (0, 255, 0)
                    cv2.putText(frame, "Color: GREEN", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
                # 4 Fingers: Blue
                elif fingers == [0, 1, 1, 1, 1]:
                    draw_color = (255, 0, 0)
                    cv2.putText(frame, "Color: BLUE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
            
            # STATE 2: Eraser Mode (5 fingers up - Open Palm)
            elif fingers == [1, 1, 1, 1, 1]:
                xp, yp = 0, 0
                draw_color = (0, 0, 0) # Black acts as an eraser on our canvas
                cv2.putText(frame, "Mode: ERASER", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.circle(frame, (x1, y1), brush_thickness + 20, draw_color, cv2.FILLED)
                cv2.circle(canvas, (x1, y1), brush_thickness + 20, draw_color, cv2.FILLED) # Erase on canvas
            
            # STATE 3: Drawing Mode (ONLY Index finger is up)
            elif fingers == [0, 1, 0, 0, 0]:
                cv2.circle(frame, (x1, y1), 10, draw_color, cv2.FILLED) # Visual indicator on finger
                cv2.putText(frame, "Mode: DRAW", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, draw_color, 2)
                
                # If this is the first frame of drawing, start the line at the current position
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                    
                # Draw the line on the transparent canvas
                cv2.line(canvas, (xp, yp), (x1, y1), draw_color, brush_thickness)
                
                # Update previous positions for the next frame
                xp, yp = x1, y1
                
        else:
            # If no hand is detected, reset the previous points so lines don't drag across the screen when your hand returns
            xp, yp = 0, 0
            
        # 4. Blend the Canvas onto the Video Frame
        # Because our eraser draws black (0,0,0), this np.where logic automatically makes those erased spots transparent again!
        frame_with_canvas = np.where(canvas != 0, canvas, frame)
        
        cv2.imshow('AI Virtual Painter', frame_with_canvas)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()