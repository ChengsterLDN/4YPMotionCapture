"""
Motion tracking with proper segmented lines
Every disappearance of the marker ends the current line.
A new line starts independently when the marker reappears.
Input: video file sample3.mp4
Output: output_yellow_calibrated.mp4
"""

import cv2
import numpy as np
import math
from collections import deque

# Calibration values
W_real = 0.076
D_known = 0.4
W_px_known = 289
focal_px = (W_px_known * D_known) / W_real

HSV_RANGES = {'yellow': [((20, 100, 100), (35, 255, 255))]}  # yellow range

def make_mask(hsv, colour):
    ranges = HSV_RANGES[colour]
    mask = None
    for lo, hi in ranges:
        low = np.array(lo, dtype=np.uint8)
        high = np.array(hi, dtype=np.uint8)
        m = cv2.inRange(hsv, low, high)
        mask = m if mask is None else cv2.bitwise_or(mask, m)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

def main():
    cap = cv2.VideoCapture('sample3.mp4')
    if not cap.isOpened():
        raise SystemExit('Cannot open video source')
    
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter('output_yellow_calibrated.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    lines = []  # list of line segments
    current_line = None
    total_distance_m = 0.0
    prev_pt = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
        blurred = cv2.GaussianBlur(frame, (5,5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = make_mask(hsv, 'yellow')
        
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centre = None
        Z = None
        metres_per_pixel = None
        
        if contours:
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) >= 150:
                M = cv2.moments(c)
                if M['m00'] != 0:
                    cx = int(M['m10']/M['m00'])
                    cy = int(M['m01']/M['m00'])
                    centre = (cx, cy)
                    x_b, y_b, w_b, h_b = cv2.boundingRect(c)
                    W_px_frame = w_b
                    Z = (focal_px * W_real) / float(W_px_frame)
                    metres_per_pixel = W_real / float(W_px_frame)
                    cv2.circle(frame, centre, int(w_b/2), (0,255,0), 2)
                    cv2.circle(frame, centre, 3, (0,0,255), -1)
        
        # Start a new line segment whenever the marker appears
        if centre is not None:
            if current_line is None:
                current_line = deque()
                lines.append(current_line)
            current_line.append(centre)
        else:
            # End the current line if marker disappears
            current_line = None
        
        # Draw each segment independently
        for segment in lines:
            pts = list(segment)
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (255,0,0), 2)
        
        # Compute speed/distance
        if centre is not None and prev_pt is not None and metres_per_pixel is not None:
            dx = centre[0] - prev_pt[0]
            dy = centre[1] - prev_pt[1]
            dist_m = math.hypot(dx, dy) * metres_per_pixel
            total_distance_m += dist_m
            speed_m_s = dist_m * fps
            angle = math.degrees(math.atan2(-dy, dx))
            
            cv2.putText(frame, f'Speed: {speed_m_s:.3f} m/s', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
            cv2.putText(frame, f'Total distance: {total_distance_m:.3f} m', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            cv2.putText(frame, f'Angle: {angle:.1f} deg', (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
            if Z is not None:
                cv2.putText(frame, f'Distance to camera: {Z:.3f} m', (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        else:
            cv2.putText(frame, 'Detecting marker...', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        
        prev_pt = centre
        out.write(frame)
    
    cap.release()
    out.release()
    print('Finished. Output saved to output_yellow_calibrated.mp4')

if __name__ == '__main__':
    main()
