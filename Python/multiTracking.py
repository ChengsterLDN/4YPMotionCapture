import cv2
import numpy as np
import math

capture = cv2.VideoCapture('FaresCropped_colours_preserved.mp4')

ret, frame = capture.read()
if not ret:
    print("Error")
    exit()

# Ask user how many objects to track
num_objects = int(input("Enter the number of objects to track: "))

# Lists to store color ranges for each object
lower_colors = []
upper_colors = []
object_names = []

# Multiple colour sampling for each object
for i in range(num_objects):
    print(f"\nSelect color for Object {i+1}")
    roi = cv2.selectROI(f'Select region for Object {i+1}', frame, False)
    
    if roi == (0, 0, 0, 0):
        print(f"No region selected for Object {i+1}. Using default color range.")
        # Assign different default colors for each object
        default_hues = [20, 100, 160, 60, 120]  # Different hues for different objects
        hue = default_hues[i % len(default_hues)]
        lower_color = np.array([hue - 10, 100, 100])
        upper_color = np.array([hue + 10, 255, 255])
    else:
        # Extract ROI
        x, y, w, h = [int(i) for i in roi]
        colour_sample = frame[y:y+h, x:x+w]

        sample_hsv = cv2.cvtColor(colour_sample, cv2.COLOR_BGR2HSV)

        # Calculate min, max and standard deviation of ROI colour
        h_low, h_high, h_std = np.min(sample_hsv[:,:,0]), np.max(sample_hsv[:,:,0]), np.std(sample_hsv[:,:,0])
        s_low, s_high, s_std = np.min(sample_hsv[:,:,1]), np.max(sample_hsv[:,:,1]), np.std(sample_hsv[:,:,1])
        v_low, v_high, v_std = np.min(sample_hsv[:,:,2]), np.max(sample_hsv[:,:,2]), np.std(sample_hsv[:,:,2])

        # Define colour range based on min/max ± 2*std
        h_range = 2 * h_std
        s_range = 2 * s_std
        v_range = 2 * v_std
        
        lower_color = np.array([max(0, h_low - h_range), 
                              max(0, s_low - s_range), 
                              max(0, v_low - v_range)])
        upper_color = np.array([min(255, h_high + h_range), 
                              min(255, s_high + s_range), 
                              min(255, v_high + v_range)])
        
        print(f"Object {i+1} color range:")
        print(f"H: {lower_color[0]:.1f} - {upper_color[0]:.1f}")
        print(f"S: {lower_color[1]:.1f} - {upper_color[1]:.1f}")
        print(f"V: {lower_color[2]:.1f} - {upper_color[2]:.1f}")

    lower_colors.append(lower_color)
    upper_colors.append(upper_color)
    object_names.append(f"Object {i+1}")
    
    cv2.destroyWindow(f'Select region for Object {i+1}')

# Reset video to beginning
capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Determine initial distances between consecutive objects
ret, cal_frame = capture.read()
if not ret:
    print("Error reading calibration frame")
    exit()

# Process first frame to find all objects for calibration
cal_blurred = cv2.GaussianBlur(cal_frame, (5, 5), 0)
cal_hsv = cv2.cvtColor(cal_blurred, cv2.COLOR_BGR2HSV)

# Find all objects in calibration frame
cal_centres = []
for i in range(num_objects):
    cal_mask = cv2.inRange(cal_hsv, lower_colors[i], upper_colors[i])
    
    # Denoise calibration frame
    cal_kernel = np.ones((5, 5), np.uint8)
    cal_mask = cv2.morphologyEx(cal_mask, cv2.MORPH_CLOSE, cal_kernel, anchor=(-1, -1), iterations=20)
    cal_mask = cv2.morphologyEx(cal_mask, cv2.MORPH_OPEN, cal_kernel, anchor=(-1, -1), iterations=5)
    cal_contours, _ = cv2.findContours(cal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(cal_contours) > 0:
        # Find largest contour for this color
        largest_contour = max(cal_contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M['m00'] != 0:
            centre_x = int(M['m10'] / M['m00'])
            centre_y = int(M['m01'] / M['m00'])
            cal_centres.append((centre_x, centre_y, i))  # Store with object index
        else:
            cal_centres.append((0, 0, i))  # Placeholder if not found
    else:
        cal_centres.append((0, 0, i))  # Placeholder if not found

# Calculate initial distances between consecutive objects
initial_distances = []
if len(cal_centres) >= 2:
    for i in range(len(cal_centres) - 1):
        x1, y1, idx1 = cal_centres[i]
        x2, y2, idx2 = cal_centres[i + 1]
        distance_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        initial_distances.append(distance_px)
        
        print(f"Initial distance between {object_names[idx1]} and {object_names[idx2]}: {distance_px:.2f} pixels")

# Ask user for real-world distance between first two objects for calibration
if len(initial_distances) > 0:
    real_distance = float(input("Enter the real-world distance between the first two objects (metres): "))
    pixels_per_metre = initial_distances[0] / real_distance
    print(f"Calibration: {pixels_per_metre:.2f} pixels per metre")
    
    # Show calibration frame with distances
    for i in range(len(cal_centres) - 1):
        x1, y1, idx1 = cal_centres[i]
        x2, y2, idx2 = cal_centres[i + 1]
        cv2.line(cal_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        cv2.putText(cal_frame, f'{initial_distances[i]:.1f}px', 
                   (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    cv2.putText(cal_frame, f'Calibration: {initial_distances[0]:.1f} px = {real_distance:.3f} m', 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Calibration Frame', cal_frame)
    cv2.waitKey(1000)  # Show for 1 second
    cv2.destroyWindow('Calibration Frame')
else:
    print("Could not find enough objects for calibration")
    pixels_per_metre = None
    real_distance = None

# Reset capture to beginning for main loop
capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

# Trail will now store centers for all objects in each frame
trail = []

frame_count = 0
fps = capture.get(cv2.CAP_PROP_FPS) or 30

# Define colors for different objects
object_colors = [
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 0, 255),    # Red
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 255, 255),  # Yellow
    (128, 0, 128),  # Purple
    (255, 165, 0),  # Orange
]

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Convert to HSV 
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Track all objects
    centres = []
    
    for i in range(num_objects):
        # Create mask for each object's color range
        mask = cv2.inRange(hsv, lower_colors[i], upper_colors[i])
        
        # Denoise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, anchor=(-1, -1), iterations=20)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, anchor=(-1, -1), iterations=5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Find largest contour for this color
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M['m00'] != 0:
                centre_x = int(M['m10'] / M['m00'])
                centre_y = int(M['m01'] / M['m00'])
                centre = (centre_x, centre_y)
                centres.append(centre)
                
                # Draw circle and label for each object with unique color
                color = object_colors[i % len(object_colors)]
                cv2.circle(frame, center=centre, radius=5, color=color, thickness=2)
                cv2.putText(frame, object_names[i], (centre[0], centre[1] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                centres.append(None)
        else:
            centres.append(None)
    
    # Draw lines between consecutive objects (1-2, 2-3, etc.)
    valid_centres = [c for c in centres if c is not None]
    if len(valid_centres) >= 2:
        for i in range(len(valid_centres) - 1):
            if valid_centres[i] is not None and valid_centres[i + 1] is not None:
                cv2.line(frame, valid_centres[i], valid_centres[i + 1], (0, 255, 0), 2)
                
                # Display distance between consecutive objects
                x1, y1 = valid_centres[i]
                x2, y2 = valid_centres[i + 1]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                
                mid_x = (x1 + x2) // 2
                mid_y = (y1 + y2) // 2
                
                if pixels_per_metre is not None:
                    distance_m = distance / pixels_per_metre
                    cv2.putText(frame, f'{distance_m:.2f}m', 
                               (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                else:
                    cv2.putText(frame, f'{distance:.1f}px', 
                               (mid_x, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # Calculate and display velocity (using midpoint of all objects)
    if len(trail) > 0 and len(valid_centres) >= 2:
        # Calculate current midpoint
        current_mid_x = sum(c[0] for c in valid_centres) / len(valid_centres)
        current_mid_y = sum(c[1] for c in valid_centres) / len(valid_centres)
        
        # Calculate previous midpoint
        prev_valid_centres = [c for c in trail[-1] if c is not None]
        if len(prev_valid_centres) >= 2:
            prev_mid_x = sum(c[0] for c in prev_valid_centres) / len(prev_valid_centres)
            prev_mid_y = sum(c[1] for c in prev_valid_centres) / len(prev_valid_centres)
            
            displacement = math.sqrt((current_mid_x - prev_mid_x)**2 + (current_mid_y - prev_mid_y)**2)
            velocity_pixels_per_second = displacement * fps
            
            if pixels_per_metre is not None:
                velocity_m_per_second = velocity_pixels_per_second / pixels_per_metre
                cv2.putText(frame, f'Velocity: {velocity_m_per_second:.2f} m/s', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, f'Velocity: {velocity_pixels_per_second:.2f} px/s', 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Store centers for this frame
    trail.append(tuple(centres))
    
    # Display the original frame with tracking
    cv2.imshow('Multi-Object Tracking', frame)
    
    # Escape via keypress
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()

# Create culmination sweep (trace)
if ret:
    height, width = frame.shape[:2]
else:
    height, width = 1080, 1920

canvas = np.zeros((height, width, 3)) + 255  # White background
cv2.namedWindow('Trace', cv2.WINDOW_AUTOSIZE)

# Draw chronological trails for each object
for obj_idx in range(num_objects):
    color = object_colors[obj_idx % len(object_colors)]
    
    # Draw trail for this object across frames
    for i in range(len(trail) - 1):
        if (i < len(trail) and i + 1 < len(trail) and 
            obj_idx < len(trail[i]) and obj_idx < len(trail[i + 1]) and
            trail[i][obj_idx] is not None and trail[i + 1][obj_idx] is not None):
            
            x1, y1 = trail[i][obj_idx]
            x2, y2 = trail[i + 1][obj_idx]
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

# Draw connecting lines between consecutive objects within each frame
for frame_centres in trail:
    valid_centres = [c for c in frame_centres if c is not None]
    
    if len(valid_centres) >= 2:
        # Draw lines between consecutive objects (1-2, 2-3, etc.)
        for i in range(len(valid_centres) - 1):
            cv2.line(canvas, 
                    (int(valid_centres[i][0]), int(valid_centres[i][1])),
                    (int(valid_centres[i + 1][0]), int(valid_centres[i + 1][1])),
                    (0, 0, 0), 1)  # Black for connecting lines

cv2.imshow("Trace", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()