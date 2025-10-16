import cv2
import numpy as np
import math

capture = cv2.VideoCapture('stickTest.mp4')

ret, frame = capture.read()
if not ret:
    print("Error")
    exit()

# Colour picker
roi = cv2.selectROI('Select region for colour sampling', frame, False)
if roi == (0,0,0,0):
    print("No region was selected. Using yellow threshold.")
    lower_color = np.array([20, 100, 100])   
    upper_color = np.array([30, 255, 255])  

else:
    #Extract ROI
    x,y,w,h = [int(i) for i in roi]
    colour_sample = frame[y:y+h, x:x+w]

    sample_hsv = cv2.cvtColor(colour_sample, cv2.COLOR_BGR2HSV)

    # Calculate mean and standard deviation of ROI colour
    #h_mean, h_std = np.mean(sample_hsv[:,:,0]), np.std(sample_hsv[:,:,0])
    #s_mean, s_std = np.mean(sample_hsv[:,:,1]), np.std(sample_hsv[:,:,1])
    #v_mean, v_std = np.mean(sample_hsv[:,:,2]), np.std(sample_hsv[:,:,2])
    
    # Calculate high, low and standard deviation of ROI colour
    h_low, h_high, h_std = np.min(sample_hsv[:,:,0]),np.max(sample_hsv[:,:,0]), np.std(sample_hsv[:,:,0])
    s_low, s_high, s_std = np.min(sample_hsv[:,:,1]), np.max(sample_hsv[:,:,1]), np.std(sample_hsv[:,:,1])
    v_low, v_high, v_std = np.min(sample_hsv[:,:,2]), np.max(sample_hsv[:,:,2]), np.std(sample_hsv[:,:,2])

    # Define colourr range based on mean \pm 2*std
    h_range = 5 * h_std
    s_range = 5 * s_std
    v_range = 5 * v_std
    
    lower_colour = np.array([max(0, h_low - h_range), 
                           max(0, s_low - s_range), 
                           max(0, v_low - v_range)])
    upper_colour = np.array([min(255, h_high + h_range), 
                           min(255, s_high + s_range), 
                           min(255, v_high + v_range)])
    
    print(f"Selected colour range:")
    print(f"H: {lower_colour[0]:.1f} - {upper_colour[0]:.1f}")
    print(f"S: {lower_colour[1]:.1f} - {upper_colour[1]:.1f}")
    print(f"V: {lower_colour[2]:.1f} - {upper_colour[2]:.1f}")

cv2.destroyWindow('Select region for colour sampling')

# reset
capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

# determine initial distance
ret, cal_frame = capture.read()
if not ret:
    print("Error reading calibration frame")
    exit()

# Process first frame to find objects for calibration
cal_blurred = cv2.GaussianBlur(cal_frame, (5,5), 0)
cal_hsv = cv2.cvtColor(cal_blurred, cv2.COLOR_BGR2HSV)
cal_mask = cv2.inRange(cal_hsv, lower_colour, upper_colour)

# Denoise calibration frame
cal_kernel = np.ones((5, 5), np.uint8)
cal_mask = cv2.morphologyEx(cal_mask, cv2.MORPH_CLOSE, cal_kernel, anchor=(-1,-1), iterations=20)
cal_mask = cv2.morphologyEx(cal_mask, cv2.MORPH_OPEN, cal_kernel, anchor=(-1,-1), iterations=5)
cal_contours, _ = cv2.findContours(cal_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find objects in calibration frame
cal_centres = []
if len(cal_contours) >= 2:
    sorted_contours = sorted(cal_contours, key=cv2.contourArea, reverse=True)[:2]
    
    for cnt in sorted_contours:
        M = cv2.moments(cnt)
        if M['m00'] != 0:
            centre_x = int(M['m10'] / M['m00'])
            centre_y = int(M['m01'] / M['m00'])
            cal_centres.append((centre_x, centre_y))
    
    if len(cal_centres) == 2:
        # Calculate initial distance in pixels
        x1, y1 = cal_centres[0]
        x2, y2 = cal_centres[1]
        initial_distance_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        print(f"Initial distance between objects: {initial_distance_px:.2f} pixels")
        
        # Ask user for real-world distance
        real_distance = float(input("Enter the real-world distance between objects (metres): "))
        pixels_per_metre = initial_distance_px / real_distance
        print(f"Calibration: {pixels_per_metre:.2f} pixels per metre")
        
        # Show calibration frame with distance
        cv2.line(cal_frame, cal_centres[0], cal_centres[1], (0, 255, 0), 2)
        cv2.putText(cal_frame, f'Calibration: {initial_distance_px:.1f} px = {real_distance:.3f} m', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Calibration Frame', cal_frame)
        cv2.waitKey(1000)  # Show for 1 second
        cv2.destroyWindow('Calibration Frame')
    else:
        print("Could not find two objects for calibration")
        pixels_per_metre = None
        real_distance = None
else:
    print("Could not find enough objects for calibration")
    pixels_per_metre = None
    real_distance = None

# Reset capture to beginning for main loop
capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

trail = []

initial_distance = None
frame_count = 0
fps = capture.get(cv2.CAP_PROP_FPS) or 30

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Convert to HSV 
    blurred = cv2.GaussianBlur(frame, (5,5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Create mask for the specified colour range
    mask = cv2.inRange(hsv, lower_colour, upper_colour)
    
    # Denoise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, anchor = (-1,-1), iterations = 20)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, anchor = (-1,-1), iterations = 5)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #max_area = 0
    
    # Track detected objects
    num_objects = 2  
    centres = []

    if len(contours) > 0:
        # Sort contours by area and get the largest N number
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:num_objects]
        
        for i, cnt in enumerate(sorted_contours):
            # Calculate centre using moments
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                centre_x = int(M['m10'] / M['m00'])
                centre_y = int(M['m01'] / M['m00'])
                centre = (centre_x, centre_y)
                centres.append(centre)
                
                # Draw circle and label for each object
                cv2.circle(frame, center=centre, radius=5, color=(0, 255, 0), thickness=2)
                cv2.putText(frame, f'Object {i+1}', (centre[0], centre[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        # Calculate and display distance between objects
        if len(centres) == 2:
            x1, y1 = centres[0]
            x2, y2 = centres[1]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

             # Calculate velocity if we have a previous frame to compare with
            if len(trail) > 0 and len(trail[-1]) == 2:
                prev_x1, prev_y1 = trail[-1][0]
                prev_x2, prev_y2 = trail[-1][1]
                
                # Calculate displacement of the midpoint
                mid_x = (x1 + x2) / 2
                mid_y = (y1 + y2) / 2
                prev_mid_x = (prev_x1 + prev_x2) / 2
                prev_mid_y = (prev_y1 + prev_y2) / 2
                
                displacement = math.sqrt((mid_x - prev_mid_x)**2 + (mid_y - prev_mid_y)**2)
                velocity_pixels_per_second = displacement * fps
                
                if 'pixels_per_metre' is not None:
                    velocity_m_per_second = velocity_pixels_per_second / pixels_per_metre
                    # Display velocity in both pixels and metres
                    cv2.putText(frame, f'Velocity: {velocity_m_per_second:.2f} m/s', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    # Fallback to pixels only if calibration not done
                    cv2.putText(frame, f'Velocity: {velocity_pixels_per_second:.2f} px/s', 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Draw lines between all detected centres
        if len(centres) > 1:
            for i in range(len(centres)):
                for j in range(i+1, len(centres)):
                    cv2.line(frame, centres[i], centres[j], (0, 255, 0), 2)
        
        # Store ALL centre pairs for line tracing 
        if len(centres) > 1:
            trail.append(tuple(centres))

    # Display the original frame with tracking boxes
    cv2.imshow('Colour Tracking', frame)

    # Escape via keypress
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


capture.release()

# Line sweeping trail
if ret:
    height, width = frame.shape[:2]
else:
    height, width = 1080, 1920

canvas = np.zeros((height, width, 3)) + 255
cv2.namedWindow('Trace', cv2.WINDOW_AUTOSIZE)

# Colour list - do we need this?
trail_col = (0,0,255) #Red

# Draw lines between corresponding objects across frames
for i in range(len(trail) - 1):
    if len(trail[i]) == len(trail[i+1]):  # Only draw if same number of objects
        num_pairs = len(trail[i])
        
        # Draw lines between each pair of objects
        for pair_idx in range(num_pairs):
            if pair_idx < len(trail[i]) and pair_idx < len(trail[i+1]):
                x1, y1 = trail[i][pair_idx]
                x2, y2 = trail[i+1][pair_idx]
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), trail_col, 2)

# Draw connecting lines between objects within each frame
for frame_centres in trail:
    if len(frame_centres) > 1:
        # Draw lines between all objects in this frame
        for i in range(len(frame_centres)):
            for j in range(i+1, len(frame_centres)):
                cv2.line(canvas, 
                        (int(frame_centres[i][0]), int(frame_centres[i][1])),
                        (int(frame_centres[j][0]), int(frame_centres[j][1])),
                        trail_col, 1)

cv2.imshow("Trace", canvas)
cv2.waitKey(0)