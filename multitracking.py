import cv2
import numpy as np

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

trail = []
while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Convert to HSV color space
    blurred = cv2.GaussianBlur(frame, (5,5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Create mask for the specified color range
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
        # Sort contours by area and get the largest N contours
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
colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

# Draw lines between corresponding objects across frames
for i in range(len(trail) - 1):
    if len(trail[i]) == len(trail[i+1]):  # Only draw if same number of objects
        num_pairs = len(trail[i])
        
        # Draw lines between each pair of objects
        for pair_idx in range(num_pairs):
            if pair_idx < len(trail[i]) and pair_idx < len(trail[i+1]):
                x1, y1 = trail[i][pair_idx]
                x2, y2 = trail[i+1][pair_idx]
                colour = colours[pair_idx % len(colours)]
                cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), colour, 2)

# Draw connecting lines between objects within each frame
for frame_centres in trail:
    if len(frame_centres) > 1:
        # Draw lines between all objects in this frame
        for i in range(len(frame_centres)):
            for j in range(i+1, len(frame_centres)):
                colour = colours[(i + j) % len(colours)]
                cv2.line(canvas, 
                        (int(frame_centres[i][0]), int(frame_centres[i][1])),
                        (int(frame_centres[j][0]), int(frame_centres[j][1])),
                        colour, 1)

cv2.imshow("Trace", canvas)
cv2.waitKey(0)