import cv2
import numpy as np

capture = cv2.VideoCapture('sample3.mp4')

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
    #frame = cv2.convertScaleAbs(frame, alpha=1.3, beta=20)
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
    if len(contours) > 0:
        # find the largest contour
        cnt = sorted(contours, key = cv2.contourArea, reverse = True)[0]
        
        # Get the radius of the enclosing circle around the found contour
        ((x, y), radius) = cv2.minEnclosingCircle(cnt)
        
        # Calculate centre
        M = cv2.moments(cnt)
        centre = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        cv2.circle(frame, center=centre, radius=5, color=(0, 255, 0), thickness=2)
        cv2.putText(frame, 'Object', (centre[0], centre[1] - 2), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        trail.append(tuple([x,y]))

# Old Method
    """for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:  
            max_area = area
            x, y, w, h = cv2.boundingRect(contour)
            M = cv2.moments(cnt)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            trail.append(tuple([x,y]))
            xc = int(x + 0.5 * w)
            yc = int(y + 0.5 * h)

            cv2.circle(frame, center=(xc, yc), radius=5, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, 'Object', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            continue
    """
    # Display the original frame with tracking boxes
    cv2.imshow('Colour Tracking', mask)

    # Escape via keypress
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
capture.release()
"""
# Traced Outline
if ret:
    height, width = frame.shape[:2]
else:
    height, width = 1080, 1920

canvas = np.zeros((height, width, 3)) + 255
cv2.namedWindow('Trace', cv2.WINDOW_AUTOSIZE)
for i in range(len(trail) - 1):
    trail_x1, trail_y1 = trail[i]
    trail_x2, trail_y2 = trail[i+1]
    cv2.line(canvas, (int(trail_x1), int(trail_y1)), (int(trail_x2), int(trail_y2)), (255,0,0), 2)
cv2.imshow("Trace", canvas)"""
cv2.waitKey(0) 