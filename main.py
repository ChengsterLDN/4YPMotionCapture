import cv2
import numpy as np

capture = cv2.VideoCapture('sample1.mp4')

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
    h_mean, h_std = np.mean(sample_hsv[:,:,0]), np.std(sample_hsv[:,:,0])
    s_mean, s_std = np.mean(sample_hsv[:,:,1]), np.std(sample_hsv[:,:,1])
    v_mean, v_std = np.mean(sample_hsv[:,:,2]), np.std(sample_hsv[:,:,2])
    
    # Define colourr range based on mean \pm 2*std
    h_range = 1.5 * h_std
    s_range = 1.5 * s_std
    v_range = 1.5 * v_std
    
    lower_color = np.array([max(0, h_mean - h_range), 
                           max(0, s_mean - s_range), 
                           max(0, v_mean - v_range)])
    upper_color = np.array([min(179, h_mean + h_range), 
                           min(255, s_mean + s_range), 
                           min(255, v_mean + v_range)])
    
    print(f"Selected color range:")
    print(f"H: {lower_color[0]:.1f} - {upper_color[0]:.1f}")
    print(f"S: {lower_color[1]:.1f} - {upper_color[1]:.1f}")
    print(f"V: {lower_color[2]:.1f} - {upper_color[2]:.1f}")

cv2.destroyWindow('Select region for colour sampling')

# reset
capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for the specified color range
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # Denoise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw bounding boxes around detected objects
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, 'Object', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the original frame with tracking boxes
    cv2.imshow('Color Tracking', frame)

    # Escape via keypress
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()