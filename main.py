import cv2
import numpy as np

capture = cv2.VideoCapture('sample3.mp4')

ret, frame = capture.read()
if not ret:
    print("Error")
    exit()

# Colour picker 1
roi1 = cv2.selectROI('Select region for colour sampling', frame, False)
if roi1 == (0,0,0,0):
    print("No region was selected. Using yellow threshold.")
    lower_color = np.array([20, 100, 100])   
    upper_color = np.array([30, 255, 255])  

else:
    #Extract ROI
    x,y,w,h = [int(i) for i in roi1]
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
    h_range = 1.5 * h_std
    s_range = 2.5 * s_std
    v_range = 1.5 * v_std
    
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

cv2.destroyWindow('Select region 1 for colour sampling')

# reset
capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Create mask for the specified color range
    mask = cv2.inRange(hsv, lower_colour, upper_colour)
    
    # Denoise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    # Track detected objects
    for contour in contours:
        area = cv2.contourArea(contour)
        print(area)
        if area > max_area:  
            max_area = area
            x, y, w, h = cv2.boundingRect(contour)
            #cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            xc = int(x + 0.5 * w)
            yc = int(y + 0.5 * h)

            cv2.circle(frame, center=(xc, yc), radius=5, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, 'Object', (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            continue

    # Display the original frame with tracking boxes
    cv2.imshow('Colour Tracking', mask)

    # Escape via keypress
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

capture.release()
cv2.destroyAllWindows()
