import cv2
import numpy as np
from scipy.spatial.distance import cdist

def resize_frame(frame, scale_percent=50):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

def roi_colour(frame):
    # User specifies marker colour ranges
    
    # Resize frame for ROI selection
    frame_resized = resize_frame(frame, scale_percent=50)
    
    print("Select ROI for RED color (SPACE or ENTER to confirm, ESC to skip)")
    red_roi = cv2.selectROI('Select RED region', frame_resized, False)
    cv2.destroyWindow('Select RED region')
    
    print("Select ROI for GREEN color (SPACE or ENTER to confirm, ESC to skip)")
    green_roi = cv2.selectROI('Select GREEN region', frame_resized, False)
    cv2.destroyWindow('Select GREEN region')
    
    # Scale ROI back to original frame coordinates
    scale_factor = 100 / 50

    # RED ROI - two ranges as red hue-value goes beyond 0 or 360 (opencv is actually 0 or 179)
    red_lower1, red_upper1, red_lower2, red_upper2 = None, None, None, None
    
    if red_roi != (0, 0, 0, 0):
        x, y, w, h = [int(v * scale_factor) for v in red_roi]
        red_sample = frame[y:y+h, x:x+w]
        
        if red_sample.size > 0:
            red_hsv = cv2.cvtColor(red_sample, cv2.COLOR_BGR2HSV)
            
            # Calculate min, max and standard deviation
            h_min, h_max, h_std = np.min(red_hsv[:,:,0]), np.max(red_hsv[:,:,0]), np.std(red_hsv[:,:,0])
            s_min, s_max, s_std = np.min(red_hsv[:,:,1]), np.max(red_hsv[:,:,1]), np.std(red_hsv[:,:,1])
            v_min, v_max, v_std = np.min(red_hsv[:,:,2]), np.max(red_hsv[:,:,2]), np.std(red_hsv[:,:,2])
            
            # Define ranges (handle red's circular nature in HSV)
            h_range = 2 * h_std
            
            # Red typically has hue values near 0 or near 180
            if h_max - h_min < 90:  # Not crossing the circular boundary
                red_lower1 = np.array([max(0, h_min - h_range), 
                                      max(50, s_min - 2*s_std), 
                                      max(50, v_min - 2*v_std)])
                red_upper1 = np.array([min(179, h_max + h_range), 
                                      255, 255])
                red_lower2, red_upper2 = None, None
            else:
                # Handle case where red wraps around 0/180
                red_lower1 = np.array([0, max(50, s_min - 2*s_std), max(50, v_min - 2*v_std)])
                red_upper1 = np.array([min(10, h_max + h_range), 255, 255])
                red_lower2 = np.array([max(170, h_min - h_range), max(50, s_min - 2*s_std), max(50, v_min - 2*v_std)])
                red_upper2 = np.array([179, 255, 255])

    green_lower, green_upper = None, None
    
    if green_roi != (0, 0, 0, 0):
        x, y, w, h = [int(v * scale_factor) for v in green_roi]
        green_sample = frame[y:y+h, x:x+w]
        
        if green_sample.size > 0:
            green_hsv = cv2.cvtColor(green_sample, cv2.COLOR_BGR2HSV)
            
            # Calculate min, max and standard deviation
            h_min, h_max, h_std = np.min(green_hsv[:,:,0]), np.max(green_hsv[:,:,0]), np.std(green_hsv[:,:,0])
            s_min, s_max, s_std = np.min(green_hsv[:,:,1]), np.max(green_hsv[:,:,1]), np.std(green_hsv[:,:,1])
            v_min, v_max, v_std = np.min(green_hsv[:,:,2]), np.max(green_hsv[:,:,2]), np.std(green_hsv[:,:,2])
            
            # Define ranges
            h_range = 2 * h_std
            green_lower = np.array([max(0, h_min - h_range), 
                                   max(50, s_min - 2*s_std), 
                                   max(50, v_min - 2*v_std)])
            green_upper = np.array([min(179, h_max + h_range), 
                                   255, 255])
    
    # Set defaults if no ROI selected
    if red_lower1 is None:
        red_lower1 = np.array([0, 70, 60])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 70, 60])
        red_upper2 = np.array([180, 255, 255])
    
    if green_lower is None:
        green_lower = np.array([40, 60, 60])
        green_upper = np.array([85, 255, 255])
    
    return {
        'red': (red_lower1, red_upper1, red_lower2, red_upper2),
        'green': (green_lower, green_upper)
    }

# Colour Segmentation - isolating the two tones (red and green)

def get_colour_masks(frame, colour_range):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green
    green_lower, green_upper = colour_range['green']
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    # Red (two HSV ranges)
    red_lower1, red_upper1, red_lower2, red_upper2 = colour_range['red']
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)

    if red_lower2 is not None and red_upper2 is not None:
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    else:
        red_mask = red_mask1

    return red_mask, green_mask

# Marker Detection - quadrant two-tone marker

def detect_markers(frame, colour_range):
    red_mask, green_mask = get_colour_masks(frame, colour_range)

    # Combine both tones' masks to get  full marker
    marker_mask = cv2.bitwise_or(red_mask, green_mask)

    # Denoising via morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_CLOSE, kernel)
    marker_mask = cv2.morphologyEx(marker_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(
        marker_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    markers = []

    if len(contours) > 0:
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        marker_contours = contours[:2]

        for cnt in marker_contours:
            #area = cv2.contourArea(cnt)
            #if area < 800:
            #    continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            centre = (int(x), int(y))
            radius = int(radius)

            # Orientation estimation (experimental and do we need this?)
            mask = np.zeros(marker_mask.shape, dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)

            green_only = cv2.bitwise_and(green_mask, green_mask, mask=mask)
            M = cv2.moments(green_only)

            orientation = None
            if M["m00"] > 0:
                gx = M["m10"] / M["m00"]
                gy = M["m01"] / M["m00"]
                orientation = np.arctan2(gy - y, gx - x)

            markers.append({
                "centre": centre,
                "radius": radius,
                "orientation": orientation
            })

    return markers, red_mask, green_mask, marker_mask

# Marker association

def associate_markers(prev_markers, cur_markers, max_dist=50):
    if not prev_markers or not cur_markers:
        return []

    prev_pts = np.array([m["centre"] for m in prev_markers])
    curr_pts = np.array([m["centre"] for m in cur_markers])

    D = cdist(prev_pts, curr_pts)
    pairs = []

    for i in range(len(prev_pts)):
        j = np.argmin(D[i])
        if D[i, j] < max_dist:
            pairs.append((prev_markers[i], cur_markers[j]))

    return pairs


# Main

video_path = "Sample2ToneMk1.mp4"   
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read video")

colour_range = roi_colour(frame)

# Reset
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

h, w = frame.shape[:2]
sweep = np.zeros((h, w, 3), dtype=np.uint8)

prev_markers = []

# DEBUG
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.namedWindow("Sweep", cv2.WINDOW_NORMAL)
cv2.namedWindow("Red Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Green Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Combined Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("HSV Preview", cv2.WINDOW_NORMAL)  

cv2.resizeWindow("Tracking", 640, 480)
cv2.resizeWindow("Sweep", 640, 480)
cv2.resizeWindow("Red Mask", 320, 240)
cv2.resizeWindow("Green Mask", 320, 240)
cv2.resizeWindow("Combined Mask", 320, 240)
cv2.resizeWindow("HSV Preview", 320, 240)

while ret:
    markers, red_mask, green_mask, combined_mask = detect_markers(frame, colour_range)

    # Convert masks to 3-channel for colored display
    red_display = cv2.cvtColor(red_mask, cv2.COLOR_GRAY2BGR)
    green_display = cv2.cvtColor(green_mask, cv2.COLOR_GRAY2BGR)
    combined_display = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    
    # Colorize the masks for better visualization
    # Make red mask actually red
    red_display[:, :, 2] = cv2.add(red_display[:, :, 2], red_mask // 2)
    red_display[:, :, 0] = cv2.subtract(red_display[:, :, 0], red_mask // 2)
    red_display[:, :, 1] = cv2.subtract(red_display[:, :, 1], red_mask // 2)
    
    # Make green mask actually green
    green_display[:, :, 1] = cv2.add(green_display[:, :, 1], green_mask // 2)
    green_display[:, :, 0] = cv2.subtract(green_display[:, :, 0], green_mask // 2)
    green_display[:, :, 2] = cv2.subtract(green_display[:, :, 2], green_mask // 2)
    
    # Make combined mask cyan (red + green)
    combined_display[:, :, 0] = cv2.add(combined_display[:, :, 0], combined_mask // 3)
    combined_display[:, :, 1] = cv2.add(combined_display[:, :, 1], combined_mask // 3)
    combined_display[:, :, 2] = cv2.subtract(combined_display[:, :, 2], combined_mask // 3)

    for idx, m in enumerate(markers):


        cv2.circle(frame, m["centre"], m["radius"], (255, 255, 255), 2)
        cv2.circle(frame, m["centre"], 3, (0, 0, 0), -1)

        if m["orientation"] is not None:
            dx = int(40 * np.cos(m["orientation"]))
            dy = int(40 * np.sin(m["orientation"]))
            cv2.line(
                frame,
                m["centre"],
                (m["centre"][0] + dx, m["centre"][1] + dy),
                (255, 0, 0),
                2
            )

    # Associate markers between frames
    pairs = associate_markers(prev_markers, markers)

    
    for m_prev, m_curr in pairs:
        cv2.line(sweep, m_prev["centre"], m_curr["centre"], (255, 255, 255), 1)

    # Resizable Windows

    # cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("Sweep", cv2.WINDOW_NORMAL)

    # Create HSV preview (optional)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv_preview = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    display = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("Tracking", display)
    sweepFix = cv2.resize(sweep, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("Sweep", sweepFix)

    # cv2.imshow("Tracking", frame)
    #cv2.imshow("Sweep", sweep)

    cv2.imshow("Red Mask", cv2.resize(red_display, (320, 240)))
    cv2.imshow("Green Mask", cv2.resize(green_display, (320, 240)))
    cv2.imshow("Combined Mask", cv2.resize(combined_display, (320, 240)))
    cv2.imshow("HSV Preview", cv2.resize(hsv_preview, (320, 240)))


    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('p'):  # Pause
        print("Paused. Press any key to continue...")
        cv2.waitKey(0)

    prev_markers = markers
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
