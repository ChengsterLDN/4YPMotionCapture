# 4YP Motion Tracking with Two-Tone
# J. Cheng

import cv2
import numpy as np
from scipy.spatial.distance import cdist

# Makes sure the vid doesn't zoom into like 5 pxls of the bloody video
def resize_frame(frame, scale_percent=50):
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

# Region of interest i.e. the two-tone marker
def roi_colour(frame):
    red_lower1 = np.array([0, 125, 60])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([170, 125, 60])
    red_upper2 = np.array([180, 255, 255])
    green_lower = np.array([40, 40, 40])
    green_upper = np.array([85, 255, 255])

    # perhaps it will be better for this to be manually selected - colour picker?
    return {
        'red': (red_lower1, red_upper1, red_lower2, red_upper2),
        'green': (green_lower, green_upper)
    }

def get_colour_masks(frame, colour_range):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_lower, green_upper = colour_range['green']
    green_mask = cv2.inRange(hsv, green_lower, green_upper)

    red_lower1, red_upper1, red_lower2, red_upper2 = colour_range['red']
    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    if red_lower2 is not None and red_upper2 is not None:
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    else:
        red_mask = red_mask1

    return red_mask, green_mask

def detect_markers(frame, colour_range,
                   min_blob_area=1000,
                   max_pair_dist_ratio=1.5,
                   min_combined_area=5000):

    raw_red, raw_green = get_colour_masks(frame, colour_range)

    kernel = np.ones((5, 5), np.uint8)

    # TUNE!!!! 
    def clean_mask(mask):
        m = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel, iterations=2)
        return m

    clean_red = clean_mask(raw_red)
    clean_green = clean_mask(raw_green)

    def get_blobs(mask):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_blob_area:
                continue
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            (_, _), radius = cv2.minEnclosingCircle(cnt)
            blobs.append({"centre": (cx, cy), "radius": radius, "area": area, "contour": cnt})
        return blobs

    red_blobs = get_blobs(clean_red)
    green_blobs = get_blobs(clean_green)

    markers = []
    used_green = set()

    for rb in red_blobs:
        best_dist = None
        best_gb_idx = None

        for gi, gb in enumerate(green_blobs):
            if gi in used_green:
                continue
            dist = np.hypot(rb["centre"][0] - gb["centre"][0],
                            rb["centre"][1] - gb["centre"][1])
            avg_radius = (rb["radius"] + gb["radius"]) / 2
            if dist < avg_radius * max_pair_dist_ratio:
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_gb_idx = gi

        if best_gb_idx is None:
            continue  # Isolated red — skip

        gb = green_blobs[best_gb_idx]
        used_green.add(best_gb_idx)

        # Build combined mask for this pair
        pair_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.drawContours(pair_mask, [rb["contour"]], -1, 255, -1)
        cv2.drawContours(pair_mask, [gb["contour"]], -1, 255, -1)
        pair_mask = cv2.morphologyEx(pair_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        combined_area = cv2.countNonZero(pair_mask)
        if combined_area < min_combined_area:
            continue

        contours2, _ = cv2.findContours(pair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours2:
            continue
        cnt2 = max(contours2, key=cv2.contourArea)

        (cx, cy), radius = cv2.minEnclosingCircle(cnt2)
        centre = (int(cx), int(cy))

        # Orientation; basically the angle from marker centre towards green centroid
        #CORRECTION - do we need this?
        gx, gy = gb["centre"]
        orientation = np.arctan2(gy - cy, gx - cx)

        markers.append({
            "centre": centre,
            "radius": int(radius),
            "orientation": orientation,
            "area": combined_area
        })

    return markers, raw_red, raw_green, clean_red, clean_green

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


# MAIN 

# -----------------------------------------------------------------------------------
video_path = "Sample2ToneMk2.mp4"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
if not ret:
    raise RuntimeError("Could not read video. I.e. make sure the vid file is in the same directory as this python script OR just paste in the whole file path")

colour_range = roi_colour(frame)

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

h, w = frame.shape[:2]
sweep = np.zeros((h, w, 3), dtype=np.uint8)

prev_markers = []

# DEBUGGING WINDOWS
cv2.namedWindow("Tracking", cv2.WINDOW_NORMAL)
cv2.namedWindow("Sweep", cv2.WINDOW_NORMAL)
cv2.namedWindow("Red Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Green Mask", cv2.WINDOW_NORMAL)
cv2.namedWindow("Combined Mask", cv2.WINDOW_NORMAL)

cv2.resizeWindow("Tracking", 640, 480)
cv2.resizeWindow("Sweep", 640, 480)
cv2.resizeWindow("Red Mask", 320, 240)
cv2.resizeWindow("Green Mask", 320, 240)
cv2.resizeWindow("Combined Mask", 320, 240)

while ret:
    markers, raw_red, raw_green, filtered_red, filtered_green = detect_markers(frame, colour_range)

    # Build combined mask for display
    combined_mask = cv2.bitwise_or(filtered_red, filtered_green)

    # Coloured mask displays
    red_display = cv2.cvtColor(filtered_red, cv2.COLOR_GRAY2BGR)
    green_display = cv2.cvtColor(filtered_green, cv2.COLOR_GRAY2BGR)
    combined_display = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)

    red_display[:, :, 2] = cv2.add(red_display[:, :, 2], filtered_red // 2)
    red_display[:, :, 0] = cv2.subtract(red_display[:, :, 0], filtered_red // 2)
    red_display[:, :, 1] = cv2.subtract(red_display[:, :, 1], filtered_red // 2)

    green_display[:, :, 1] = cv2.add(green_display[:, :, 1], filtered_green // 2)
    green_display[:, :, 0] = cv2.subtract(green_display[:, :, 0], filtered_green // 2)
    green_display[:, :, 2] = cv2.subtract(green_display[:, :, 2], filtered_green // 2)

    combined_display[:, :, 0] = cv2.add(combined_display[:, :, 0], combined_mask // 3)
    combined_display[:, :, 1] = cv2.add(combined_display[:, :, 1], combined_mask // 3)
    combined_display[:, :, 2] = cv2.subtract(combined_display[:, :, 2], combined_mask // 3)

    for m in markers:
        # Draw marker circle and centre
        cv2.circle(frame, m["centre"], m["radius"], (255, 255, 255), 2)
        cv2.circle(frame, m["centre"], 3, (0, 0, 0), -1)

        # Draw orientation arrow - but this is hella buggy ngl
        if m["orientation"] is not None:
            dx = int(40 * np.cos(m["orientation"]))
            dy = int(40 * np.sin(m["orientation"]))
            cv2.line(frame, m["centre"],
                     (m["centre"][0] + dx, m["centre"][1] + dy),
                     (255, 0, 0), 2)

        cx, cy = m["centre"]
        cv2.putText(frame, f"({cx}, {cy})", (cx + 10, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Sweep trail
    pairs = associate_markers(prev_markers, markers)
    for m_prev, m_curr in pairs:
        cv2.line(sweep, m_prev["centre"], m_curr["centre"], (255, 255, 255), 1)

    # DISPLAY
    display = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("Tracking", display)

    sweep_fix = cv2.resize(sweep, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow("Sweep", sweep_fix)

    cv2.imshow("Red Mask", cv2.resize(red_display, (320, 240)))
    cv2.imshow("Green Mask", cv2.resize(green_display, (320, 240)))
    cv2.imshow("Combined Mask", cv2.resize(combined_display, (320, 240)))

    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC to quit
        break
    elif key == ord('p'):  # P to pause
        print("Paused. Press any key to continue...")
        cv2.waitKey(0)

    prev_markers = markers
    ret, frame = cap.read()

cap.release()
cv2.destroyAllWindows()