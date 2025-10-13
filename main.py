import cv2
import numpy as np

capture = cv2.VideoCapture('sample.mp4')
ret, frame = capture.read()
trackBox = cv2.selectROI('Track', frame, False)

# Template image method

template = frame[int(trackBox[1]):int(trackBox[1]+trackBox[3]), int(trackBox[0]):int(trackBox[0]+trackBox[2])]

while True:
    ret, frame = capture.read()
    if not ret:
        break

    # Template matching
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # Bind box around tracked object
    top_left = max_loc
    bottom_right = (top_left[0] + int(trackBox[2]), top_left[1] + int(trackBox[3]))
    cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)

    # Display the tracked object
    cv2.imshow('Tracked Object', frame)

    # Escape via keypress
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break