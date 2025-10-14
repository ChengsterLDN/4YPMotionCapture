import cv2
import sys
import matplotlib.pyplot as plt

s = 0 # Default camera device index
if len(sys.argv) > 1: # Checks if any command line arguments are passed
    s = sys.argv[1]

source = cv2.VideoCapture(s)

# Create a window to show the camera preview
win_name = 'Camera Preview'
cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

# While loop to continuously stream to camera unless Escape key is hit
while cv2.waitKey(1) != 27:
    has_frame, frame = source.read() #Return a frame from the video screen, and returns a logical variable has_frame
    if not has_frame:
        break
    cv2.imshow(win_name,frame)
    plt.imshow(frame)

#source.release()
#cv2.destroyWindow(win_name)



# Create the video writer object
frame_width = int(source.get(3))
frame_height = int(source.get(4))

out_mp4 = cv2.VideoWriter('test.avi', cv2.VideoWriter_fourcc(*'XVID'),30,(frame_width, frame_height))


# Whilst the source is still opened
while(source.isOpened()):
    # Capture frame by frame
    has_frame, frame = source.read()

    if has_frame == True:
        out_mp4.write(frame)
    else:
        break

# When everything is done, release the VideoCapture and VideoWriter Objects
source.release()
out_mp4.release()

