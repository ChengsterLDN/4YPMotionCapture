import cv2
import threading
import time
from datetime import datetime

class WebcamRecorder:
    def __init__(self, camera_index1=0, camera_index2=1, output1="camera1_output.mp4", output2="camera2_output.mp4"):
        self.camera_index1 = camera_index1
        self.camera_index2 = camera_index2
        self.output1 = output1
        self.output2 = output2
        
        # Video writer objects
        self.writer1 = None
        self.writer2 = None
        
        # Recording flags
        self.recording = False
        self.frames1 = []
        self.frames2 = []
        
    def setup_cameras(self):
        """Initialise both cameras"""
        print("Initialising cameras...")
        
        # Open cameras
        self.cap1 = cv2.VideoCapture(self.camera_index1)
        self.cap2 = cv2.VideoCapture(self.camera_index2)
        
        # Set camera properties 
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap1.set(cv2.CAP_PROP_FPS, 30)
        
        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap2.set(cv2.CAP_PROP_FPS, 30)
        
        # Check if cameras opened successfully
        if not self.cap1.isOpened():
            raise Exception(f"Could not open camera 1 (index {self.camera_index1})")
        if not self.cap2.isOpened():
            raise Exception(f"Could not open camera 2 (index {self.camera_index2})")
            
        # Get frame properties for video writers
        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()
        
        if not ret1 or not ret2:
            raise Exception("Could not read frames from cameras")
            
        self.frame_width = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = 30.0
        
        # Setup video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer1 = cv2.VideoWriter(self.output1, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.writer2 = cv2.VideoWriter(self.output2, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        print(f"Cameras initialised successfully")
        print(f"Resolution: {self.frame_width}x{self.frame_height}")
        print(f"FPS: {self.fps}")
        
    def start_recording(self):
        """Start recording from both cameras"""
        self.recording = True
        print("Starting recording...")
        
        # Create threads for each camera
        thread1 = threading.Thread(target=self.record_camera1)
        thread2 = threading.Thread(target=self.record_camera2)
        
        thread1.daemon = True
        thread2.daemon = True
        
        thread1.start()
        thread2.start()
        
        return thread1, thread2
        
    def record_camera1(self):
        """Record from camera 1"""
        while self.recording:
            ret, frame = self.cap1.read()
            if ret:
                self.writer1.write(frame)
            else:
                print("Warning: Could not read frame from camera 1")
                
    def record_camera2(self):
        """Record from camera 2"""
        while self.recording:
            ret, frame = self.cap2.read()
            if ret:
                self.writer2.write(frame)
            else:
                print("Warning: Could not read frame from camera 2")
                
    def stop_recording(self):
        """Stop recording and release resources"""
        print("Stopping recording...")
        self.recording = False
        time.sleep(0.1)  # Allow threads to finish
        
        # Release everything
        if self.writer1:
            self.writer1.release()
        if self.writer2:
            self.writer2.release()
        if self.cap1:
            self.cap1.release()
        if self.cap2:
            self.cap2.release()
            
        cv2.destroyAllWindows()
        print(f"Recording saved as '{self.output1}' and '{self.output2}'")

def main():
    # Configuration
    CAMERA_INDEX_1 = 0  # Change to your first camera index
    CAMERA_INDEX_2 = 1  # Change to your second camera index
    
    # Generate output filenames with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FILE_1 = f"camera1_{timestamp}.mp4"
    OUTPUT_FILE_2 = f"camera2_{timestamp}.mp4"
    
    recorder = WebcamRecorder(
        camera_index1=CAMERA_INDEX_1,
        camera_index2=CAMERA_INDEX_2,
        output1=OUTPUT_FILE_1,
        output2=OUTPUT_FILE_2
    )
    
    try:
        # Setup cameras
        recorder.setup_cameras()
        
        # Start recording
        threads = recorder.start_recording()
        
        print("Recording in progress...")
        print("Press 'q' to stop recording")
        
        # Display preview (optional)
        while recorder.recording:
            # You can add preview display here if desired
            # For minimal resource usage, we'll just check for quit command
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        recorder.stop_recording()

if __name__ == "__main__":
    main()