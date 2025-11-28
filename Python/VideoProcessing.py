import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox

class ColourPicker:
    def __init__(self):
        self.selected_colours = []
        self.current_frame = None
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Get dat colour
            color_bgr = self.current_frame[y, x]
            color_hsv = cv2.cvtColor(np.uint8([[color_bgr]]), cv2.COLOR_BGR2HSV)[0][0]
            
            self.selected_colours.append({
                'bgr': color_bgr.tolist(),
                'hsv': color_hsv.tolist()
            })
            
            # Draw a marker on the clicked position
            frame_with_marker = self.current_frame.copy()
            cv2.circle(frame_with_marker, (x, y), 10, (255, 255, 255), -1)  # White circle
            cv2.circle(frame_with_marker, (x, y), 8, (0, 0, 0), -1)        # Black ring
            cv2.circle(frame_with_marker, (x, y), 6, color_bgr.tolist(), -1)  # Colour center
            
            # Show count
            cv2.putText(frame_with_marker, f"Selected: {len(self.selected_colours)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow('Click on markers. Press SPACE when done', frame_with_marker)
            print(f"Selected colour {len(self.selected_colours)}: BGR {color_bgr}")

def select_video_file():
    """Open file dialog to select video file"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    file_types = [
        ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
        ("MP4 files", "*.mp4"),
        ("AVI files", "*.avi"),
        ("MOV files", "*.mov"),
        ("All files", "*.*")
    ]
    
    video_path = filedialog.askopenfilename(
        title="Select video file",
        filetypes=file_types
    )
    
    root.destroy()
    return video_path

def select_output_directory():
    """Open file dialog to select output directory"""
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    
    output_dir = filedialog.askdirectory(
        title="Select output directory for processed video"
    )
    
    root.destroy()
    return output_dir

def get_output_path(input_path):
    """
    Generate output path in the same directory as input file
    
    """
    directory = os.path.dirname(input_path)
    filename = os.path.basename(input_path)
    name, ext = os.path.splitext(filename)
    
    output_filename = f"{name}_colours_preserved.mp4"
    output_path = os.path.join(directory, output_filename)
    
    # If file already exists, add a number to avoid overwriting
    counter = 1
    while os.path.exists(output_path):
        output_filename = f"{name}_colours_preserved_{counter}.mp4"
        output_path = os.path.join(directory, output_filename)
        counter += 1
    
    return output_path


def select_marker_colours(video_path, frame_number=0):
    """Simple function to select marker colours from a video frame"""
    
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return None
    
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: Could not read frame from video!")
        return None
    
    picker = ColourPicker()
    picker.current_frame = frame
    
    cv2.imshow('Click on markers. Press SPACE when done', frame)
    cv2.setMouseCallback('Click on markers. Press SPACE when done', picker.mouse_callback)
    
    print("Click on each marker in the image.")
    print("Press SPACE when done, 'r' to reset, or 'q' to quit.")
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord(' ') or key == 13:  # SPACE or ENTER
            if picker.selected_colours:
                break
            else:
                print("Please select at least one marker colour!")
                
        elif key == ord('r'):  # Reset
            picker.selected_colours = []
            cv2.imshow('Click on markers. Press SPACE when done', frame)
            print("Selections reset!")
            
        elif key == ord('q'):  # Quit
            print("Quitting.")
            cv2.destroyAllWindows()
            return None
    
    cv2.destroyAllWindows()
    return picker.selected_colours

def process_video(input_path, output_path, selected_colours):
    """
    Process video to keep ONLY the selected colours, make everything else grayscale
    """
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create HSV ranges for the selected colours
    hsv_ranges = []
    for colour in selected_colours:
        h, s, v = colour['hsv']
        # Create a range around the selected HSV colour
        lower_hsv = [max(0, h-15), max(0, s-80), max(0, v-80)]
        upper_hsv = [min(179, h+15), min(255, s+80), min(255, v+80)]
        hsv_ranges.append((lower_hsv, upper_hsv))
    
    print(f"Processing video... Keeping {len(selected_colours)} colours visible.")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert to grayscale (this will be our background)
        gray_background = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_background = cv2.cvtColor(gray_background, cv2.COLOR_GRAY2BGR)
        
        # Convert to HSV for colour detection
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for ALL selected colours combined
        combined_mask = np.zeros((height, width), dtype=np.uint8)
        
        for lower, upper in hsv_ranges:
            mask = cv2.inRange(hsv_frame, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Clean up the mask (remove noise)
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        
        # where mask is white, keep original colours; elsewhere use grayscale
        result = np.where(combined_mask[:, :, np.newaxis] == 255, frame, gray_background)
        
        out.write(result.astype(np.uint8))
        frame_count += 1
        
        # progress
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    out.release()
    print(f"Done! Processed {frame_count} frames total.")
    print(f"Output saved to: {output_path}")

def main():
    # Select input video file
    print("=== SELECT INPUT VIDEO ===")
    video_path = select_video_file()
    
    if not video_path:
        print("No video file selected. Exiting.")
        return
    
    print(f"Selected video: {video_path}")
    
    # Select output location
    print("\n=== SELECT OUTPUT LOCATION ===")
    output_path = get_output_path(video_path)
    print(f"Output will be saved to: {output_path}")
    
    # Select frame
    frame_num = input("\nEnter frame number to pick colours from (default: 0): ").strip()
    frame_num = int(frame_num) if frame_num.isdigit() else 0
    
    # Select marker colours
    print("\n=== SELECT MARKER COLOURS ===")
    selected_colours = select_marker_colours(video_path, frame_num)
    
    if not selected_colours:
        print("No colours selected. Exciting.")
        return
    
    print(f"\nSelected {len(selected_colours)} marker colours:")
    for i, colour in enumerate(selected_colours, 1):
        print(f"  Colour {i}: BGR {colour['bgr']}")
    
    # Process the video
    print("\n=== PROCESSING VIDEO ===")
    process_video(
        video_path, 
        output_path, 
        selected_colours
    )
    
    print(f"\n=== COMPLETE ===")
    print(f"Input: {video_path}")
    print(f"Output: {output_path}")
    print(f"Only the {len(selected_colours)} selected marker colours are now visible in colour.")
    print("Everything else is grayscale.")

if __name__ == "__main__":
    main()