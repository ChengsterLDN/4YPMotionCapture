import cv2
import numpy as np
import pickle
import json

class StereoCamera:
    def __init__(self, intrinsic_json):

        self.load_intrinsic(intrinsic_json)
        self.R = None # Rotation matrix
        self.F = None # Fundamental matrix
        self.T = None # Translation Vector
        self.E = None # Essential Matrix - camera matrix * rotation matrix
        self.rectify1 = None
        self.rectify2 = None
        self.proj1 = None
        self.proj2 = None
        self.Q = None

        self.lower_colour = None
        self.upper_colour = None

    def load_intrinsic(self, json_file):
        with open(json_file, 'r') as f:
            calibration = json.load(f)

        self.K = np.array(calibration[0]["intrinsic_matrix"]) # Camera Matrrix - could be ['camera_matrix' - see JSON or pickle file]
        self.D = np.array(calibration[1]["distortion_coef"]) # Distortion Vector

        print(self.K)
        print(self.D)
        print("Intrinsic parameters loaded from .JSON file")

    def stereo_calibration(self, frame1, frame2):
        grey1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        grey2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        #SIFT
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(frame1,None)
        kp2, des2 = sift.detectAndCompute(frame2,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)

        flann = cv2.FlannBasedMatcher(index_params,search_params)
        matches = flann.knnMatch(des1,des2,k=2)

        pts1 = []
        pts2 = []

        # ratio test as per Lowe's paper
        for i,(m,n) in enumerate(matches):
            if m.distance < 0.8*n.distance:
                pts2.append(kp2[m.trainIdx].pt)
                pts1.append(kp1[m.queryIdx].pt)

        pts1 = np.int32(pts1)
        pts2 = np.int32(pts2)

        # Compute essential
        E, mask = cv2.findEssentialMat(pts1,pts2, self.K)
        _, R, T, mask = cv2.recoverPose(E, pts1, pts2, self.K)

        self.R = R
        self.T = T
        self.E = E

        # Compute fundamental
        self.F = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)[0]

        # We select only inlier points
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]

        self.compute(grey1.shape[::-1])

        # Visualisation
        self.visualise(grey1, grey2, pts1, pts2, mask)

        return True

    def compute(self,image_size):
        R1, R2, P1, P2, self.Q, roi1, roi2 = cv2.stereoRectify(self.K, self.D, self.K, self.D, image_size, self.R, self.T)

        self.rectify1 = cv2.initUndistortRectifyMap(self.K, self.D, R1, P1, image_size, cv2.CV_16SC2)
        self.rectify2 = cv2.initUndistortRectifyMap(self.K, self.D, R2, P2, image_size, cv2.CV_16SC2)

        self.proj1 = P1
        self.proj2 = P2
    
    def visualise(self, frame1, frame2, pts1, pts2, mask):

        # Convert to colour BGR
        colour1 = cv2.cvtColor(frame1, cv2.COLOR_GRAY2BGR)
        colour2 = cv2.cvtColor(frame2, cv2.COLOR_GRAY2BGR)

        # Identify feature matches

        matches = np.hstack([colour1, colour2])
        h, w = frame1.shape

        for i in range(min(50,len(pts1))):
            if mask is not None and len(mask) > i and mask[i] > 0:
                pt1 = (int(pts1[i][0]), int(pts1[i][1]))
                pt2 = (int(pts2[i][0]) + w, int(pts2[i][1]))
                color = tuple(np.random.randint(0, 255, 3).tolist())
                cv2.line(matches, pt1, pt2, color, 1)
                cv2.circle(matches, pt1, 3, color, -1)
                cv2.circle(matches, pt2, 3, color, -1)
        
        cv2.putText(matches, f'SIFT Matches: {len(pts1)}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Feature Matches for Calibration', matches)
        cv2.waitKey(2000)
        cv2.destroyWindow('Feature Matches for Calibration')

    
    def detect(self, frame, lower_colour, upper_colour):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Ensure color ranges are proper numpy arrays with correct dtype
        self.lower_colour = np.array(lower_colour, dtype=np.uint8)
        self.upper_colour = np.array(upper_colour, dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_colour, upper_colour)

        # Denoise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, anchor = (-1,-1), iterations = 20)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, anchor = (-1,-1), iterations = 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # find the largest contour
            cnt = sorted(contours, key = cv2.contourArea, reverse = True)[0]
            
            # Get the radius of the enclosing circle around the found contour
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)
            
            # Calculate centre
            M = cv2.moments(cnt)
            if M['m00'] != 0:
                centre = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
                return centre
        return None

    def triangulate(self, point1, point2):
        # Convert 2D to 3D
        points3D = cv2.traingulatePoints(self.proj1, self.proj2, point1, point2)

    def track(self, video1, video2):
        print("Loading Videos")
        cap1 = cv2.VideoCapture(video1)
        cap2 = cv2.VideoCapture(video2)

        ret, frame1 = cap1.read()
        ret, frame2 = cap2.read()
        
        # SIFT-based extrinsic calibration
        print("Performing SIFT-based extrinsic calibration")
        extrinsic_cal = False
        calib_frame1 = frame1.copy()
        calib_frame2 = frame2.copy()

        while not extrinsic_cal:
            display1 = calib_frame1.copy()
            display2 = calib_frame2.copy()

            cv2.putText(display1, "Show scene, press 'c'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display2, "Show scene, press 'c'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            combined = np.hstack([display1, display2])
            cv2.imshow('SIFT Extrinsic Calibration' , combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                if self.stereo_calibration(calib_frame1, calib_frame2):
                    extrinsic_cal = True
                    cv2.destroyWindow('SIFT Extrinsic Calibration')
                    break
            elif key == 27:
                print("calibration cancelled")
                cap1.release()
                cap2.release()
                cv2.destroyAllWindows()
                return
            
        # Colour picker
        roi = cv2.selectROI('Select region for colour sampling', frame1, False)
        if roi == (0,0,0,0):
            print("No region was selected. Using yellow threshold.")
            self.lower_colour = np.array([20, 100, 100])   
            self.upper_colour = np.array([30, 255, 255])  

        else:
            #Extract ROI
            x,y,w,h = [int(i) for i in roi]
            colour_sample = frame1[y:y+h, x:x+w]

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
            h_range = 2 * h_std
            s_range = 2 * s_std
            v_range = 2 * v_std
            
            self.lower_colour = np.array([max(0, h_low - h_range), 
                                max(0, s_low - s_range), 
                                max(0, v_low - v_range)] , dtype=np.uint8)
            self.upper_colour = np.array([min(255, h_high + h_range), 
                                min(255, s_high + s_range), 
                                min(255, v_high + v_range)] , dtype=np.uint8)
            
            print(f"Selected colour range:")
            print(f"H: {self.lower_colour[0]:.1f} - {self.upper_colour[0]:.1f}")
            print(f"S: {self.lower_colour[1]:.1f} - {self.upper_colour[1]:.1f}")
            print(f"V: {self.lower_colour[2]:.1f} - {self.upper_colour[2]:.1f}")

        cv2.destroyAllWindows()


        cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        cap2.set(cv2.CAP_PROP_POS_FRAMES, 1)

        frame_count = 0

        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2= cap2.read()
            if not ret1 or not ret2:
                break

            frame1_rectify = cv2.remap(frame1, self.rectify1[0], self.rectify1[1], cv2.INTER_LINEAR)
            frame2_rectify = cv2.remap(frame2, self.rectify2[0], self.rectify2[1], cv2.INTER_LINEAR)

            centroid1= self.detect(frame1_rectify, self.lower_colour, self.upper_colour)
            centroid2 = self.detect(frame2_rectify, self.lower_colour, self.upper_colour)

            if centroid1 and centroid2:
                pt1 = np.array([[centroid1[0], centroid1[1]]], dtype=np.float32)
                pt2 = np.array([[centroid2[0], centroid2[1]]], dtype=np.float32)

                pt_3D = self.triangulate(pt1.T, pt2.T)
                

                cv2.circle(frame1_rectify, centroid1, 8, (0, 255, 0), 2)
                cv2.circle(frame2_rectify, centroid1, 8, (0, 255, 0), 2)

            combined = np.hstack([frame1_rectify, frame2_rectify])
            cv2.imshow('Stereo 3D tracking', combined)

            frame_count +=1

            if cv2.waitKey(30) & 0xFF == 27:
                break

        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()



    def save_extrinsic(self,filename):

        if self.R is None:
            print("No extrinsic calibration")
            return
        
        extrinsic_params = {
            'R': self.R,
            'T': self.T,
            'E': self.E,
            'F': self.F,
            'rectify1': self.rectify1,
            'rectify2': self.rectify2,
            'proj1': self.proj1,
            'proj2': self.proj2,
            'Q': self.Q

        }

        with open(filename, 'wb') as f:
            json.dump(extrinsic_params, f)

        print(f" Extrinsic parameters saved to {filename}")

        def load_extrinsic(self, filename):

            with open(filename, 'rb') as f:
                extrinsic_params = json.load(f)

                self.R = extrinsic_params['R']
                self.T = extrinsic_params['T']
                self.E = extrinsic_params['E']
                self.F = extrinsic_params['F']
                self.rectify1 = extrinsic_params['rectify1']
                self.rectify2 = extrinsic_params['rectify2']
                self.proj1 = extrinsic_params['proj1']
                self.proj2 = extrinsic_params['proj2']
                self.Q = extrinsic_params['Q']


if __name__ == "__main__":

    tracker = StereoCamera('intrinsic_calibration.json')

    tracker.track('video1.mp4', 'video2.mp4')

    tracker.save_extrinsic('extrinsic_calibration.json')




