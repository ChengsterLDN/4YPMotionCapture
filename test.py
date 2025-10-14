import cv2
import numpy as np
import pandas as pd
import pickle

class StereoCamera:
    def __init__(self):
        self.R = None # Rotation matrix
        self.F = None # Fundamental matrix
        self.T = None # Translation Vector
        self.E = None # Essential Matrix - camera matrix * rotation matrix
        self.rectify1 = None
        self.rectify2 = None
        self.proj1 = None
        self.proj2 = None
        self.Q = None

    def load_intrinsic_props(self, file):
        with open(file, 'r') as f:
            calibration = pickle.load(f)
        self.K = calibration[0] # Camera Matrrix - could be ['camera_matrix' - see JSON or pickle file]
        self.D = calibration[1] # Distortion Vector
    
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

        self.compute(grey1[::-1])

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

    
    def detect(self, frame, lower_colour, upper_colour):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_colour, upper_colour)

        # Denoise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, anchor = (-1,-1), iterations = 20)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, anchor = (-1,-1), iterations = 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        
        def drawlines(frame1,frame2,lines,pts1,pts2):
            ''' img1 - image on which we draw the epilines for the points in img2
                lines - corresponding epilines '''
            r,c = frame1.shape
            frame1 = cv2.cvtColor(frame1,cv2.COLOR_GRAY2BGR)
            frame2 = cv2.cvtColor(frame2,cv2.COLOR_GRAY2BGR)
            for r,pt1,pt2 in zip(lines,pts1,pts2):
                color = tuple(np.random.randint(0,255,3).tolist())
                x0,y0 = map(int, [0, -r[2]/r[1] ])
                x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
                frame1 = cv2.line(frame1, (x0,y0), (x1,y1), color,1)
                frame1 = cv2.circle(frame1,tuple(pt1),5,color,-1)
                frame2 = cv2.circle(frame2,tuple(pt2),5,color,-1)
            return frame1,frame2

        # Find epilines corresponding to points in right image (second image) and drawing its lines on left image
        lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
        lines1 = lines1.reshape(-1,3)
        img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)

        # Find epilines corresponding to points in left image (first image) and drawing its lines on right image
        lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
        lines2 = lines2.reshape(-1,3)
        img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)

        plt.subplot(121),plt.imshow(img5)
        plt.subplot(122),plt.imshow(img3)
        plt.show()



