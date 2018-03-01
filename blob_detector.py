import yaml
import cv2
import numpy as np 

params_file = open("budger_localizer_params.yaml", "r")
params = yaml.load(params_file)

class Detector:
    def __init__(self):
        """
        @brief      initialize the detector

        @param      None

        @return     None
        """
        self.params = cv2.SimpleBlobDetector_Params()
        self.detector = self.create_detector()
        self.keypoints = None
        self.centers = None

    def create_detector(self):
        """
        @brief      Setup SimpleBlobDetector parameters.
        
        @param      None
        
        @return     detector object
        
        """      
        # Change thresholds
        self.params.minThreshold = params["blob_detector"]["minThreshold"]
        self.params.maxThreshold = params["blob_detector"]["maxThreshold"]

        # Filter by Area.
        self.params.filterByArea = params["blob_detector"]["filterByArea"]
        self.params.minArea = params["blob_detector"]["minArea"]
        self.params.maxArea = params["blob_detector"]["maxArea"]

        # Filter by Circularity
        self.params.filterByCircularity = params["blob_detector"]["filterByCircularity"]
        self.params.minCircularity = params["blob_detector"]["minCircularity"]
         
        # Filter by Convexity
        self.params.filterByConvexity = params["blob_detector"]["filterByConvexity"]
        self.params.minConvexity = params["blob_detector"]["minConvexity"]
         
        # Filter by Inertia
        self.params.filterByInertia = params["blob_detector"]["filterByInertia"]
        self.params.minInertiaRatio = params["blob_detector"]["minInertiaRatio"]
         
        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            return cv2.SimpleBlobDetector(self.params)
        else : 
            return cv2.SimpleBlobDetector_create(self.params)
    
    def detect(self, image):
        """
        @brief      Apply the bolb detector
        
        @param      image: the input image (can be both color and grayscale image)
        
        @return     the centers of blobs {[[x, y], [x, y], ...]}
        """
        # make sure the input is grayscale image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        keypoints = self.detector.detect(image)
        centers = []
        for keypoint in keypoints:
            centers.append(list(np.array(keypoint.pt).astype(int)))
        self.keypoints = keypoints
        self.centers = np.array(centers)

        return np.array(centers)

    def drawKeypoints(self, image, color=(0, 0, 255)):
        # make sure the input is grayscale image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if self.keypoints is None:
            print "No keypoints found"
            return None

        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        image = cv2.drawKeypoints(image, self.keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return self.drawCross(image, self.keypoints)

    def drawCross(self, image, keypoints, color=(0, 255, 0), cross_length=20):
        half_length1 = np.array([cross_length/2, cross_length/2])
        half_length2 = np.array([cross_length/2, -cross_length/2])
        for keypoint in keypoints:
            pt = np.array(keypoint.pt).astype(int)
            cv2.line(image, tuple(pt - half_length1), tuple(pt + half_length1), color=color, thickness=3)
            cv2.line(image, tuple(pt - half_length2), tuple(pt + half_length2), color=color, thickness=3)
        return image
    
    def detectAndDrawKeypoints(self, image, color=(0, 0, 255)):
        """
        @brief      Detect circles and draw detected blobs.
        
        @param      image:  the input image (can be both color and grayscale image)
        
        @return     the result image
        """
        # make sure the input is grayscale image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect keypoints
        self.keypoints = self.detector.detect(image)
        centers = []
        for keypoint in self.keypoints:
            centers.append(list(np.array(keypoint.pt).astype(int)))
        self.centers = centers

        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        image = cv2.drawKeypoints(image, self.keypoints, np.array([]), color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return self.drawCross(image, self.keypoints)
