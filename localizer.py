import os
import cv2
import copy
import time 
import yaml
import numpy as np 
from tf import transformations
from subprocess import call 
import lxml.etree as ET

from blob_detector import Detector
from camera import Camera 
from fiducialDetector import FiducialDetector

params_file = open("budger_localizer_params.yaml", "r")
params = yaml.load(params_file)

call(["mkdir", "-p", params["Localizer"]["result_Path"]])

class Localizer:
    def __init__(self):
        self.detector = Detector()
        self.fiducial_detector = FiducialDetector()
        self.camera = Camera()
        self.camera.enable() # starting camera node
        self.init_pts = {} # {2D pixel : 3D coordinates}
        self.image = None
        self.extrinsic = None
        self.model, self.model_reverse = self.getModelFromFile()

    def getModelFromFile(self):
        model, model_reverse = {}, {}
        tree = ET.parse(params["Localizer"]["model_file"])
        root = tree.getroot()
        for elem in root:  
            tmp = elem.attrib
            ID = tuple([int(s) for s in tmp["ID"][1:-1].split(",")])
            coordinates =  tuple([float(s) for s in tmp["Coordinates"][1:-1].split(",")])
            model[ID] = coordinates
            model_reverse[coordinates] = ID
        return model, model_reverse

    def select_by_mouse(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        def find_detected_point(x, y):
            x *= params["Localizer"]["resize_ratio"]
            y *= params["Localizer"]["resize_ratio"]
            dists = np.linalg.norm(self.detector.centers - np.array([x, y]), axis=1)
            min_idx = np.argmin(dists)
            return tuple(self.detector.centers[min_idx])

        if event == cv2.EVENT_LBUTTONDOWN:
            self.init_pts[find_detected_point(x, y)] = None

    def getRealOriginCoordinates(self):
        """
        @brief      obtain the real-world coordinates for the origin point 
                
        @return     real_origin_coordinates: [x, y] where x, y are integer
        """
        real_origin_coordinates = None
        while real_origin_coordinates is None:
            tmp = raw_input("Please input the origin coordinates (x, y) in real world (default %d, %d): " %\
                                 (params["Localizer"]["real_origin_coordinates"][0], params["Localizer"]["real_origin_coordinates"][1]))
            try:
                tmp = [int(ss) for ss in tmp.split(",")]
            except ValueError:
                if len(tmp) != 0:
                    print "Input is not a valid format. Should be: x, y"
                else:
                    real_origin_coordinates = params["Localizer"]["real_origin_coordinates"]
                continue
            if len(tmp) == 2:
                real_origin_coordinates = tmp
            else:
                print "Input is not a valid format. Should be: x, y"
        return real_origin_coordinates

    def getInitPtsPairs(self, image_with_circles):
        """
        @brief      manually selected a number of detected-real world point pair
                
        @param      image_with_circles: the original image with circle marks
        """
        while params["Localizer"]["num_initial_points"] > len(self.init_pts):
            cv2.imshow("Select points for computing initial points", self.resize(image_with_circles))
            cv2.setMouseCallback("Select points for computing initial points", self.select_by_mouse)

            prev_num_init_pts = len(self.init_pts)
            while len(self.init_pts) == prev_num_init_pts:
                cv2.waitKey(1)

            cv2.destroyWindow("Select points for computing initial points")

            real_origin_coordinates = None
            while real_origin_coordinates is None:
                tmp = raw_input("Please input the model relative coordinates (x, y): ")
                try:
                    tmp = [int(ss) for ss in tmp.split(",")]
                except ValueError:
                    print "Input is not a valid format. Should be: x, y"
                    continue
                if len(tmp) == 2:
                    real_origin_coordinates = tmp
                else:
                    print "Input is not a valid format. Should be: x, y"
            
            for key, item in self.init_pts.items():
                if item is None:
                    self.init_pts[key] = self.model[tuple(real_origin_coordinates)]
                    break

            xys = []
            for value in  self.init_pts.values():
                xys.append(self.model_reverse[value])
            self.drawPoints(self.init_pts.keys(), xys, image_with_circles)


    def localize(self, image=None):
        """
        @brief      Localizing table given undistorted image (using solvePerspectiveDistortaion
        
        @param      image: the undistorted image
        
        @return     pose: [x, y, z, roll, pitch, yaw]
        @return     extrinsic: a 3*4 ndarray 
        """
        def clip_angle(angle):
            return abs(angle) % 1.57

        if image is None:
            image = self.camera.frame_rect
        original_image = image.copy()

        # initial detection
        image_with_circles = self.detector.detectAndDrawKeypoints(image.copy())

        # manually select pairs of points and compute the initial pose
        self.getInitPtsPairs(image_with_circles)
        points = np.array(self.init_pts.keys())
        points3d = np.array(self.init_pts.values())
        init_extrinsic = self.computeExtrinsic(points, points3d)
        init_pose = self.convertExtrinsic2Pose(init_extrinsic, inverse=True)
        if clip_angle(init_pose[-1]) > 0.1:
            xyAline = False
        else:
            xyAline = True

        # use the initial pose for homography undistortion
        image, H = self.solvePerspectiveDistortaion(image, init_extrinsic) 

        # use the undistorted image to detect budger
        points = self.detector.detect(image)
        image_with_circles = self.detector.drawKeypoints(image.copy())

        # project the points back to original image
        points = cv2.perspectiveTransform(np.array([points]).astype(np.float32), np.linalg.inv(H))[0]
         
        points2d, points3d, xys = self.matchPoints2Dwith3D(points, init_extrinsic)

        extrinsic = self.computeExtrinsic(points2d, points3d)
        points3d_original = points3d.copy()
        origin_pt3d = sorted(xys, key = lambda x: (x[0], x[1]))[0]
        for idx, pt3d in enumerate(xys):
            if (origin_pt3d == pt3d).all():
                origin_pt3d_idx = idx
        
        cv2.imshow("Original Image", self.resize(image_with_circles))

        k = -1
        while k != 13:
            points, points3d = self.movePoints(points3d, k, extrinsic, xyAline, origin_pt3d_idx)
            self.drawPointsAndShow(points, xys, original_image.copy(), message=params["Localizer"]["message_moving"])
            k = cv2.waitKey()
        cv2.destroyAllWindows()
        # real_origin_coordinates = self.getRealOriginCoordinates()
        
        fianl_extrinsic = self.computeExtrinsic(points, points3d_original)
        points = cv2.perspectiveTransform(np.array([points]).astype(np.float32), H)[0]
        self.extrinsic = fianl_extrinsic.copy()
        
        cv2.imshow("Original Image", self.resize(image_with_circles))
        self.drawPointsAndShow(points, xys, image.copy(), message=params["Localizer"]["message_final"])
        
        pose = self.convertExtrinsic2Pose(fianl_extrinsic, inverse=True)
        self.savePoseInXML(pose)
        
        while True:
            k = cv2.waitKey()
            if k == 27: # press esc to quit
                break

        return pose, fianl_extrinsic

    def movePoints(self, points3d, key, extrinsic, xyAline, origin_pt3d_idx):
        """
        @brief      Move the 3D points given the keyboard command
        
        @param      points3d: 3D points, [[x, y, z], [x, y, z], ...]
        @param      key: keyboard command
        @param      extrinsic: 3*4 extrinsic matrix
        
        @return     points: 2D points [[x, y], [x, y], ...]
        @return     points3d: 3D points [[x, y, z], [x, y, z], ...]
        """
        def applyRotationM(points3d, M):
            points3d_tmp = points3d.copy()
            points3d_tmp[:, 2] = 1
            points3d[:, :2] = M.dot(points3d_tmp.T).T
            return points3d

        def transformPts(points3d, extrinsic):
            points = []
            for pt3d in points3d:
                pt3d = pt3d.tolist()
                pt3d.append(1)
                pt = self.camera.camera_P.dot(extrinsic.dot(np.reshape(pt3d, [4, 1]))).flatten()
                pt = pt[:2] / pt[2]
                points.append(pt)
            return points

        points3d = np.array(points3d).copy()
        origin_pt3d = points3d[origin_pt3d_idx]

        if key == 82: # up
            print "Detected key: up"
            if not xyAline:
                points3d[:, 0] -= params["Localizer"]["budger_distance_3d"]
            else:
                points3d[:, 1] -= params["Localizer"]["budger_distance_3d"]
        elif key == 84: # down
            print "Detected key: down"
            if not xyAline:
                points3d[:, 0] += params["Localizer"]["budger_distance_3d"]
            else:
                points3d[:, 1] += params["Localizer"]["budger_distance_3d"]
        elif key == 81: # left
            print "Detected key: left"
            if not xyAline:
                points3d[:, 1] -= params["Localizer"]["budger_distance_3d"]
            else:
                points3d[:, 0] -= params["Localizer"]["budger_distance_3d"]
        elif key == 83: # right
            print "Detected key: right"
            if not xyAline:
                points3d[:, 1] += params["Localizer"]["budger_distance_3d"]
            else:
                points3d[:, 0] += params["Localizer"]["budger_distance_3d"]
        elif key == 55: # num: 7, left rotation
            print "Detected key: rotate left"
            if not xyAline:
                M = cv2.getRotationMatrix2D((origin_pt3d[0], origin_pt3d[1]), 90, 1)
            else:
                M = cv2.getRotationMatrix2D((origin_pt3d[0], origin_pt3d[1]), -90, 1)
            points3d = applyRotationM(points3d, M)
        elif key == 56: # num: 8, right rotation
            print "Detected key: rotate right"
            if not xyAline:
                M = cv2.getRotationMatrix2D((origin_pt3d[0], origin_pt3d[1]), -90, 1)
            else:
                M = cv2.getRotationMatrix2D((origin_pt3d[0], origin_pt3d[1]), 90, 1)
            points3d = applyRotationM(points3d, M)
        elif key == 57: # num: 9, flip along x and y
            print "Detected key: flip"
            tmp = points3d.copy()
            points3d[:, 0] = tmp[:, 1]
            points3d[:, 1] = tmp[:, 0]
        else:
            print "   !!!! Unkown key detected"

        points = transformPts(points3d, extrinsic)
        return points, points3d


    def drawPointsAndShow(self, points, xys, image, message=None):
        """
        @brief      draw 2D point and show the result image
        
        @param      image: the image
        @param      points: the list of 2D points
        @param      xys: the list of 2D xy coordinates
        @param      message: the info writing on top of image

        @return     None
        """
        image_with_points = self.drawPoints(points, xys, image)
        if message is not None:
            cv2.putText(image_with_points, message, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.imshow("Find real origin", self.resize(image_with_points))

    def drawCross(self, image, pt, color=(0, 255, 0), cross_length=20):
        half_length1 = np.array([cross_length/2, cross_length/2])
        half_length2 = np.array([cross_length/2, -cross_length/2])
        pt = np.array(pt).astype(int)
        cv2.line(image, tuple(pt - half_length1), tuple(pt + half_length1), color=color, thickness=3)
        cv2.line(image, tuple(pt - half_length2), tuple(pt + half_length2), color=color, thickness=3)
        return image

    def drawPoints(self, points, xys, image):
        """
        @brief      draw 2D point on image
        
        @param      image: the image
        @param      points: the list of 2D points
        @param      xys: the list of 2D xy coordinates
        
        @return     painted image
        """
        h, w = image.shape[:2]

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2COLOR)
        for i, (pt, xy) in enumerate(zip(points, xys)):
            cv2.circle(image, tuple(np.array(pt).astype(int)), 10, (255, 0, 0), -1)
            self.drawCross(image, pt)
            cv2.putText(image,'(%d, %d)' % (xy[0], xy[1]), (int(pt[0] + 10), int(pt[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        center_camera = self.camera.camera_C
        cv2.circle(image, (int(center_camera[0]), int(center_camera[1])), 10, (0, 255, 0), -1)
        return image

    def backProjection(self, pt2d, extrinsic, zHeight):
        """
        @brief      compute back-projection from 2D to 3D, given z (height of fiducial) 
                    a: | m11   m12   -x2d | b:     | m13 * height + tx |    
                       | m21   m22   -y2d |   -1 * | m23 * height + ty |    
                       | m31   m32   -1   |        | m33 * height + tz |        
        @param      pt2d  The point in 2D
        @return     pt3d  The point in 3D
        """
        projectionM = self.camera.camera_P.dot(extrinsic)
        a = np.zeros([3, 3], np.float64)
        a[:3, :2] = projectionM[:, :2]
        a[0, 2] = -1 * pt2d[0]
        a[1, 2] = -1 * pt2d[1]
        a[2, 2] = -1
        b = np.zeros([3, 1], np.float64)
        b[:, 0] = -1 * (projectionM[:, 2] * (-1) * zHeight + projectionM[:, 3])
        result = np.squeeze(np.linalg.solve(a, b))
        pt3d = [result[0], result[1], -1 * zHeight]

        return np.array(pt3d)


    def getModelSamplePoints(self, center, radius, nSample=100):
        """
        Sample the points from the circle model.
        Params: 
            center: center point
            radius: radius of the circle model. Unit: meter
            nSample: number of samples (this value can affect the percision of ICP)
        Return:
            array([[p0x, p0y], [p1x, p1y], ...])
        """
        sampleAngles = np.linspace(0, 2*np.pi, nSample).astype(np.float64)
        model = []
        for a in sampleAngles:
            pt = np.array([np.cos(a), np.sin(a)]) * radius + np.array(center[:2]).astype(np.float64)
            model.append(pt.tolist() + [center[2]])
        return np.array(model)

    def testLocalization(self, image_input=None, extrinsic=None):
        """
        @brief      test the localization accuracy based on the Aruco test
        
        @param      image_input: the distorted image
        @param      extrinsic: the 3*4 extriansic matrix from table to camera
        
        @return     None
        """
        def transformPts(points3d, extrinsic):
            points = []
            for pt3d in points3d:
                pt3d = pt3d.tolist()
                pt3d.append(1)
                pt = self.camera.camera_P.dot(extrinsic.dot(np.reshape(pt3d, [4, 1]))).flatten()
                pt = pt[:2] / pt[2]
                points.append(pt)
            return points

        if image_input is None:
            image = self.camera.frame_rect
        else:
            image = image_input.copy()
        if self.extrinsic is not None:
            extrinsic = self.extrinsic.copy()
        assert image is not None
        assert extrinsic is not None

        while True:
            if image_input is None:
                image = self.camera.frame_rect
            else:
                image = image_input.copy()

            ret, fiducial_center = self.fiducial_detector.detectFiducial(image)
            if not ret:
                print "Fail to detect fiducial"
            else:
                cv2.circle(image, tuple(fiducial_center.astype(int)), 10, (255, 0, 255), -1)
                fiducial_center_3d = self.backProjection(fiducial_center, 
                                                         extrinsic, 
                                                         zHeight=params["FiducialDetector"]["fiducial_height"])
                print "Fiducial Center Location: {}".format(fiducial_center_3d)
                model_pts3d = self.getModelSamplePoints(fiducial_center_3d, radius=params["FiducialDetector"]["circle_radius"])
                model_pts2d = transformPts(model_pts3d, extrinsic)
                for pt2d in model_pts2d:
                    cv2.circle(image, tuple(np.array(pt2d).astype(int)), 2, (0, 255, 0), -1)

                cv2.imshow("Test Result", self.resize(image))
                k = cv2.waitKey(10)
                if k == 27:
                    break

    def testFiducialDetection(self, image=None):
        """
        @brief      test the fiducial detector
        
        @param      image: the distorted image
        
        @return     None
        """
        if image is None:
            image = self.camera.frame_rect

        print "detecting... "
        ret, fiducial_center = self.fiducial_detector.detectFiducial(image, testMode=True)
        if not ret:
            print "Fail to detect fiducial! "
        else:
            print "Detect fiducial successful! "
            cv2.circle(image, tuple(fiducial_center.astype(int)), 10, (255, 0, 255), -1)
            cv2.imshow("Detection Result", self.resize(image))
            cv2.waitKey()

    def computeExtrinsic(self, pts2d, pts3d, real_origin_coordinates=None):
        """
        @brief      using solvePnP to compute extrinsic matrix from 2D to 3D given points
        
        @param      pts2d  The points in piexls
        @param      pts3d  The points in real world
        @param      real_origin_coordinates  The origin point coordinates in real world
        
        @return     The 3*4 extrinsic matrix
        """
        pts3d = np.array(pts3d).astype(np.float32)
        pts2d = np.array(pts2d).astype(np.float32)

        if real_origin_coordinates is not None:
            pts3d[:, 0] += real_origin_coordinates[0] * params["Localizer"]["budger_distance_3d"]
            pts3d[:, 1] += real_origin_coordinates[1] * params["Localizer"]["budger_distance_3d"]

        ret, rvect, tvect = cv2.solvePnP(pts3d, pts2d, self.camera.camera_P, None)
        if not ret:
            return None
        else:
            return self.convertvects2Extrinsic(tvect, rvect)

    def distanc2D(self, pt1, pt2, axis=None):
        """
        @brief      compute the distance of 2D points
        
        @param      pt1   The point 1
        @param      pt2   The point 2
        @param      axis  The axis of the points to be computed
        
        @return     distance in float
        """
        if axis is None:
            return np.linalg.norm(np.array(pt1) - np.array(pt2))
        else:
            return np.linalg.norm(np.array(pt1[axis]) - np.array(pt2[axis]))

    def matchPoints2Dwith3D(self, points, init_extrinsic):
        """
        @brief      given a list of sorted 2D points, compute their corresponding 3D points
        
        @param      points          The points in pixels
        @param      init_extrinsic  the initial extransic matrix 
        
        @return     2D points
        @return     3D points
        @return     corresponding coordinates in x, y
        """
        pts3d, xys, pts2d = [], [], []
        model_pts_values = np.array(self.model.values())
        model_pts_keys = np.array(self.model.keys())
        for pt in points:
            tmp_pt3d = self.backProjection(pt, init_extrinsic, zHeight=params["Localizer"]["budger_height"])
            dists = np.linalg.norm(model_pts_values - tmp_pt3d, axis=1)
            min_idx = np.argmin(dists)
            min_dist = dists[min_idx]
            if min_dist < params["Localizer"]["tolerance_3d"]: 
                pts2d.append(pt)
                pts3d.append(model_pts_values[min_idx])
                xys.append(model_pts_keys[min_idx])

        return np.array(pts2d), np.array(pts3d), np.array(xys)

    def convertExtrinsic2Pose(self, extrinsic, inverse=False):
        """
        @brief      convert the 3*4 extrinsic matrix to 6-value pose (x, y, z, roll, pitch, yaw)
        
        @param      extrinsic  The 3*4 extrinsic matrix
        @param      inverse  if inverse the matrix 
        
        @return     [x, y, z, roll, pitch, yaw]
        """
        if inverse:
            tmp = np.identity(4, np.float32)
            tmp[:3] = extrinsic
            extrinsic = np.linalg.inv(tmp)[:3]

        r = extrinsic[:3, :3]
        # angle = cv2.Rodrigues(r)[0]
        angle = transformations.euler_from_matrix(r)
        translation = extrinsic[:3, 3]
        return np.concatenate([np.squeeze(translation), np.squeeze(angle)])

    def convertvects2Extrinsic(self, tvect, rvect):
        """
        @brief      convert the rotation vecter and translation vector to 3*4 extrinsic matrix
        
        @param      tvect  The translation vector
        @param      rvect  The rotation vector
        
        @return     3*4 extrinsic matrix
        """
        rotationM = cv2.Rodrigues(rvect)[0]
        translationM = np.array(tvect).reshape(3)

        ExCamera2Table = np.zeros([3, 4], np.float64)
        ExCamera2Table[:, :3] = rotationM
        ExCamera2Table[:, 3] = translationM
        return ExCamera2Table

    def projectionPts3D(self, pts3d, extrinsic):
        """
        @brief      using the feature points and expected distance to undistort the image using homography
        
        @param      pts3d  the list of 3D points, [[x, y, z], [x, y, z], ...]
        @param      extrinsic  the extrinsic matrix 
        
        @return     pts3d  the list of 2D points, [[x, y], [x, y], ...]
        """
        pts2d = []
        for pt in pts3d:
            pt = pt.tolist()
            pt.append(1)
            tmp = self.camera.camera_P.dot(extrinsic.dot(np.array(pt).reshape([4, 1])))
            pts2d.append(tmp.flatten())
        return np.array(pts2d) 

    def solvePerspectiveDistortaion(self, image, init_extrinsic):
        """
        @brief      using the feature points and expected distance to undistort the image using homography
        
        @param      image  The image with perspective distortion
        @param      init_extrinsic  The initial extrinsic matrix 
        
        @return     undistorted image
        @return     homography matrix
        """
        h, w = image.shape[:2]
        points = self.detector.detect(image)
        points2d, points3d, xys = self.matchPoints2Dwith3D(points, init_extrinsic)
        points2d_model = self.projectionPts3D(points3d, init_extrinsic)
        H, status = cv2.findHomography(points2d, points2d_model)
        return cv2.warpPerspective(image, H, (w, h)), H

    # def solvePerspectiveDistortaionIteration(self, image, doTranspose=False):
    #     """
    #     @brief      iteratively compute the homograpy
        
    #     @param      image  The image
    #     @param      doTranspose  if transpose the x, y coordinates
        
    #     @return     undistorted image
    #     @return     homography matrix
    #     """
    #     if image is None:
    #         image = self.camera.frame_rect

    #     prev_mean_dist = None
    #     H = np.identity(3, np.float32)
    #     for i in range(params["Localizer"]["max_iteration"]):
    #         image, currH = self.solvePerspectiveDistortaion(image, doTranspose=doTranspose)
    #         H = H.dot(currH)
    #         if prev_mean_dist is None:
    #             prev_mean_dist = mean_dist
    #         else:
    #             if abs(prev_mean_dist - mean_dist) < 1:
    #                 return image, H
    #     print "Reach the maximum iteration! Current difference of mean distance is %f" % abs(prev_mean_dist - mean_dist)
    #     return image, H

    def resize(self, image):
        """
        @brief      resize the image given ratio
        
        @param      image  The image
        
        @return     resized image
        """
        h, w = image.shape[:2]
        return cv2.resize(image, (w / params["Localizer"]["resize_ratio"], h / params["Localizer"]["resize_ratio"]))

    def savePoseInXML(self, pose):
        """
        Save the final result into yaml and xml files.
        Params:
            pose: the list of 6 values. eg: [roll, pitch, yaw, x, y, z]
        """
        pose = np.array(pose).tolist()
        
        with open(os.path.join(params["Localizer"]["result_Path"], 'CameraPose.xml'), 'w') as outfile:
            string = inner_contents = '<origin x="{}" y="{}" z="{}" roll="{}" pitch="{}" yaw="{}" />'.format(pose[0],
                                                                                                             pose[1],
                                                                                                             pose[2],
                                                                                                             pose[3],
                                                                                                             pose[4],
                                                                                                             pose[5])
            outfile.write(string)
            print "Pose: ", string
        print "Final pose saves in ", os.path.join(params["Localizer"]["result_Path"], "CameraPose.xml")

extrinsic = np.array([[ 0.99997345,  0.00602243, -0.00410289, -0.1604342 ],
 [-0.00597383,  0.99991304,  0.01175732, -0.32755519],
 [ 0.00417335, -0.01173249,  0.99992246,  1.14454347]], np.float32)

if __name__ == "__main__":
    # image = cv2.imread("table1.png")
    localizer = Localizer()
    localizer.localize()
    print "extrinsic: \n", localizer.extrinsic
    localizer.testLocalization(extrinsic=extrinsic)
    # localizer.testFiducialDetection()