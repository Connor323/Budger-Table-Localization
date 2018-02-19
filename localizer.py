import cv2
import copy
import yaml
import numpy as np 
from tf import transformations

from blob_detector import Detector
from camera import Camera 

params_file = open("budger_localizer_params.yaml", "r")
params = yaml.load(params_file)

class Localizer:
    def __init__(self):
        self.detector = Detector()
        self.camera = Camera()
        self.camera.enable() # starting camera node
        self.ROI = []
        self.image = None

    def ROI_by_mouse(self, event, x, y, flags, param):
        # if the left mouse button was clicked, record the starting
        # (x, y) coordinates and indicate that cropping is being
        # performed
        if event == cv2.EVENT_LBUTTONDOWN:
            self.ROI += [x, y]

    def localize(self, image=None):
        """
        @brief      Localizing table given undistorted image (using solvePerspectiveDistortaion
        
        @param      image: the undistorted image
        
        @return     pose: [x, y, z, roll, pitch, yaw]
        @return     extrinsic: a 3*4 ndarray 
        """
        if image is None:
            image = self.camera.frame_rect
        original_image = image.copy()

        # obtain ROI
        if params["Localizer"]["use_mouse_select"]:
            image_with_circles = self.detector.detectAndDrawKeypoints(image.copy())
            cv2.imshow("Select ROI by click top-left and button-right points", self.resize(image_with_circles))
            cv2.setMouseCallback("Select ROI by click top-left and button-right points", self.ROI_by_mouse)
            while len(self.ROI) < 4:
                cv2.waitKey(1)
            cv2.destroyWindow("Select ROI by click top-left and button-right points")
            self.ROI = (np.array(self.ROI) * params["Localizer"]["resize_ratio"]).tolist()
        else:
            self.ROI = params["Localizer"]["ROI"]

        print self.ROI

        image, H = self.solvePerspectiveDistortaionIteration(image) 
        # 
        # Note: use the undistorted image to find the circle centers, and 
        # convert the points back to the original image space; use these
        # points to further compute solvePnP
        # 

        points = self.detector.detect(image)
        image_with_circles = self.detector.drawKeypoints(image.copy())

        points, dist_2d, _ = self.filterPoints2D(image, points, return_dist=True)
        points = self.sortPotins2D(points, dist_2d)
        points = cv2.perspectiveTransform(np.array([points]).astype(np.float32), np.linalg.inv(H))[0]
         
        points3d, xys = self.matchPoints2Dwith3D(points, dist_2d=dist_2d, 
                                         dist_3d=params["Localizer"]["budger_distance_3d"], 
                                         doTranspose=True)

        extrinsic = self.computeExtrinsic(points, points3d)
        points3d_original = points3d.copy()

        
        cv2.imshow("Original Image", self.resize(image_with_circles))

        k = -1
        while k != 13:
            points, points3d = self.movePoints(points3d, k, extrinsic)
            self.drawPointsAndShow(points, xys, original_image.copy(), isFinal=False)
            k = cv2.waitKey()
        
        fianl_extrinsic = self.computeExtrinsic(points, points3d_original)
        points = cv2.perspectiveTransform(np.array([points]).astype(np.float32), H)[0]
        self.drawPointsAndShow(points, xys, image.copy(), isFinal=True)
        pose = self.convertExtrinsic2Pose(fianl_extrinsic, inverse=True)
        return pose, fianl_extrinsic

    def transformPts(self, points3d, extrinsic):
        points = []
        for pt3d in points3d:
            pt3d = pt3d.tolist()
            pt3d.append(1)
            pt = self.camera.camera_P.dot(extrinsic.dot(np.reshape(pt3d, [4, 1]))).flatten()
            pt = pt[:2] / pt[2]
            points.append(pt)
        return points

    def movePoints(self, points3d, key, extrinsic):
        points3d = np.array(points3d).copy()

        if key == 82: # up
            print "Detected key: up"
            points3d[:, 0] -= params["Localizer"]["budger_distance_3d"]
        elif key == 84: # down
            print "Detected key: down"
            points3d[:, 0] += params["Localizer"]["budger_distance_3d"]
        elif key == 81: # left
            print "Detected key: left"
            points3d[:, 1] -= params["Localizer"]["budger_distance_3d"]
        elif key == 83: # right
            print "Detected key: right"
            points3d[:, 1] += params["Localizer"]["budger_distance_3d"]
        else:
            print "   !!!! Unkown key detected"

        points = self.transformPts(points3d, extrinsic)
        return points, points3d


    def drawPointsAndShow(self, points, xys, image, isFinal=False):
        image_with_points = self.drawPoints(points, xys, image)
        if not isFinal:
            cv2.putText(image_with_points, "Press up/down/left/right to move points; press enter to confirm", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image_with_points, "Final Undistorted Result", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        cv2.imshow("Find real origin", self.resize(image_with_points))

    def drawPoints(self, points, xys, image):
        """
        @brief      draw 2D point on image
        
        @param      image: the image
        @param      points: the list of 2D points
        @param      xy: the list of 2D xy corrdinates
        
        @return     painted image
        """
        h, w = image.shape[:2]

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2COLOR)
        for i, (pt, xy) in enumerate(zip(points, xys)):
            cv2.circle(image, tuple(np.array(pt).astype(int)), 10, (i * 255 / len(points), 0, 0), -1)
            cv2.putText(image,'(%d, %d)' % (xy[0], xy[1]), (int(pt[0] + 10), int(pt[1] + 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
        center_camera = self.camera.camera_C
        cv2.circle(image, (int(center_camera[0]), int(center_camera[1])), 10, (i * 255 / len(points), 0, 0), -1)
        return image

    # TODO: find a way for testing 
    def testLocalization(self):
        pass

    def computeExtrinsic(self, pts2d, pts3d):
        """
        @brief      using solvePnP to compute extrinsic matrix from 2D to 3D given points
        
        @param      pts2d  The points in piexls
        @param      pts3d  The points in real world
        
        @return     The 3*4 extrinsic matrix
        """
        ret, rvect, tvect = cv2.solvePnP(np.array(pts3d).astype(np.float32), np.array(pts2d).astype(np.float32), self.camera.camera_P, None)
        if not ret:
            return None
        else:
            return self.convertvects2Extrinsic(tvect, rvect)

    def filterPoints2D(self, image, points, return_dist=False):
        """
        @brief      filter the 2D points based on the rough expected distance 
        
        @param      image             The image
        @param      points            The points in pixels
        @param      return_dist       If return the distance
        
        @return     points after filtering 
        @return     If return_dist == True, return the mean distance
        """
        def rect_contains(rect, point):
            '''
            Check if a point is inside the image

            Input: the size of the image 
                   the point that want to test

            Output: if the point is inside the image

            '''
            if point[0] < rect[0] :
                return False
            elif point[1] < rect[1] :
                return False
            elif point[0] > rect[2] :
                return False
            elif point[1] > rect[3] :
                return False
            return True

        points_dict = {}
        size = image.shape
        rect = (0, 0, size[1], size[0])
        subdiv  = cv2.Subdiv2D(rect)
        image_show = image.copy()
        for pt in points:
            if not rect_contains(tuple(self.ROI), pt): continue
            subdiv.insert(tuple(pt))
            points_dict[tuple(pt)] = []

        for t in subdiv.getTriangleList()[1:]: 
            pt1 = (int(t[0]), int(t[1]))
            pt2 = (int(t[2]), int(t[3]))
            pt3 = (int(t[4]), int(t[5]))
            if rect_contains(rect, pt1) and rect_contains(rect, pt2) and rect_contains(rect, pt3):
                points_dict[pt1] += [pt2, pt3]
                points_dict[pt2] += [pt1, pt3]
                points_dict[pt3] += [pt1, pt2]
                cv2.line(image_show, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA, 0)
                cv2.line(image_show, pt2, pt3, (0, 255, 0), 1, cv2.LINE_AA, 0)
                cv2.line(image_show, pt3, pt1, (0, 255, 0), 1, cv2.LINE_AA, 0)
        cv2.imshow("delaunay", self.resize(image_show))

        dists = []
        res = []
        for pt in points:
            if not rect_contains(tuple(self.ROI), pt): continue
            pts = list(set(points_dict[tuple(pt)]))
            dist = np.linalg.norm(pts - np.array(pt), axis=1)
            dist_diff = dist % params["Localizer"]["budger_distance_2d"]
            if (dist_diff < params["Localizer"]["tolerance_2d"]).any():
                res.append(pt)
                dists += dist[dist_diff < params["Localizer"]["tolerance_2d"]].tolist()

        if not return_dist:
            return np.array(res)
        else:
            return np.array(res), np.median(dists), np.mean(dists)

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

    def sortPotins2D(self, points, dist):
        """
        @brief      sort the 2D points based on the top-left point of their bounding box
        
        @param      points  The points in pixels
        @param      dist    The expected average distance between each point
        
        @return     sorted points
        """
        def getBbox(points):
            points = np.array(points)
            minx, miny, maxx, maxy = np.min(points[:, 0]), np.min(points[:, 1]), np.max(points[:, 0]), np.max(points[:, 1])
            return np.array([minx, miny, maxx-minx, maxy-miny])

        bbox = getBbox(points)
        dists = np.linalg.norm(points - bbox[:2], axis=1)
        origin = points[np.argmin(dists)]
        return sorted(points, key = lambda pt: (round(self.distanc2D(origin, pt, axis=1) / dist), 
                                                round(self.distanc2D(origin, pt, axis=0) / dist)))

    def matchPoints2Dwith3D(self, points, dist_2d, dist_3d, doTranspose):
        """
        @brief      given a list of sorted 2D points, compute their corresponding 3D points
        
        @param      points   The points in pixels
        @param      dist_2d  The expected average distance in pixels
        @param      dist_3d  The expected average distance in meter
        
        @return     3D points
        @return     corresponding corrdinates in x, y
        """
        pts3d = []
        origin = points[0]
        objp = [0., 0., 0.]
        pts3d.append(objp)
        xy = [[0, 0]]

        for pt in points[1:]:
            if doTranspose:
                y = round((pt[0] - origin[0]) / dist_2d)
                x = round((pt[1] - origin[1]) / dist_2d)
            else:
                x = round((pt[0] - origin[0]) / dist_2d)
                y = round((pt[1] - origin[1]) / dist_2d)
            objp = [x * dist_3d, y * dist_3d, 0.]
            pts3d.append(objp)
            xy.append([x, y])
        tmp = [tuple(pt) for pt in pts3d]
        assert len(tmp) == len(set(tmp)), "incorrect 3D points found: {}".format(xy)
        # if len(tmp) != len(set(tmp)):
        #     print "     -> !!!!incorrect 3D points!!!!"
        #     print xy
        return np.array(pts3d), np.array(xy)

    def convertExtrinsic2Pose(self, extrinsic, inverse=False):
        """
        @brief      convert the 3*4 extrinsic matrix to 6-value pose (x, y, z, roll, pitch, yaw)
        
        @param      extrinsic  The 3*4 extrinsic matrix
        
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

    def solvePerspectiveDistortaion(self, image, doTranspose=False):
        """
        @brief      using the feature points and expected distance to undistort the image using homography
        
        @param      image  The image with perspective distortion
        
        @return     undistorted image
        @return     homography matrix
        """
        h, w = image.shape[:2]
        points = self.detector.detect(image)
        points, mid_dist, mean_dist = self.filterPoints2D(image, points, return_dist=True)
        points = self.sortPotins2D(points, dist=mid_dist)
        pts3d, _ = self.matchPoints2Dwith3D(points, dist_2d=mean_dist, dist_3d=mean_dist, doTranspose=doTranspose)
        pts3d[:, :2] += points[0]
        H, status = cv2.findHomography(np.array(points), pts3d[..., :2])
        return cv2.warpPerspective(image, H, (w, h)), H, mean_dist

    def solvePerspectiveDistortaionIteration(self, image, doTranspose=False):
        """
        @brief      iteratively compute the homograpy
        
        @param      image  The image
        
        @return     undistorted image
        @return     homography matrix
        """
        if image is None:
            image = self.camera.frame_rect

        prev_mean_dist = None
        H = np.identity(3, np.float32)
        for i in range(params["Localizer"]["max_iteration"]):
            image, currH, mean_dist = self.solvePerspectiveDistortaion(image, doTranspose=doTranspose)
            H = H.dot(currH)
            if prev_mean_dist is None:
                prev_mean_dist = mean_dist
            else:
                if abs(prev_mean_dist - mean_dist) < 1:
                    return image, H
        print "Reach the maximum iteration! Current difference of mean distance is %f" % abs(prev_mean_dist - mean_dist)
        return image, H

    def resize(self, image):
        """
        @brief      resize the image given ratio
        
        @param      image  The image
        @param      ratio  The resize ratio
        
        @return     resized image
        """
        h, w = image.shape[:2]
        return cv2.resize(image, (w / params["Localizer"]["resize_ratio"], h / params["Localizer"]["resize_ratio"]))

    def testPerspective(self, image=None, max_iteration=params["Localizer"]["max_iteration"]):
        """
        @brief      test the solvePerspectiveDistortaion method 
        
        @param      image          The image with perspective distortion
        @param      max_iteration  The maximum number of iteration
        
        @return     None
        """
        if image is None:
            image = self.camera.frame_rect

        original_image = image.copy()
        cv2.imshow("original", self.resize(original_image))
        for i in range(max_iteration):
            image, _ = self.solvePerspectiveDistortaion(image)
            cv2.imshow("result", self.resize(image))
            image_with_circles = self.detector.detectAndDrawKeypoints(image)
            cv2.imshow("blub detection", self.resize(image_with_circles))
            cv2.waitKey()

if __name__ == "__main__":
    # image = cv2.imread("../data/images/budger1.jpg")
    # image = cv2.imread("table1.png")
    localizer = Localizer()
    print localizer.localize()[0]
    # localizer.testPerspective()
    while True:
        k = cv2.waitKey()
        if k == 27:
            break
