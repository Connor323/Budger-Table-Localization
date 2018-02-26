import cv2
import yaml
import numpy as np 
import cv2.aruco as aruco
import os.path as osp

params_file = open("budger_localizer_params.yaml", "r")
params = yaml.load(params_file)

class FiducialDetector:
	"""docstring for FiducialDetector"""
	def __init__(self):
		"""
		Detect the fiducial (Aruco). 
		"""
		self.aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_1000)
		self.arucoID = params["FiducialDetector"]["arucoID"]
		self.parameters =  aruco.DetectorParameters_create()
		# Adaptive Thresholding
		self.parameters.adaptiveThreshWinSizeMin = 10
		self.parameters.adaptiveThreshWinSizeMax = 25
		self.parameters.adaptiveThreshWinSizeStep = 10
		self.parameters.adaptiveThreshConstant = 7
		# Contour filtering
		self.parameters.minMarkerPerimeterRate = 0.2
		self.parameters.maxMarkerPerimeterRate = 4.0
		self.parameters.polygonalApproxAccuracyRate = 0.05
		self.parameters.minCornerDistanceRate = 0.05
		self.parameters.minDistanceToBorder = 3
		# Bits Extraction
		self.parameters.markerBorderBits = 1
		self.parameters.minOtsuStdDev = 5.0
		# self.parameters.perpectiveRemovePixelPerCell = 4
		self.parameters.perspectiveRemoveIgnoredMarginPerCell = 0.13
		# Marker Identification
		self.parameters.maxErroneousBitsInBorderRate = 0.35
		self.parameters.errorCorrectionRate = 0.6
		# Corner Refinement
		# self.parameters.doCornerRefinement = False
		# self.parameters.cornerRefinementWinSize = 5
		# self.parameters.cornerRefinementMaxIterations = 30
		# self.parameters.cornerRefinementMinAccuracy = 0.1		


	def findCenterFiducial(self, corners):
		"""
		Note: this function assumes the order of corners is ordered.
		Find the center of fiducial based on the intersection of the 4 corners. 
		Params:
			corners: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
		Return:
			center: [x, y]
		"""
		assert len(corners) == 4
		x0, y0 = corners[0]
		x1, y1 = corners[1]
		x2, y2 = corners[2]
		x3, y3 = corners[3]

		if abs(x0 - x2) < 1e-5 or abs(x1 - x3) < 1e-5: # in case that the two-diagonal corners aline with x/y axis.
			if abs(x0 - x2) < 1e-5 and abs(x1 - x3) >= 1e-5:
				x = np.mean([x0, x2])
				k2 = float(y1 - y3) / (x1 - x3)
				b2 = float(x1*y3 - y1*x3) / (x1 - x3)
				y = k2 * x + b2
			elif abs(x1 - x3) < 1e-5 and abs(x0 - x2) >= 1e-5:
				x = np.mean([x1, x3])
				k1 = float(y0 - y2) / (x0 - x2)
				b1 = float(x0*y2 - y0*x2) / (x0 - x2)
				y = k1 * x + b1
			else:
				raise ValueError("Incorrect value of corners: %f %f %f %f" % (x0, x1, x2, x3))
		else:
			k1 = float(y0 - y2) / (x0 - x2)
			b1 = float(x0*y2 - y0*x2) / (x0 - x2)
			k2 = float(y1 - y3) / (x1 - x3)
			b2 = float(x1*y3 - y1*x3) / (x1 - x3)

			x = float(b2 - b1) / (k1 - k2)
			y = x * k1 + b1

		return np.array([x, y])

	def detectFiducial(self, image, testMode=False):
		"""
		Detect fiducial.
		Params:
			image: input color image
			testMode: if true, don't save the fiducial point
		Return:
			if detect the fiducial or not (boolean)
			center of aruco tag
		"""
		img = image.copy()
		corners, ids, rejectedImgPoints = aruco.detectMarkers(img, self.aruco_dict, parameters=self.parameters)
		ret = False
		fiducialPt = None
		if corners is not None and len(corners) > 0:
			corners = np.squeeze(np.array(corners))
			if len(corners) != 4:
				print "Incorrect corners: {}".format(corners)
			elif ids != self.arucoID and not testMode:
				print "Incorrect Aruco ID: {}".format(ids)
			else:  
				fiducialPt = self.findCenterFiducial(corners)

				if testMode:
					print "Aruco ID: {}".format(ids)
					cv2.circle(img, tuple(fiducialPt.astype(int)), 10, (255, 0, 255), -1)
					path = osp.join(params["Localizer"]["result_Path"], "tmpFiducial.png")
					if cv2.imwrite(path, img):
						print "Save the Aruco detection result to %s" % path
				ret = True

		return ret, fiducialPt