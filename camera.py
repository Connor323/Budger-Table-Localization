import threading
import rospy
import yaml
import image_geometry
import time
import numpy as np

from subprocess import call
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

params_file = open("budger_localizer_params.yaml", "r")
params = yaml.load(params_file)

class Camera:
	"""docstring for Camera"""
	def __init__(self, topic_rect=params["Camera"]["topic_rect"], topic_info=params["Camera"]["topic_info"]):
		"""
		Camera class for updating camera frames, camera information and publising the debug image.
		Params:
			topic_rect: the ROS topic of image_rect
			topic_info: the ROS topic of camera_info
		"""
		rospy.init_node('camera', anonymous=True)

		self.frame_rect = None
		self.debugImage = None
		self.debugImageOK = False
		self.camera_info_msg = None
		self.bridge = CvBridge()
		self.camera_rect = rospy.Subscriber(topic_rect, Image, self.callback_rect)
		self.debugImagePub = rospy.Publisher("debugImage", Image, queue_size=2)
		self.camera_model = image_geometry.PinholeCameraModel()

		self.setAutoExposure(True)
		self.setAutoFrameRate(True)
		self.changeSubsampling(1)
		time.sleep(0.5)

		self.camera_info = rospy.Subscriber(topic_info, CameraInfo, self.callback_info)
		rospy.wait_for_message(params["Camera"]["topic_info"], CameraInfo)
		rospy.wait_for_message(params["Camera"]["topic_rect"], Image)
		time.sleep(0.5)
		
		self.camera_model.fromCameraInfo(self.camera_info_msg)

	def callback_rect(self, data):
		"""
		Callback function of camera subscriber for updating debug image and image_rect.
		"""
		try:
			self.frame_rect = self.bridge.imgmsg_to_cv2(data, "bgr8")
			if not self.debugImageOK:
				self.debugImage = self.frame_rect
			self.debugImagePub.publish(self.bridge.cv2_to_imgmsg(self.debugImage, "bgr8"))
		except CvBridgeError as e:
			print(e)

	def callback_info(self, data):
		"""
			camera_info_msg.width
			camera_info_msg.height
			camera_info_msg.K
			camera_info_msg.D
			camera_info_msg.R
			camera_info_msg.P
		"""
		self.camera_info_msg = data
		camera_P = np.array(self.camera_info_msg.P)
		self.camera_P = np.delete(camera_P, [3, 7, 11]).reshape([3,3])
		self.camera_C = [camera_P[2], camera_P[6]]

	def setAutoExposure(self, value):
		"""
		Set the auto exposure using the ueye node.
		Params:
			value: Boolean. 
		"""
		if type(value) != type(True):
			print colored("Incorrect type of input, %s. Must be boolean" % type(value), "red")
		else:
			call(["rosrun", "dynamic_reconfigure", "dynparam", "set",
				     params["Camera"]["topic_ueye"],
				     "auto_exposure", "%s" % ("true" if value else "false")])

	def changeSubsampling(self, value):
		"""
		Change the subsampling value in ueye node.
		value: subsampling value.
		"""
		call(["rosrun", "dynamic_reconfigure", "dynparam", "set",
				     params["Camera"]["topic_ueye"],
				     "subsampling", str(value)])

	def changeExposure(self, value):
		"""
		Change the exposure value.
		Params:
			value: exposure value.
		"""
		call(["rosrun", "dynamic_reconfigure", "dynparam", "set",
				     params["Camera"]["topic_ueye"],
				     "exposure", str(value)])

	def setAutoFrameRate(self, value):
		"""
		Set the ueye node to auto frame rate for reducing motion blurry. 
		value: boolean. 
		"""
		if type(value) != type(True):
			print colored("Incorrect type of input, %s. Must be boolean" % type(value), "red")
		else:
			call(["rosrun", "dynamic_reconfigure", "dynparam", "set",
				     params["Camera"]["topic_ueye"],
				     "auto_frame_rate", "%s" % ("true" if value else "false")])
			
	def start(self):
		"""
		Spin the camera node. 
		"""
		rospy.spin()

	def enable(self):
		cameraThread = threading.Thread(target=self.start)
		cameraThread.daemon = True
		cameraThread.start()