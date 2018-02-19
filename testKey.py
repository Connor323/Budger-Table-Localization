import cv2
import numpy as np 

img = np.zeros([512, 512], np.uint8)

cv2.imshow("img", img)
while True:	
	k = cv2.waitKey()
	print k