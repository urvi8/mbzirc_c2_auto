 #!/usr/bin/env python

import roslib
#roslib.load_manifest('/home/urvi/mbzric/src/simulation')
import sys
import rospy
import cv2
from cv2 import cv
import cv2.cv as cv
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class image_converter:
 
	def __init__(self):
		self.image_pub = rospy.Publisher("image_1",Image,queue_size = 20)
 
		self.bridge = CvBridge()
		self.image_sub = rospy.Subscriber("/ur5_arm_camera/image_raw",Image,self.callback)
 
	def callback(self,data):
		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)
		
		def kalmanpredict():
			predict = cv.KalmanPredict(kalman)
#			return predict 

		def kalmancorrect(x, y):
			measurement = cv.CreateMat(2,1,cv.CV_32FC1)
			measurement[0,0] = x
			measurement[1,0] = y
		
			kalman.state_pre[0,0] = x
			kalman.state_pre[1,0] = y
			kalman.state_pre[2,0] = 0
			kalman.state_pre[3,0] = 0

			correct = cv.KalmanCorrect(kalman, measurement)
			return correct

# Setup SimpleBlobDetector parameters.

		params = cv2.SimpleBlobDetector_Params()

		params.minThreshold = 40
		params.maxThreshold = 125

# Filter by Area.
		params.filterByArea = True
		params.minArea = 80
		params.maxArea = 5000
	
# Filter by Circularity
		params.filterByCircularity = True
		params.minCircularity = 0.1
		params.maxCircularity = 0.95
	
# Filter by Convexity
		params.filterByConvexity = False
#		params.minConvexity = 0.01
#		params.maxConvexity = 0.25
	
# Filter by Inertia
		params.filterByInertia = True
		params.minInertiaRatio = 0.02
		params.maxInertiaRatio = 0.15
	
# detector with the parameters
		detector = cv2.SimpleBlobDetector(params)

# Detect blobs.
		keypoints = detector.detect(cv_image)

# Draw detected blobs as red circles.

		im = cv2.drawKeypoints(cv_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Kalman filter
	
		for i in range(len(keypoints)):
#			KP = kalmanpredict()
			kalmanpredict()

			x1 = keypoints[i].pt[0]
			y1 = keypoints[i].pt[1]
	
			KC = kalmancorrect(x1,y1)
			xval = KC[0,0]
			yval = KC[1,0]
#			xvalp = KP[0,0]
#			yvalp = KP[1,0]
#			cv2.circle(im, (int(xvalp),int(yvalp)),9,(0,255,0),4)	
			cv2.circle(im, (int(xval),int(yval)),20,(255,0,0),3)


# Show blobs
		cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
		cv2.imshow("Image window",im)
		cv2.waitKey(3)

		try:
			self.image_pub.publish(self.bridge.cv2_to_imgmsg(im, "bgr8"))
		except CvBridgeError as e:
			print(e)

def initkalman(x,y):

	global kalman
	kalman = cv.CreateKalman(4,2,0)

	kalman.state_pre[0,0] = x
	kalman.state_pre[1,0] = y
	kalman.state_pre[2,0] = 0
	kalman.state_pre[3,0] = 0

	kalman.transition_matrix[0,0] = 1
	kalman.transition_matrix[0,2] = 1
	kalman.transition_matrix[1,3] = 1
	kalman.transition_matrix[1,1] = 1
	kalman.transition_matrix[2,2] = 1
	kalman.transition_matrix[3,3] = 1
		
	cv.SetIdentity(kalman.measurement_matrix, cv.RealScalar(1))
	cv.SetIdentity(kalman.process_noise_cov, cv.RealScalar(1e-4))
	cv.SetIdentity(kalman.measurement_noise_cov, cv.RealScalar(1e-1))
	cv.SetIdentity(kalman.error_cov_post, cv.RealScalar(0.1))

def main(args):
	initkalman(0,0)	
	ic = image_converter()
	rospy.init_node('image_converter', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")
	cv2.destroyAllWindows()

if __name__ == '__main__':
	main(sys.argv)
