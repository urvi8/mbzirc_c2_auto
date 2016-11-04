#!/usr/bin/python

from cv2 import cv
import cv2.cv as cv
import cv2
import numpy as np;

cap = cv2.VideoCapture('/home/urvi/Desktop/MBZIRC/Image Processing/video1.mp4')
#cap = cv2.VideoCapture(1)
while(cap.isOpened()):
    # Capture frame-by-frame
	ret, frame = cap.read()
	kalman = cv2.cv.CreateKalman(4,2,0)

	def initkalman(x,y):
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
	
		cv.SetIdentity(kalman.measurement_matrix,cv.RealScalar(1))
		cv.SetIdentity(kalman.process_noise_cov,cv.RealScalar(1e-3))
		cv.SetIdentity(kalman.measurement_noise_cov,cv.RealScalar(0.5))
		cv.SetIdentity(kalman.error_cov_post,cv.RealScalar(0.1))

	def kalmanpredict():
		predict = cv.KalmanPredict(kalman)
		prex = predict[0,0]
		prey = predict[1,0]
		return predict 

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
	params.minArea = 400
	params.maxArea = 22000

# Filter by Circularity
	params.filterByCircularity = True
	params.minCircularity = 0.03
	params.maxCircularity = 1

# Filter by Convexity
	params.filterByConvexity = False
	params.minConvexity = 0.01
	params.maxConvexity = 0.5

# Filter by Inertia
	params.filterByInertia = True
	params.minInertiaRatio = 0.05
	params.maxInertiaRatio = 0.55
	
# detector with the parameters
	detector = cv2.SimpleBlobDetector(params)

# Detect blobs.
	keypoints = detector.detect(frame)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

	im = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	initkalman(0,0)
	for i in range(len(keypoints)):
		KP = kalmanpredict()
		x1 = keypoints[i].pt[0]
		y1 = keypoints[i].pt[1]
	
		KC = kalmancorrect(x1,y1)
		xval = KC[0,0]
		yval = KC[1,0]
#		xvalp = KP[0,0]
#		yvalp = KP[1,0]
#		cv2.circle(im, (int(xvalp),int(yvalp)),9,(0,255,0),4)	
		cv2.circle(im, (int(xval),int(yval)),8,(255,0,0),3)
#		print 'The x value is ', xval
#		print 'The y value is ', yval
		x_w = (x1 - 610.966)*(-1.2)/1003.7624
		y_w = (y1 - 361.7742)*(-1.2)/1002.6516

#		print "The location is at ", x_w, y_w

# Show blobs

	cv2.imshow("Keypoints", im)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



