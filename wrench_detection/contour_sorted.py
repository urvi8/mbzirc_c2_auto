#!/usr/bin/python

import cv2
#import cv2 as cv2.cv
import numpy as np
import sys

cap = cv2.VideoCapture('/home/urvi/Desktop/MBZIRC/Image Processing/video.mp4')
#cap = cv2.VideoCapture('/home/urvi/Desktop/MBZIRC/Image Processing/sample.mkv')
#cap = cv2.VideoCapture(1)

def detect_wrench(frame):
	img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#img = cv2.imread('/home/urvi/Desktop/MBZIRC/Image Processing/snap1.png',0)
	ret,thresh = cv2.threshold(img,127,255,0)
	contours,hierarchy = cv2.findContours(thresh, 1, 2)
	for cnt in contours:

		x,y,w,h = cv2.boundingRect(cnt)
		if h > 85 and h < 370:
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,100),2)
			
	return contours, frame, img

def detect_size(contours,frame,img):

	scale_factor = 4
	pose = np.array([])
	for cnt in contours:

		x,y,w,h = cv2.boundingRect(cnt)
		if h > 85 and h < 370:
#		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
#		cv2.rectangle(frame,(x,y),(x+w,y+h/5),(0,0,255),2)
			crop = img[y-10:y+20+h/4,x:x+w]  # slice image to crop top part of the wrench
			crop = cv2.resize(crop,(0,0),fx=scale_factor,fy=scale_factor) #resize cropped image to detect circle
			crop1 = cv2.medianBlur(crop,5)
			circles = cv2.HoughCircles(crop1, cv2.cv.CV_HOUGH_GRADIENT,1,10,np.array([]),100,30,0,0)
			if circles != None:
				a,b,c = circles.shape
				for i in range(b):
					print circles[0][i][2]/scale_factor
					cv2.circle(crop,(circles[0][i][0], circles[0][i][1]), circles[0][i][2], (255,0,0), 1, cv2.CV_AA)
					cv2.circle(crop,(circles[0][i][0], circles[0][i][1]), 2, (255,100,0), 1, cv2.CV_AA)
					x0 = int(circles[0][i][0]/scale_factor)
					y0 = int(circles[0][i][1]/scale_factor)
#					print type(x), type(x0)
					
					cv2.circle(frame,(x0+x, y0+y-10), int(circles[0][i][2]/scale_factor), (255,0,0), 3, cv2.CV_AA)
					cv2.circle(frame,(x0+x, y0+y-10), 2, (255,100,0), 3, cv2.CV_AA)

			return crop

#while(cap.isOpened()):
while(True):
   	ret, frame = cap.read()
	contours,frame,img = detect_wrench(frame)
	crop = detect_size(contours,frame,img)
	cv2.imshow("size",crop)
	cv2.imshow("Detection", frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
#cv2.waitKey(0)

