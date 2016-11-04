#!/usr/bin/python

import cv2
#import cv2 as cv2.cv
import numpy as np
import sys

#cap = cv2.VideoCapture('/home/urvi/Desktop/MBZIRC/Image Processing/video.mp4')
#cap = cv2.VideoCapture('/home/urvi/Desktop/MBZIRC/Image Processing/sample.mkv')
#cap = cv2.VideoCapture(1)

#while(cap.isOpened()):
#while(True):
#ret, frame = cap.read()
#img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
scale_factor = 3
img = cv2.imread('/home/urvi/Desktop/MBZIRC/Image Processing/snap3.png',0)
ret,thresh = cv2.threshold(img,127,255,0)
contours,hierarchy = cv2.findContours(thresh, 1, 2)

a = []

for cnt in contours:

#		epsilon = 0.1*cv2.arcLength(cnt,True)
#		approx = cv2.approxPolyDP(cnt,epsilon,True)

#		if len(approx) >= 5:
			
#	cnt = contours[0]
	x,y,w,h = cv2.boundingRect(cnt)
	if h > 170 and h < 400:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
#			cv2.rectangle(frame,(x,y),(x+w,y+h/5),(0,0,255),2)
		crop = img[y-20:y+h/5,x:x+w]
                crop = cv2.resize(crop,(0,0),fx=scale_factor,fy=scale_factor)
		crop1 = cv2.medianBlur(crop,5)
		circles = cv2.HoughCircles(crop1, cv2.cv.CV_HOUGH_GRADIENT,1,10,np.array([]),100,30,0,0)
		if circles != None:
			a,b,c = circles.shape
			for i in range(b):
				cv2.circle(crop,(circles[0][i][0], circles[0][i][1]), circles[0][i][2], (255,0,0), 1, cv2.CV_AA)
				cv2.circle(crop,(circles[0][i][0], circles[0][i][1]), 2, (255,100,0), 1, cv2.CV_AA)
				x0 = int(circles[0][i][0]/scale_factor)
				y0 = int(circles[0][i][1]/scale_factor)
#					print type(x), type(x0)
				print circles[0][i][2]
				cv2.circle(img,(x0+x, y0+y-20), int(circles[0][i][2]/scale_factor), (255,0,0), 2, cv2.CV_AA)
				cv2.circle(img,(x0+x, y0+y-20), 2, (255,100,0), 1, cv2.CV_AA)
                       
#			print np.shape(circles)

cv2.imshow("size",crop)
cv2.imshow("Detection", img)
#	if cv2.waitKey(1) & 0xFF == ord('q'):
#		break

#cap.release()
#cv2.destroyAllWindows()
cv2.waitKey(0)

