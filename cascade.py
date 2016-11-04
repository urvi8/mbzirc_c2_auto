#opens up a webcam feed so you can then test your classifer in real time
#using detectMultiScale
import numpy
import cv2

fps = 20
fourcc = cv2.cv.CV_FOURCC(*'XVID')
vout = cv2.VideoWriter()
cap = cv2.VideoCapture(0)
size = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))

#success = vout.open('output.avi',fourcc,fps,size,True)

def detect(img):
    #cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml')
    cascade = cv2.CascadeClassifier('/home/jonathan/wrench.xml')
    rects = cascade.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (35,200))
    print 'I am trying!'
    if len(rects) == 0: 
        print 'Did not find any!'
        return [], img

    rects[:, 2:] += rects[:, :2]
    return rects, img

def box(rects, img):
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(img, (x1, y1), (x2, y2), (127, 255, 0), 2)


#cap = cv2.VideoCapture(0)
#cap.set(3,800)
#cap.set(4,600)

i = 0;

while(True):
    ret, img = cap.read()
    #cv2.imwrite('wrench'+str(i)+'.jpg', img);
    i=i+1;
    rects, img = detect(img)
    box(rects, img)
    cv2.imshow("frame", img)
    vout.write(img)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
	break

cap.release()
vout.release()
cv2.destroyAllWindows()

