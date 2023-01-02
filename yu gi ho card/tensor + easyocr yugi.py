import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

img=cv2.imread(r"C:\Users\9820937G\OneDrive - SNCF\Bureau\yugi.png")
plt.imshow(img)
plt.show()

bfilter=cv2.bilateralFilter(img,11,17,17)
edged=cv2.Canny(bfilter,30,200)
plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))

keypoints=cv2.findContours(edged.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
countours=imutils.grab_contours(keypoints)
countours=sorted(countours,key=cv2.contourArea,reverse=True)[:10]

location = None
for countours in countours:
    approx=cv2.approxPolyDP(countours,10,True)
    if len(approx) ==4:
        location=approx
        break
print(location)

mask=np.zeros(img.shape[0:2],dtype='uint8')
img2=cv2.drawContours(mask, [location], 0,255,-1)

reader=easyocr.Reader(['fr'])

result=reader.readtext(img)
            