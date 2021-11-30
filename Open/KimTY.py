import cv2
from matplotlib import pyplot as plt
import numpy as np

original = cv2.imread('./Open/t03/1.jpg')
gray_im = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
plt.subplot(221)
plt.title('Grayscale image')
plt.imshow(gray_im, cmap="gray", vmin=0, vmax=255)

# Contrast adjusting with gamma correction y = 1.2
def fit_rotated_ellipse(data):

	xs = data[:,0].reshape(-1,1)
	ys = data[:,1].reshape(-1,1)

	J = np.mat( np.hstack((xs*ys,ys**2,xs, ys, np.ones_like(xs,dtype=float))) )
	Y = np.mat(-1*xs**2)
	P= (J.T * J).I * J.T * Y

	a = 1.0; b= P[0,0]; c= P[1,0]; d = P[2,0]; e= P[3,0]; f=P[4,0];
        # To do implementation
        #a,b,c,d,e,f 를 통해 theta, 중심(cx,cy) , 장축(major), 단축(minor) 등을 뽑아 낼 수 있어요

	theta = 0.5* np.arctan(b/(a-c))
	cx = (2*c*d - b*e)/(b**2-4*a*c)
	cy = (2*a*e - b*d)/(b**2-4*a*c)
	cu = a*cx**2 + b*cx*cy + c*cy**2 -f
	w= np.sqrt(cu/(a*np.cos(theta)**2 + b* np.cos(theta)*np.sin(theta) + c*np.sin(theta)**2))
	h= np.sqrt(cu/(a*np.sin(theta)**2 - b* np.cos(theta)*np.sin(theta) + c*np.cos(theta)**2))

	return (cx,cy,w,h,theta)


gray_correct = np.array(255 * (gray_im / 255) ** 3.5 , dtype='uint8')
plt.subplot(222)
plt.title('Gamma Correction y= 1.2')
plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)
# Contrast adjusting with histogramm equalization
gray_equ = cv2.equalizeHist(gray_im)
plt.subplot(223)
plt.title('Histogram equilization')
plt.imshow(gray_correct, cmap="gray", vmin=0, vmax=255)

thresh = cv2.adaptiveThreshold(gray_correct, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 255, 19)

#thresh = cv2.bitwise_not(thresh)plt.subplot(221)
plt.title('Local adapatative Threshold')

plt.imshow(thresh, cmap="gray", vmin=0, vmax=255)
kernel = np.ones((11,11), np.uint8)
img_dilation = cv2.dilate(thresh, kernel, iterations=1)
img_erode = cv2.erode(img_dilation,kernel, iterations=1)
# clean all noise after dilatation and erosion
img_erode = cv2.medianBlur(img_erode, 39)

plt.subplot(221)
plt.title('Dilatation + erosion')
plt.imshow(img_erode, cmap="gray", vmin=0, vmax=255)
ret, labels = cv2.connectedComponents(img_erode)
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
labeled_img[label_hue == 0] = 0

plt.subplot(222)
plt.title('Objects counted:'+ str(ret-1))
plt.imshow(labeled_img)
print('objects number is:', ret-1)


plt.show()

gray = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)


retv, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
_,contours,_ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for con in contours:
	approx = cv2.approxPolyDP(con, 0.01 * cv2.arcLength(con, True), True)
	area = cv2.contourArea(con)
	if (len(approx) > 10 and area > 30000):
		# cv2.drawContours(src,[con],0,color_list[2],2)
		cx, cy, w, h, theta = fit_rotated_ellipse(con.reshape(-1, 2))
		cv2.ellipse(original, (int(cx), int(cy)), (int(w), int(h)), theta * 180.0 / np.pi, 0.0, 360.0, (238, 0, 0), 2)

cv2.imwrite('out.jpg', original)