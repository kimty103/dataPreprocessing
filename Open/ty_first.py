import cv2, sys
from matplotlib import pyplot as plt
import numpy as np


#   이미지 블러처리 외곽선 검출
#   참고 https://youbidan.tistory.com/19
# image = cv2.imread('Open/t02/2.jpg')
#
# image_gray = cv2.imread('Open/t02/2.jpg', cv2.IMREAD_GRAYSCALE)
# blur = cv2.GaussianBlur(image_gray, ksize=(31,31), sigmaX=0)
# ret, thresh1 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
# edged = cv2.Canny(blur, 10,80)
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
# closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
# contours, _ = cv2.findContours(closed.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# total = 0
# contours_image = cv2.drawContours(image, contours, -1, (0,255,0), 3)
#
# plt.imshow(blur)
#
# plt.show()
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




fsrc = cv2.imread('./Open/t08/4.jpg')

resized = cv2.resize(fsrc, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
x, y, r = 515, 376, 350

rectX = (x - r)
rectY = (y - r)
hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
white_out = cv2.inRange(hsv, (100, 0, 200), (140, 100, 255))

plt.imshow(white_out)
#
plt.show()

# cv2.imwrite('./result3/{}{}{}.png'.format(200, 200, 200), white_out)
# print(type(white_out))
# circles = cv2.HoughCircles(white_out, cv2.HOUGH_GRADIENT, 1, 700,
#                            param1=50, param2=30, minRadius=300, maxRadius=400)
# circles = np.uint16(np.around(circles))
#
# if len(circles) == 1:
#     x, y, r = circles[0][0]
#     #         print x, y, r
#     mask = np.zeros((resized.shape[0], resized.shape[1]), dtype=np.uint8)
#     cv2.circle(mask, (x, y), r, (255, 255, 255), -1, 8, 0)
#     #         cv2.imwrite(argv[2],mask)
#     out = resized * mask
#     white = 255 - mask
#
#
# cv2.imshow("white_out", white_out)
# cv2.waitKey()
# cv2.destroyAllWindows()

color_list = [(238, 0, 0), (0, 252, 124), (142, 56, 142)]

src = cv2.imread('./Open/t01/4.jpg', cv2.IMREAD_COLOR)

gray = cv2.cvtColor(white_out, cv2.COLOR_GRAY2RGB)

gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)

cv2.imwrite('ab.jpg', white_out)
retv, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for con in contours:
	approx = cv2.approxPolyDP(con, 0.01 * cv2.arcLength(con, True), True)
	area = cv2.contourArea(con)
	if (len(approx) > 10 and area > 30000):
		# cv2.drawContours(src,[con],0,color_list[2],2)
		cx, cy, w, h, theta = fit_rotated_ellipse(con.reshape(-1, 2))
		cv2.ellipse(src, (int(cx), int(cy)), (int(w), int(h)), theta * 180.0 / np.pi, 0.0, 360.0, color_list[0], 2)

cv2.imwrite('out.jpg', src)