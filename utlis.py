import numpy as np
import cv2
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

img_path = "images/example_05.jpg"
image = cv2.imread(img_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)
cv2.imshow("blur",blur)
edged = cv2.Canny(blur, 50, 100)
cv2.imshow("edged",edged)
edged = cv2.dilate(edged, None, iterations=1)
cv2.imshow("edged2",edged)
edged = cv2.erode(edged, None, iterations=1)
cv2.imshow("edged3",edged)
# Find contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# Sort contours from left to right as leftmost contour is reference object
(cnts, _) = imutils.contours.sort_contours(cnts)

# Remove contours which are not large enough
contours = [x for x in cnts if cv2.contourArea(x) > 100]
cv2.drawContours(image, contours, -1, (0,255,0), 3)

ref_object = cnts[0]
box = cv2.minAreaRect(ref_object)
box = cv2.boxPoints(box)
box = np.array(box, dtype="int")
box = perspective.order_points(box)
(tl, tr, br, bl) = box
dist_in_pixel = euclidean(tl, tr)
dist_in_cm = 2
a = dist_in_cm/dist_in_pixel

for contour in contours[1:]:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(image, [approx], -1, (0, 0, 255), 3)

    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    print(len(approx))
    if len(approx) == 3:
        cv2.putText(image, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))


    elif len(approx) == 4:
        x1 ,y1, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        #print(aspectRatio)
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            area = w * h * a * a
            cv2.putText(image, "Square: {:.1f}cm2".format(area), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
        else:
            area=w*h*a*a
            cv2.putText(image, "Rectangle: {:.1f}cm2".format(area), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

    elif len(approx)>=13 and len(approx)<19:
        x1, y1, w, h = cv2.boundingRect(approx)
        area = 3.14*(w+h)*(w+h)*a*a/4
        cv2.putText(image, "Circle: {:.1f}cm2".format(area), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))

cv2.imshow("shapes", image)
cv2.waitKey(0)
cv2.destroyAllWindows()