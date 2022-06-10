"""
Usage: This script will measure different objects in the frame using a reference object of known dimension.
The object with known dimension must be the leftmost object.
"""
from scipy.spatial.distance import euclidean
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from PIL import Image
import io
import base64
import time

timestr = time.strftime("%Y-%m-%d")

# Read image and preprocess

def measure_distance(our_image):
	image = np.array(our_image.convert('RGB'))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (9, 9), 0)

	edged = cv2.Canny(blur, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	# Find contours
	cont = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cont = imutils.grab_contours(cont)

	# Sort contours from left to right as leftmost contour is reference object
	(sort_cont, _) = contours.sort_contours(cont)

	# Remove contours which are not large enough
	sig_contours = [x for x in sort_cont if cv2.contourArea(x) > 100]

	cv2.drawContours(image, sig_contours, -1, (0,255,0), 3)

	# Reference object dimensions
	# Here for reference we have used a 2cm x 2cm square
	ref_object = sig_contours[0]
	box = cv2.minAreaRect(ref_object)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	box = perspective.order_points(box)
	(tl, tr, br, bl) = box
	dist_in_pixel = euclidean(tl, tr)
	dist_in_cm = 2
	pixel_per_cm = dist_in_pixel / dist_in_cm

	# Draw remaining contours
	for contr in sig_contours:
		box = cv2.minAreaRect(contr)
		box = cv2.boxPoints(box)
		box = np.array(box, dtype="int")
		box = perspective.order_points(box)
		(tl, tr, br, bl) = box
		cv2.drawContours(image, [box.astype("int")], -1, (0, 0, 255), 2)
		mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0]) / 2), tl[1] + int(abs(tr[1] - tl[1]) / 2))
		mid_pt_vertical = (tr[0] + int(abs(tr[0] - br[0]) / 2), tr[1] + int(abs(tr[1] - br[1]) / 2))
		wid = euclidean(tl, tr) / pixel_per_cm
		ht = euclidean(tr, br) / pixel_per_cm
		cv2.putText(image, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
		cv2.putText(image, "{:.1f}cm".format(ht), (int(mid_pt_vertical[0] + 10), int(mid_pt_vertical[1])),
					cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

	return image


def measure_area(our_image):
	image = np.array(our_image.convert('RGB'))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray, (9, 9), 0)
	edged = cv2.Canny(blur, 50, 100)
	edged = cv2.dilate(edged, None, iterations=1)
	edged = cv2.erode(edged, None, iterations=1)

	# Find contours
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	cnts = imutils.grab_contours(cnts)

	# Sort contours from left to right as leftmost contour is reference object
	(cnts, _) = imutils.contours.sort_contours(cnts)

	# Remove contours which are not large enough
	contours = [x for x in cnts if cv2.contourArea(x) > 100]
	cv2.drawContours(image, contours, -1, (0, 255, 0), 3)

	ref_object = cnts[0]
	box = cv2.minAreaRect(ref_object)
	box = cv2.boxPoints(box)
	box = np.array(box, dtype="int")
	box = perspective.order_points(box)
	(tl, tr, br, bl) = box
	dist_in_pixel = euclidean(tl, tr)
	dist_in_cm = 2
	a = dist_in_cm / dist_in_pixel

	for contour in contours[1:]:
		approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
		cv2.drawContours(image, [approx], -1, (0, 0, 255), 3)

		# cv2.drawContours(img, [approx], 0, (0, 0, 0), 5)
		x = approx.ravel()[0]
		y = approx.ravel()[1] - 5
		print(len(approx))
		if len(approx) == 3:
			cv2.putText(image, "Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))


		elif len(approx) == 4:
			x1, y1, w, h = cv2.boundingRect(approx)
			aspectRatio = float(w) / h
			# print(aspectRatio)
			if aspectRatio >= 0.95 and aspectRatio <= 1.05:
				area = w * h * a * a
				cv2.putText(image, "Square: {:.1f}cm2".format(area), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
			else:
				area = w * h * a * a
				cv2.putText(image, "Rectangle: {:.1f}cm2".format(area), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5,
							(0, 0, 0))

		elif len(approx) >= 15:
			x1, y1, w, h = cv2.boundingRect(approx)
			area = 3.14 * (w + h) * (w + h) * a * a / 4
			cv2.putText(image, "Circle: {:.1f}cm2".format(area), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0))
	return image


class VideoTransformer(VideoTransformerBase):
	def transform(self, frame):
		img = frame.to_ndarray(format="bgr24")
		result_img = measure_distance(img)
		#img = cv2.cvtColor(cv2.Canny(img, 100, 200), cv2.COLOR_GRAY2BGR)
		return result_img


def get_image_download_link(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = io.BytesIO()
	img.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	download_filename = "{}_{}.{}".format("measurement", timestr, "jpg")
	href = f'<a href="data:file/jpg;base64,{img_str}" download="{download_filename}"><input type="button" value="Download"></a>'
	return href

def main():
	st.image("logo_design/logo2.png", width=100)
	st.title("Object Measurement Web App \U0001F4D0")

	image_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
	if image_file is not None:
		our_image = Image.open(image_file)
		st.text("Original Image")
		st.image(our_image)

		if st.button("Measure Distance"):
			result_img = measure_distance(our_image)
			st.image(result_img)
			result = Image.fromarray(result_img)
			st.markdown(get_image_download_link(result), unsafe_allow_html=True)

		if st.button("Measure Area"):
			result_img = measure_area(our_image)
			st.image(result_img)
			result = Image.fromarray(result_img)
			st.markdown(get_image_download_link(result), unsafe_allow_html=True)

		st.text("Or Provide input from Webcam")
		webrtc_streamer(key="example", video_processor_factory=VideoTransformer)


if __name__ == '__main__':
	main()
