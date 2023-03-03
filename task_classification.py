import cv2
import numpy as np

def task_classification(img):
	#Load image
	# img_num = 4
	# img_type = 'jpg'
	# img = cv2.imread('./data/inp' + str(img_num) + '.' + img_type)

	#Convert image to grayscale
	# img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	def auto_canny(image, sigma=0.33):
		#compute the median of the single channel pixel intensities
		v = np.median(image)
		#apply automatic Canny edge detection using the computed median
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		edged = cv2.Canny(image, lower, upper)
		#return the edged image
		return edged

	#Apply edge detection
	edges = auto_canny(gray)

	#Apply Hough Transform to detect lines
	# lines = cv2.HoughLines(edges, 1, np.pi/180,150)

	rho = 1  # distance resolution in pixels of the Hough grid
	theta = np.pi / 180  # angular resolution in radians of the Hough grid
	threshold = 150  # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 400  # minimum number of pixels making up a line
	max_line_gap = 20  # maximum gap in pixels between connectable line segments
	line_image = np.copy(img) * 0  # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	hlines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
						min_line_length, max_line_gap)

	if (hlines is None):
		return img

	minY = 100000
	maxY = 0
	imgH = img.shape[0]

	final_hlines = []
	for line in hlines:
		y_mean =  int(np.mean([line[0][1], line[0][3]]))
		if y_mean > imgH*0.1:
			minY = min(minY, y_mean)
		if y_mean < 0.9*imgH:
			maxY = max(maxY, y_mean)
		if (y_mean > imgH*0.1) and (y_mean < imgH*0.9):
			(x1,y1,x2,y2) = line[0]
			slope = abs(np.arctan((y2-y1)/(x2-x1)))		
			if slope<=(np.pi/6):
				final_hlines.append(line)
				# cv2.line(img,(x1,y1),(x2,y2),(0,255,0),5)

	if (len(final_hlines) == 0):
		return img

	#Cropping
	img = img[minY:maxY,:]
	edges = edges[minY:maxY,:]

	# Detecting vertical lines in the cropped image
	rho = 1  # distance resolution in pixels of the Hough grid
	heta = np.pi / 180  # angular resolution in radians of the Hough grid
	threshold = 10  # minimum number of votes (intersections in Hough grid cell)
	min_line_length = 50  # minimum number of pixels making up a line
	max_line_gap = 20  # maximum gap in pixels between connectable line segments
	line_image = np.copy(img) * 0  # creating a blank to draw lines on

	# Run Hough on edge detected image
	# Output "lines" is an array containing endpoints of detected line segments
	vlines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
						min_line_length, max_line_gap)

	final_vlines = []
	imgW = img.shape[1]
	for line in vlines:
		for x1,y1,x2,y2 in line:
			x_mean = int(np.mean([x1, x2]))
			if (x_mean < imgW*0.05) or (x_mean > 0.95*imgW):
				continue
			
			slope = abs(np.arctan((y2-y1)/(x2-x1)))		
			if slope>=(np.pi/3) and slope<=(2*np.pi/3):
				final_vlines.append(line)
				# cv2.line(img,(x1,y1),(x2,y2),(255,0,0),5)

	for line in final_vlines:
		(x1,y1,x2,y2) = line[0]
		x_mean = int(np.mean([x1, x2]))
		img[:, x_mean-3: x_mean+3] = 255

	for line in final_hlines:
		(x1,y1,x2,y2) = line[0]
		y_mean = int(np.mean([y1, y2])) - minY
		if y_mean < 0 or y_mean > maxY-minY:
			continue
		img[y_mean-3: y_mean+3, :] = 255

	return img