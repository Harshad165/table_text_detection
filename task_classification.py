import cv2
import numpy as np

def line_detection(edges, threshold, min_line_length):
		rho = 1  # distance resolution in pixels of the Hough grid
		theta = np.pi / 180  # angular resolution in radians of the Hough grid
		max_line_gap = 20  # maximum gap in pixels between connectable line segments

		# Run Hough on edge detected image
		# Output "lines" is an array containing endpoints of detected line segments
		return cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
							min_line_length, max_line_gap)
def auto_canny(image, sigma=0.33):
			#compute the median of the single channel pixel intensities
			v = np.median(image)
			#apply automatic Canny edge detection using the computed median
			lower = int(max(0, (1.0 - sigma) * v))
			upper = int(min(255, (1.0 + sigma) * v))
			edged = cv2.Canny(image, lower, upper)
			#return the edged image
			return edged

def filter_hlines(hlines, imgH):
		minY = 100000
		maxY = 0

		final_hlines = []
		for line in hlines:
			y_mean =  int(np.mean([line[0][1], line[0][3]]))
			if y_mean > imgH*0.1:
				minY = min(minY, y_mean)
			if y_mean < 0.9*imgH:
				maxY = max(maxY, y_mean)
			if (y_mean > imgH*0.1) and (y_mean < imgH*0.9):
				(x1,y1,x2,y2) = line[0]
				np.seterr(divide='ignore')
				slope = abs(np.arctan((y2-y1)/(x2-x1)))		
				if slope<=(np.pi/6):
					final_hlines.append(line)
					# cv2.line(img,(x1,y1),(x2,y2),(0,255,0),5)
		return (final_hlines, minY, maxY)

def filter_vlines(vlines, imgW):
		final_vlines = []
		for line in vlines:
			for x1,y1,x2,y2 in line:
				x_mean = int(np.mean([x1, x2]))
				if (x_mean < imgW*0.05) or (x_mean > 0.95*imgW):
					continue
				np.seterr(divide='ignore')
				slope = abs(np.arctan((y2-y1)/(x2-x1)))		
				if slope>=(np.pi/3) and slope<=(2*np.pi/3):
					final_vlines.append(line)
					# cv2.line(img,(x1,y1),(x2,y2),(255,0,0),5)
		return final_vlines

def combine_nearby_vlines(final_vlines, img):
		final_vlines = sorted(final_vlines, key=lambda line: line[0][0])
		prev_x = 0
		output_vlines = []
		for line in final_vlines:
			(x1,y1,x2,y2) = line[0]
			x_mean = int(np.mean([x1, x2]))
			if abs(x_mean - prev_x) <= 10:
				continue
			output_vlines.append(line)
			img[:, x_mean-3: x_mean+3] = 255
			prev_x = x_mean
		return (output_vlines, img)

def combine_nearby_hlines(final_hlines, img, minY, maxY):
		final_hlines = sorted(final_hlines, key=lambda line: line[0][1])
		prev_y = -100
		output_hlines = []
		for line in final_hlines:
			(x1,y1,x2,y2) = line[0]
			y_mean = int(np.mean([y1, y2])) - minY
			if y_mean < 0 or y_mean > maxY-minY:
				continue
			if abs(y_mean - prev_y) <= 3:
				continue
			output_hlines.append([[x1,y1-minY,x2,y2-minY]])
			img[y_mean-3: y_mean+3, :] = 255
			prev_y = y_mean
		return (output_hlines, img)

def task_classification(img):
		#Convert image to grayscale
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

		#Apply edge detection
		edges = auto_canny(gray)

		#Apply Hough Transform to detect lines
		hlines = line_detection(edges, 150, 400)
		if (hlines is None):
			return (img, 0, 0)

		(final_hlines, minY, maxY) = filter_hlines(hlines, img.shape[0])
		if (len(final_hlines) == 0):
			return (img, 0, 0)

		#Cropping
		img = img[minY:maxY,:]
		edges = edges[minY:maxY,:]

		# Detecting vertical lines in the cropped image
		vlines = line_detection(edges, 10, 50)
		final_vlines = filter_vlines(vlines, img.shape[1])

		# Combine nearby vertical and horizontal lines
		(output_vlines, img) = combine_nearby_vlines(final_vlines, img)
		(output_hlines, img) = combine_nearby_hlines(final_hlines, img, minY, maxY)

		return (img, len(output_hlines), len(output_vlines))