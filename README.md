# GIA Docs - Intern 
Table Detection and Table Structure Recognition in document images

This repository contains a Python implementation of table detection on an input image using OpenCV and Pytesseract. The code detects the table structure and ouptuts the table type as well as the contents of the table in Json format.

## Requirements

    Python 3.x
    OpenCV
    NumPy
    pytesseract

## Usage

To use this code, follow these steps:

    Clone this repository to your local machine.
    Install the dependencies listed in the requirements.txt file using pip: pip install -r requirements.txt
    Place your input image in the same directory as the text_detection.py file.
    Run the script: python text_detection.py, change the input image in the main function of this code to get the output.

## Input
    The text_detection.py takes the different types of images from ./Data/input_images as input

## Output
    In the ./Data folder, I have added images to show the preprocessed images, bouding boxes which are given as input to the pytesseract library to find the text, the input being sent to the text_detection through the pipeline.

    Input: Type_1
    Output: Type 1
    {'QTY': ['1', '2', '3'], 'DESCRIPTION': ['Front and rear brake cables', 'New set of pedal arms', 'Labor Shrs'], 'UNIT PRICE': ['100.00', '15.00', '§.00'], 'AMOUNT.': ['100.00', '30.00', '15.00']}

    Input: Type_2
    Output: Type 2
    {'DESCRIPTION': ['Service Fee', 'Labor: 5 hours at $75-hr', 'Nee client discount', 'Tax (4.25% after discount'], 'AMOUNT': ['299:33', '37539', '(53.93', '26']}

    Input: Type_3
    Output: Type 3
    {'Q': ['1', '2', '3'], 'DESCRIPTION': ['Front and rear brake cables', 'New set of pedal arms.', 'Labor 3hrs'], 'UNIT PRIGE': ['100.00', '15.00', '§.00'], 'AMOUNT': ['100.00', '30.00', '15.00']}

    Input: Type_4
    Output: Type 4
    {'ITEMS': ['ITEM 1', 'JTEM 2', 'ITEM 3', 'ITEM 4', 'ITEM 5', 'ITEM 6'], 'DESCRIPTION': ['Description', 'Description', 'Description', 'Description', 'Description', 'Description'], 'QUANTITY': ['1', '1', '1', '1', '1', '1'], 'PRICE': ['$000.00', '$000.00', '$000.00', '$000.00', '$000.00', '$000.00'], 'TAX': ['1%', '1%', '1%', '1%', '1%', '1%'], 'AMOUNT': ['$000.00', '$000.00', '$000.00', '$000.00', '$000.00', '$000.00']}

As we can see from the bouding boxes, for all types of images the bouding boxes which I have drew for this algorithm are correct but the output is not a 100% accurate due to unclear images and limitations of pytesseract library, We can try to improve the image quality by using image processing techniques or by using other ocr libraries to see if there is an improvement in performance. 

Bouding_boxes example on Type_1 image.

<!-- ![alt text](http://url/to/img.png) -->

## task_classification.py 

The code performs table detection on an input image using computer vision techniques. The process can be divided into the following steps:

    1) Edge detection: 
    First, the image is converted to grayscale using the cv2.cvtColor() function. Then, an edge detection algorithm called "auto Canny" is applied to the grayscale image using the auto_canny() function. This function computes the median of the single channel pixel intensities in the image, and then applies the Canny edge detection algorithm using the computed median as the lower and upper thresholds.
    
    2) Horizontal line detection: 
    The Hough Transform algorithm is applied to the edge-detected image using the line_detection() function, which returns an array containing the endpoints of detected line segments. The algorithm is set to detect lines with a certain threshold and minimum line length. The resulting lines are then filtered to only keep lines that are approximately horizontal, based on the slope of the line segment.
    
    3) Cropping:
    The image is cropped to remove any content that is not part of the detected table. This is done by selecting a horizontal region of interest (ROI) based on the minimum and maximum Y values of the detected horizontal lines.
    
    4) Vertical line detection:
    The Hough Transform algorithm is applied again to the edge-detected and cropped image, but this time to detect vertical lines using the line_detection() function. Similar to the horizontal lines, the detected vertical lines are filtered to keep only those that are approximately vertical.
    
    5) Combining nearby lines: 
    The detected horizontal and vertical lines are combined to form a grid-like structure that represents the table. To do this, the function combine_nearby_vlines() is used to group together nearby vertical lines that are within a certain threshold distance. A similar function combine_nearby_hlines() is used to group nearby horizontal lines. This grouping is performed to eliminate overlapping and redundant lines, and to form a single grid structure that represents the table.
    
    6) Output: 
    The final output is an image where the detected table is outlined with a grid-like structure, as well as the horizontal and vertical line segments that were used to form the structure. The function task_classification() returns this final image, along with the minimum and maximum Y values of the detected horizontal lines, which can be used to locate the table within the original image.
    
## text_detection.py

This code extracts tables from images and outputs the table type and gives the text available in a json format. It makes use of OpenCV for image processing and Tesseract OCR for text recognition.

The process involves the following steps:

    1) Reading the image:
    The first step is to read in the image file that contains the table. The image can be in various formats such as PNG, JPEG, TIFF, etc.

    2) Pre-processing the image to convert it to grayscale and enhance text visibility:
    The next step is to preprocess the image to enhance the visibility of the text. This is done by converting the image to grayscale and applying various filters such as Gaussian blur, thresholding, and morphological operations to remove noise and make the text more prominent.

    3) Identifying the text regions in the image using contours and bounding boxes:
    Once the image is preprocessed, the next step is to identify the regions in the image that contain text. This is done by finding the contours of the image and drawing bounding boxes around each contour. A contour is simply a curve joining all the continuous points (along the boundary), having the same color or intensity. Bounding boxes are rectangles that encompass each contour.

    4) Filtering the text regions based on a minimum and maximum height limit:
    The next step is to filter out the text regions that are not part of the table. This is done by setting a minimum and maximum height limit for the text regions. Any regions that fall outside this height range are discarded.

    5) Clustering the text regions to identify the rows and columns of the table:
    The text regions that pass the height filter are then clustered together to identify the rows and columns of the table. This is done by computing the distances between the bounding boxes and grouping them based on their proximity.

    6) Building horizontal and vertical lines to separate the cells of the table:
    Once the rows and columns of the table have been identified, the next step is to build horizontal and vertical lines to separate the cells of the table. This is done by computing the average height and width of the bounding boxes and drawing lines at regular intervals.

    7) Extracting the text content of each cell using Tesseract OCR:
    Finally, the text content of each cell is extracted using Tesseract OCR (Optical Character Recognition). OCR is a technology that can recognize text from images and convert it into machine-readable text. The OCR engine takes each cell image as input and outputs the recognized text. The recognized text is then stored in a matrix or spreadsheet format.
    
    
## Future Possibilites:

Even with numerous rules written for detecting tables in document images, there will always be variations in layouts and types of tables that can cause the rules to fail. To overcome this challenge, machine learning (ML) and deep learning (DL) solutions are utilized. There are various pre-implemented codes available to address the problem of table detection using ML or DL techniques.
Some of the machine leanring which I feel we can include to improve the accuracy of text detection in tables are:

TableNet: Deep Learning model for end-to-end Table detection and Tabular data extraction from Scanned Document Images 
https://arxiv.org/abs/2001.01469

CascadeTabNet: An approach for end to end table detection and structure recognition from image-based documents
https://arxiv.org/abs/2004.12629

TableNet proposes a convolutional neural network (CNN) architecture that is trained on a large-scale dataset of document images with annotated tables. The model is designed to be an end-to-end solution for table detection and tabular data extraction, which means it can handle both the task of detecting tables in the image as well as extracting the data from the table cells. The authors report high accuracy on several benchmark datasets, indicating the potential for this model to be a reliable solution for table detection.

CascadeTabNet takes a similar approach to TableNet, but with a few key differences. Instead of using a single CNN to detect tables and extract data, CascadeTabNet employs a cascade of neural networks that work together to identify the table structure and contents. The model first detects the table regions in the image, then extracts the table structure and finally extracts the data from the table cells. This hierarchical approach allows CascadeTabNet to handle complex table structures with high accuracy.

Both TableNet and CascadeTabNet offer promising results for improving table detection in document images. These models can be utilized to create more accurate and efficient solutions for tasks such as data extraction and document analysis. By leveraging the power of deep learning, these approaches can overcome the limitations of rule-based methods and adapt to the wide variety of table layouts and structures found in real-world document images.
