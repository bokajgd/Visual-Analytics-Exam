# A3 - Edge Detection

# Overview 

**Jakob Grøhn Damgaard, May 2021** <br/>
This folder contains  assigmnent 3 for the course *Visual Analytics*

# Description
Edge detection refers to the notion of identifying boundary points in an image where the light or colour intensity changes drastically (Davis, 1975). For this assignment, we aim to utilize edge detection algorithms to perform object segmentation/detection on an image. The end goal, thus, is to identify and draw contours around perceptually salient segments of an image. Once all separate objects in an image have been identified, they can e.g., be classified using an object recognition algorithm. 
<br> <br>
More specifically, we were provided with a large image of a piece of text engraved onto the Jefferson memorial. The image can be found by clicking this link: <br> https://upload.wikimedia.org/wikipedia/commons/f/f4/%22We_Hold_These_Truths%22_at_Jefferson_Memorial_IMG_4729.JPG <br>
 The purpose is to detect and draw contours around each specific letter and language-like object (e.g. punctuation) in the image. The tasks were as follows (taken directly from assignment description): <br>
•	Draw a green rectangular box to show a region of interest (ROI) around the main body of text in the middle of the image. Save this as **image_with_ROI.jpg**.
•	Crop the original image to create a new image containing only the ROI in the rectangle. Save this as **image_cropped.jpg**.
•	Using this cropped image, use Canny edge detection to 'find' every letter in the image
•	Draw a green contour around each letter in the cropped image. Save this as **image_letters.jpg**

# Usage
See *General Instruction* in the home folder of the repository for instruction on how to clone the repo locally.
<br>
If not already open, open a terminal window and redirect to the home folder of the cloned repository (see General Instruction). Remember to activate the virtual environment. Then, jump into the folder called A3-Edge-Detection using the following command:
```bash
cd A3-Edge-Detection
```

Now, it should be possible to run the following command in order to get an understanding of how the script is executed and which arguments should be provided:
```bash
# Add -h to view how which arguments should be passed  
python3 src/A3-Edge-Detection.py -h                  
usage: A3-Edge-Detection.py [-h] [-inp --input]

[INFO] Image similarity using color histograms

optional arguments:
  -h, --help    show this help message and exit
  -inp --input  [DESCRIPTION] Name of the file of the input image 
                [TYPE]        str 
                [DEFAULT]     text_image.jpeg 
                [EXAMPLE]     -inp text_image.jpeg
```
<br>
It should now be clear that the script can be executed using the following command. 
```bash

# With input image specified 
python3 src/A3-Edge-Detection.py -inp text_image.jpeg

# Without specification of input image
python3 src/A3-Edge-Detection.py
Input image file name is not specified.
Setting it to 'text_image.jpeg'.

```

As can be seen, if there is no input to the *-inp* argument, *text_image.jpeg* is used as the default input image. It is possible to input a self-chosen image (takes any file type supported by *cv2.open()* function) that has been placed in the data folder. However, as the script is already specialised to the specific image provided for the assignment results (pre-defined ROI coordinates), it will likely produce futile results.

## Structure
The structure of the assignment folder can be viewed using the following command:

```bash
tree -L 2
```

This should yield the following graph:

```bash
.
├── README.md
├── data
│   └── text_image.jpeg
├── output
│   ├── image_with_ROI.png
│   ├── image_with_contours.png
│   └── only_ROI.png
└── src
    └── A3-Edge-Detection.py
```

The following table explains the directory structure in more detail:
| Column | Description|
|--------|:-----------|
```data```| A folder containing the data used for the analysis. In this folder, the image provided for analysis, *text_image.jpeg* is located.
```src``` | A folder containing the .py script (*A3-Edge-Detection.py*) created to solve the assignment.
```output``` | A folder containing the output produced by the Python script. The script generates three images:
-	*image_with_ROI.png*: An image with a green border showing the pre-defined ROI
-	*image_letters.png*: An image with green contours drawn around all letters detected in the image
-	*image_cropped.png*: An image of only the ROI cropped out of the original image


# Methods
Akin to the script in assignment 2, the script is coded using the principles of object-oriented programming. See the first paragraph of the previous methods section for a quick outline of the general script architecture.<br>
<br>
All image processing in the script is performed using OpenCV (Bradski, 2000). After loading in the input image, a version with a green border drawn around the engravings is generated and saved in the output folder. This area is then cropped out of the original image which is also saved. In order to draw contours around the letters, the cropped image is converted into a greyscale color space. A Gaussian blur filter with a kernel size of 7x7 is then applied in order reduce noise in the image. If too much smoothing is induced, the desired edges will become undetectable. Next, the image is transformed using simple binary thresholding with a static threshold set at 115 (yields best result). This converts any pixels with a brightness value below 115 to 0 and pixels with a brightness value above 115 to 255. Lastly, canny edge detection is applied with a high threshold value of 30, a low threshold value of 150 and the kernel size set to the default 3x3. <br>
<br>
This enables drawing green contours around the letters. Only extreme outer contours are retrieved and a contour approximation method (*cv2.CHAIN_APPROX_SIMPLE*) is applied to compress the contours. The final image is the saved to the output folder.


# Results
The below image displays the result of the full letter detection process (other output images can be found in output folder). The process appears to have been rather successful and most letters are correctly identified and highlighted with coherent contours. It is, however, not a flawless result and it seems to have struggled with excluding the inner regions of letters such as D and O. This problem could perhaps be alleviated by adjusting the kernel sizes of the blurring filter or during the canny edge detection. Furthermore, a few cracks in the wall has also been identified.  <br>
### Image of detected edges
![](output/image_letters.png)
<br>
<br>
In general, the script could be favourably updated to be usable on any input image by adding the ability to determine the ROI coordinates in the command line. Furthermore, it would also be relevant to improve the script by giving the user control over various parameters such as thresholding algorithms, kernel sizes and thresholds through command line arguments. This would enable the user to flexible adjust the script to varying image compositions and object detection tasks. It could e.g., be interesting to see if the result could be improved through the use of adaptive thresholding.

# References
Bradski, G. (2000). The OpenCV Library. Dr. Dobb’s Journal of Software Tools.
<br>
<br>
Davis, L. S. (1975). A survey of edge detection techniques. Computer graphics and image processing, 4(3), 248-270.

# License
Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg


