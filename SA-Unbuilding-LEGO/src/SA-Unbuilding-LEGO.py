#-----# Importing packages #-----#
from pathlib import Path
import cv2
import numpy as np
import argparse
import random as rng

#-----# Project desctiption #-----#

# Unbuilding a LEGO bricks - Segmenting a LEGO creation into seperate bricks and classifying the brick types
'''
pg
''' 

#-----# Defining main function #-----#

# Defining main function
def main(args):
    
    # Setting input image to the input given in command line
    input_creation = args.inp

    # Creating class object
    BrickRecogniton(input_creation = input_creation) 

#-----# Defining class #-----#
class BrickRecogniton:

    def __init__(self, input_creation = None):
        
        # Setting data directory and root directory 
        data_dir = self.setting_data_directory()
        
        # Setting output directory for the generated images
        out_dir = self.setting_output_directory()
        
        # Setting input image
        self.input_creation = input_creation

        # If target image is not specified, assign the fist image in folder as the target image as default
        if self.input_creation is None:

            self.input_creation = "snake.png"  # Setting default data directory

            print(f"Input image file name is not specified.\nSetting it to '{self.input_creation}'.\n")

        # Define target image file path
        input_creation_filepath = data_dir / 'creations' / str(self.input_creation)

        # Load image
        loaded_input_creation = cv2.imread(str(input_creation_filepath))

        # Detecting edges and drawing contours around letters
        bbox_image, contour_image, bboxes = self.detect_bricks(loaded_input_creation)

        bbox1 = self.extract_bbox(loaded_input_creation, bboxes[1])

        # Resizing images to be 300 pixels wide
        width = int(300)
        height = int((300 / bbox1.shape[1]) * bbox1.shape[0])

        # Resizing
        bbox1 = cv2.resize(bbox1, (width, height))

        # Background 
        final_bbox1_image = np.zeros((400, 400, 3), dtype = "uint8")

        x_offset = int((400 - bbox1.shape[1])/2)
        y_offset = int((400 - bbox1.shape[0])/2)

        final_bbox1_image[y_offset:y_offset+bbox1.shape[0], x_offset:x_offset+bbox1.shape[1]] = bbox1

        cv2.imwrite(str(out_dir) + '/' + "bricks_with_bboxes.png", bbox_image)
        
        cv2.imwrite(str(out_dir) + '/' + "bricks_with_contours.png", contour_image)

        cv2.imwrite(str(out_dir) + '/' + "bbox_image1.png", final_bbox1_image)

    #-----# Utility functions #-----#

    # Defining function for setting directory for the raw data
    def setting_data_directory(self):

        root_dir = Path.cwd()  # Setting root directory

        data_dir = root_dir / 'data'  # Setting data directory

        return data_dir

    # Defining function for setting directory for the raw data
    def setting_output_directory(self):

        root_dir = Path.cwd()  # Setting root directory

        out_dir = root_dir / 'output'  # Setting output data directory

        return out_dir


    # Defining function for finding each brick on the image
    '''
    Takes an input image, applies edge detection and returns an image with contours drawn, 
    an image with bboxes drawn and a list of bbox coordinates for all seperate bricks
    Args:
        input_image : (img) Input image
    Returns:
        bbox_image : (img) Output image with bboxes drawn around bricks
        contour_image : (img) Output image with only contours drawn around bricks
        bboxes : (list) list of coordinates for each bounding box
    '''
    def detect_bricks(self, input_image):
        # Transform input into hsv space
        hsv_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)

        # Split and extract hue channel
        h, _, _ = cv2.split(hsv_image)

        # Applying canny edge detection
        canny = cv2.Canny(h, 50, 150)

        # Applying dialation
        dialated = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

        # Setting contours
        contours, _ = cv2.findContours(dialated,
                cv2.RETR_LIST,
                cv2.CHAIN_APPROX_SIMPLE)
        
        contour_image = input_image.copy()

        cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)

        # Remove largest contour (contour drawn around all bricks)
        sorted_contours = sorted(contours, key=cv2.contourArea)

        sorted_contours.pop()

        # Creating a variable called output image that is a copy of input image
        bbox_image = input_image.copy()

        # Creating list for boundning boxes the same lenght as number of contours
        bboxes = [None]*len(sorted_contours)

        # Loop through all contours and draw them and their corresponting bounding box
        for i in range(len(sorted_contours)):
            bboxes[i] = cv2.boundingRect(sorted_contours[i])
            
            cv2.drawContours(bbox_image, sorted_contours, i, (0, 255, 0), 1)

            # Create random colour
            colour = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))

            # Draw rectangle onto image with line size of 2
            cv2.rectangle(bbox_image, (int(bboxes[i][0]), int(bboxes[i][1])), 
            (int(bboxes[i][0]+bboxes[i][2]), int(bboxes[i][1]+bboxes[i][3])), colour, 3)

        return bbox_image, contour_image, bboxes

    # Defining functiong for cropping out bbox from origianl image
    def extract_bbox(self, input_image, bbox):
        
        # Crop out bbox from original image
        bbox_image = input_image[ bbox[1]:(bbox[1]+bbox[3]), bbox[0]:(bbox[0]+bbox[2])]

        return bbox_image
    
# Executing main function when script is run
if __name__ == '__main__':
    
    #Create an argument parser from argparse
    parser = argparse.ArgumentParser(description = "[INFO] Identifying Letters using Edge Detection ",
                                formatter_class = argparse.RawTextHelpFormatter)

    # Creating argument variable for target image
    parser.add_argument('-inp',
                        metavar="--input_creation",
                        type=str,
                        help=
                        "[DESCRIPTION] Name of the file of the input image of a LEGO(R) creation \n"
                        "[TYPE]        str \n"
                        "[DEFAULT]     snake.png \n"
                        "[EXAMPLE]     -inp snake.png",
                        required=False)           

    main(parser.parse_args())