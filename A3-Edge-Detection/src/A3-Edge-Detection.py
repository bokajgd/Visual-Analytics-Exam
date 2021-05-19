#-----# Importing packages #-----#
from pathlib import Path
import cv2
import argparse

#-----# Project desctiption #-----#

# Basic image processing using python
'''
- Draw a green rectangular box to show a region of interest (ROI) around the main body of text in the middle of the image. Save this as image_with_ROI.jpg.
- Crop the original image to create a new image containing only the ROI in the rectangle. Save this as image_cropped.jpg.
- Using this cropped image, use Canny edge detection to 'find' every letter in the image
- Draw a green contour around each letter in the cropped image. Save this as image_letters.jpg
''' 

#-----# Defining main function #-----#

# Defining main function
def main(args):
    
    # Setting input image to the input given in command line
    input_image = args.inp

    # Creating class object
    EdgeDetection(input_image = input_image) 

#-----# Defining class #-----#
class EdgeDetection:

    def __init__(self, input_image = None):
        
        # Setting data directory and root directory 
        data_dir = self.setting_data_directory()
        
        # Setting output directory for the generated images
        out_dir = self.setting_output_directory()
        
        # Setting input image
        self.input_image = input_image

        # If target image is not specified, assign the fist image in folder as the target image as default
        if self.input_image is None:

            self.input_image = "text_image.jpeg"  # Setting default data directory

            print(f"\Input image file name is not specified.\nSetting it to '{self.input_image}'.\n")

        # Define target image file path
        input_image_filepath = data_dir / str(self.input_image)

        # Load image
        loaded_input_image = cv2.imread(str(input_image_filepath))

        # Creating an image with a bounding box around the text 
        ROI_image = cv2.rectangle(loaded_input_image.copy(), (1395, 880), (2854, 2778), (0, 255 ,0), 10) # Change values if you wish to analyse another image

        # Saving image 
        cv2.imwrite(str(out_dir) + '/' + "image_with_ROI.png", ROI_image)

        # Creating cropped image only containing pixels inside defined ROI - Indexing is much easier than cv2 functions
        cropped_image = ROI_image[880:2778, 1395:2854] # Change  values if you wish to analyse another image

        # Saving image 
        cv2.imwrite(str(out_dir) + '/' + "only_ROI.png", cropped_image)

        # Detecting edges and drawing contours around letters
        output_image = self.find_letters(cropped_image)

        # Saving image 
        cv2.imwrite(str(out_dir) + '/' + "image_with_contours.png", output_image)


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


    # Defining function for finding letters on an image
    '''
    Takes an input image, applies edge detection and return an output image with all conours
    Args:
        input_image: Input image
    Returns:
        output_image: Output image with drawn contours
    '''
    def find_letters(self, input_image):
        # Make input image grey
        grey_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
        
        # Blurring image for optimised detection
        blurred = cv2.GaussianBlur(grey_image, (7,7), 0)

        # Applying thresholding (makes image black and white)
        _, image_bw = cv2.threshold(blurred, 115, 255, cv2.THRESH_BINARY)

        # Applying canny edge detection
        canny = cv2.Canny(image_bw, 30, 150)

        # Setting contours
        contours, _ = cv2.findContours(canny.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        
        # Adding contours to image
        output_image = cv2.drawContours(input_image.copy(), 
                         contours,               
                         -1 , 
                         (0, 255, 0), # Making contours green
                         3)

        return output_image
    
# Executing main function when script is run
if __name__ == '__main__':
    
    #Create an argument parser from argparse
    parser = argparse.ArgumentParser(description = "[INFO] Image similarity using color histograms",
                                formatter_class = argparse.RawTextHelpFormatter)

    # Creating argument variable for target image
    parser.add_argument('-inp',
                        metavar="--input",
                        type=str,
                        help=
                        "[DESCRIPTION] Name of the file of the input image \n"
                        "[TYPE]        str \n"
                        "[DEFAULT]     text_image.jpeg \n"
                        "[EXAMPLE]     -inp text_image.jpeg",
                        required=False)           

    main(parser.parse_args())