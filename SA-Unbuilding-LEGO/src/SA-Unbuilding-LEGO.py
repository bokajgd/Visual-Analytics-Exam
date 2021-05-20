#-----# Importing packages #-----#
from os import pathconf_names
from pathlib import Path
import cv2
import numpy as np
import argparse
import random as rng
from numpy.lib.function_base import percentile
import tensorflow as tf

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
        self.data_dir = self.setting_data_directory()
        
        # Setting output directory for the generated images
        self.out_dir = self.setting_output_directory()
        
        # Setting input image
        self.input_creation = input_creation

        # If target image is not specified, assign the fist image in folder as the target image as default
        if self.input_creation is None:

            self.input_creation = "snake.png"  # Setting default data directory

            print(f"Input image file name is not specified.\nSetting it to '{self.input_creation}'.\n")

        # Define target image file path
        input_creation_filepath = self.data_dir / 'creations' / str(self.input_creation)

        # Load image
        loaded_input_creation = cv2.imread(str(input_creation_filepath))

        # Define path for pre-trained model
        model_path = self.out_dir / 'model_outputs' / "lego-CNN_2_epochs.model"

        # Load model
        self.model = tf.keras.models.load_model(str(model_path)) 

        # Detecting edges and drawing contours around letters
        bbox_image, contour_image, bboxes = self.detect_bricks(loaded_input_creation)

        # Creating header for final image 
        real_and_predicted_bricks = np.zeros((200, 800, 3), dtype = "uint8")

        # Adding header
        cv2.putText(real_and_predicted_bricks, 'Unbuilding LEGO using', (100, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.7, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(real_and_predicted_bricks, 'Object Detection and Deep Learning', (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 255, 255), 2, cv2.LINE_AA)

        # Loop through all extracted bboces
        for i in range(len(bboxes)):

            # Crop out the part of the original picture that is bounded by the bbox coordiantes
            bbox = self.extract_bbox(loaded_input_creation, bboxes[i])

            # Get image of brick on black background
            brick_image = self.get_brick_image(bbox, i)

            # Create a copy so image is not altered
            image_for_prediction = brick_image.copy() 

            # Run image through model to make prediction
            prediction = self.predict_brick(image_for_prediction) 

            # Merge cropped brick and example image of predicted brick into a collage
            real_and_predicted_bricks = self.collate_predictions(real_and_predicted_bricks, prediction, brick_image, i) 

        # Save images to output folder
        cv2.imwrite(str(self.out_dir) + '/' + f"{self.input_creation[:len(self.input_creation)-4]}_bricks_with_bboxes.png", bbox_image)
        
        cv2.imwrite(str(self.out_dir) + '/' + f"{self.input_creation[:len(self.input_creation)-4]}_bricks_with_contours.png", contour_image)

        cv2.imwrite(str(self.out_dir) + '/' + f"{self.input_creation[:len(self.input_creation)-4]}_detected_vs_predicted_bricks.png", real_and_predicted_bricks)


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

    # Defining function for generating and saving cropped bricks using extravted bboxes
    def get_brick_image(self, bbox, i):
        # Resizing image to be 300 pixels wide
        width = int(300)

        # Height is set according to ratio of how much the image was reduced in width
        height = int((300 / bbox.shape[1]) * bbox.shape[0])

        # Resizing
        bbox = cv2.resize(bbox, (width, height))

        # Creating black ackground 
        final_bbox_image = np.zeros((400, 400, 3), dtype = "uint8")

        # Calculating offsets in sides in order to place bbox image in middle of the black 400x400 frame
        x_offset = int((400 - bbox.shape[1])/2)

        y_offset = int((400 - bbox.shape[0])/2)

        # Placing image
        final_bbox_image[y_offset:y_offset+bbox.shape[0], x_offset:x_offset+bbox.shape[1]] = bbox

        # Saving image of extracted brick
        cv2.imwrite(str(self.out_dir) + '/detected_bricks/' + f"{self.input_creation[:len(self.input_creation)-4]}_brick_number{i}.png", final_bbox_image)

        return final_bbox_image
    
    # Defining function for generating a prediction of the located brick using the pre-train cnn model
    def predict_brick(self, model_input):

        # Convert to grayscale
        model_input = cv2.cvtColor(model_input, cv2.COLOR_BGR2GRAY)

        # Rescaling and  reshaping before prediction
        model_input = cv2.resize(model_input, (132, 132)) 

        model_input  = model_input.reshape(-1, 132, 132, 1)
        
        prediction = self.model.predict([model_input])

        return prediction

    # Defining function for making a collage of the predicted bricks
    def collate_predictions(self, real_and_predicted_bricks, prediction, brick_image, i):
        
        # Find predicted class and mere cropped image of brick in the creation with the example image of the predicted brick
        if np.argmax(prediction) == 0:
            
            # Read example image
            path_2x2_brick = self.data_dir / 'example_images' / 'Brick_2x2.png'

            brick_2x2 = cv2.imread(str(path_2x2_brick))  
            
            # Add text lavbles
            cv2.putText(brick_2x2, 'Predicted Brick', (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(brick_image, f'Brick_number_{i}', (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            real_and_predicted_brick = np.concatenate((brick_image, brick_2x2), axis=1)
        
        if np.argmax(prediction) == 1:
            
            path_2x2_plate = self.data_dir / 'example_images' / 'Plate_2x2.png'

            plate_2x2 = cv2.imread(str(path_2x2_plate))

            cv2.putText(plate_2x2, 'Predicted Brick', (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(brick_image, f'Brick_number_{i}', (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            real_and_predicted_brick = np.concatenate((brick_image, plate_2x2), axis=1)

        if np.argmax(prediction) == 2:
            
            path_2x3_brick = self.data_dir / 'example_images' / 'Brick_2x3.png'

            brick_2x3 = cv2.imread(str(path_2x3_brick))  
            
            cv2.putText(brick_2x3, 'Predicted Brick', (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(brick_image, f'Brick_number_{i}', (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            real_and_predicted_brick = np.concatenate((brick_image, brick_2x3), axis=1)

        if np.argmax(prediction) == 3:
            
            path_2x3_plate = self.data_dir / 'example_images' / 'Plate_2x3.png'

            plate_2x3 = cv2.imread(str(path_2x3_plate))  
            

            cv2.putText(plate_2x3, 'Predicted Brick', (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(brick_image, f'Brick_number_{i}', (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            real_and_predicted_brick = np.concatenate((brick_image, plate_2x3), axis=1)

        if np.argmax(prediction) == 4:
            
            path_2x4_brick = self.data_dir / 'example_images' / 'Brick_2x4.png'

            brick_2x4 = cv2.imread(str(path_2x4_brick))  
            
            cv2.putText(brick_2x4, 'Predicted Brick', (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(brick_image, f'Brick_number_{i}', (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            real_and_predicted_brick = np.concatenate((brick_image, brick_2x4), axis=1)

        if np.argmax(prediction) == 5:
            
            path_2x4_plate = self.data_dir / 'example_images' / 'Brick_2x4.png'

            plate_2x4 = cv2.imread(str(path_2x4_plate))  

            cv2.putText(plate_2x4, 'Predicted Brick', (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(brick_image, f'Brick_number_{i}', (80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

            real_and_predicted_brick = np.concatenate((brick_image, plate_2x4), axis=1)

        # Vertically concatenating merged images with all other 'real_and_predicted_brick' images:
        real_and_predicted_bricks = np.concatenate((real_and_predicted_bricks, real_and_predicted_brick), axis=0)

        return real_and_predicted_bricks

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