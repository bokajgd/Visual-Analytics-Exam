#-----# Importing packages #-----#
import os
from pathlib import Path
import pandas as pd
import cv2
import argparse

#-----# Project desctiption #-----#

# Image similarity check using 3D colout his5ograms

'''
- Use the cv2.compareHist()s function to compare the 3D color histogram for your target image to each of the other images in the corpus one-by-one.
- In particular, use chi-square distance method, like we used in class. Round this number to 2 decimal places.
- Save the results from this comparison as a single .csv file, showing the distance between your target image and each of the other images. The .csv file should show the filename for every image in your data except the target and the distance metric between that image and your target. Call your columns: filename, distance.
- Print the filename of the image which is 'closest' to your target image
''' 


#-----# Defining main function #-----#
def main(args):

    # Setting target image to the input given in command line
    target_image = args.ti

    # Initialisting class
    HistComparison(target_image = target_image) 

#-----# Defining class #-----#
class HistComparison:

    #-----# Defining __init__ method #-----#
    def __init__(self, target_image = None):

        # Setting data directory and root directory 
        data_dir = self.setting_data_directory() 

        # Setting output directory for the generated csv file
        out_dir = self.setting_output_directory() 

        # Getting list of filepaths for the images
        files = self.get_paths_from_data_directory(data_dir, "jpg") 

        # Setting target image
        self.target_image = target_image 

        # If target image is not specified, assign the fist image in folder as the target image as default
        if self.target_image is None:

            self.target_image = "image_0001.jpg"  

            print(f"\nTarget image filepath is not specified.\nSetting it to '{self.target_image}'.\n")

        # Define target image file path
        target_image_filepath = data_dir / str(self.target_image + '.jpg')

        # Read target image
        loaded_target_image = cv2.imread(str(target_image_filepath))

        # Creating histogram comparision image
        hist_target = cv2.calcHist([loaded_target_image], [0, 1, 2], None, [8,8,8], [0,256, 0,256, 0,256]) 

        # Normalizing hisogram using minmax normalisation
        hist_target = cv2.normalize(hist_target, hist_target, 0,255, cv2.NORM_MINMAX)

        # Defining empty lists for filenames and calculated distances:
        distances = []

        filenames = []

        # Loop through files and calculate the Chi-Square distance
        for file in files:

            # Only go through images that are not the target image
            if file != target_image_filepath: 

                # Get filenames
                filename = self.get_filename(file)

                # Load image
                image_for_comparison = cv2.imread(str(file))

                # Calculate chi square distance
                chi_sqr = self.get_distance(hist_target, image_for_comparison)

                # Appending the filename to predefined empty list
                filenames.append(filename)

                # Appending the Chi-Squared distance to the predefined empty list  
                distances.append(chi_sqr)

        # Creating a dataframe with the recorded values
        output_df = self.get_dataframe(filenames, distances) 

        # Saving the generated data frame as a csv
        output_df.to_csv(out_dir / f"chi_sqr_comparisons_{self.target_image}.csv")

        # Printing the values for the image with highest similarity
        print(f"The most similar image is {output_df.iloc[output_df['distances'].argmin()]['filename']}")
        
        print(f"This image has a chi square distance of {output_df.min()['distances']}")

        # Print name and value for most similar image in command line

    #-----# Defining utility functions used in the __init__ #-----#

    # Defining function for setting directory for the raw data
    def setting_data_directory(self):

        # Setting root directory 
        root_dir = Path.cwd()

        # Setting data directory
        data_dir = root_dir / 'data' / 'jpg'

        return data_dir

    # Defining function for setting directory for the raw data
    def setting_output_directory(self):
        
        # Setting root directory
        root_dir = Path.cwd()  

        # Setting output data directory
        out_dir = root_dir / 'output'

        return out_dir

    # Defining function for obtaining individual filepaths for data files
    '''
    Creates a list containing paths to filenames in a data directory
    Args:
        data_dir: Path to the data directory.
    Returns:
        files (list): List of individual image file paths
    '''
    def get_paths_from_data_directory(self, data_dir, file_type):

        # Creating empty list
        files = [] 

        # Loop for iterating through all images in the directory and append individual file paths to the empty list files
        for file in data_dir.glob(f'*.{file_type}'): 
            
            files.append(file)

        return files


    # Defining function for obtaining filenames from the filepaths
    '''
    Creates a list of filenames in a directory.
    Args:
        file_path: File opath
    Returns:
        filename: Name of files
    '''
    def get_filename(self, file_path):

        # Extracting last chunk of the splitted filepath as the individual filename
        file_name = os.path.split(file_path)[-1]  

        return file_name


    # Defining function for calculating Chi-Square distance value
    '''
    Creates a list of filenames in a directory.
    Args:
        file_path: File path
    Returns:
        distance: Calculated chi-square value
    '''
    def get_distance(self, hist_target, comparison_image):

        # Creating histogram comparision image
        hist_comp = cv2.calcHist([comparison_image], [0, 1, 2], None, [8,8,8], [0,256, 0,256, 0,256]) 

        # Normalizing hisogram using minmax normalisation
        hist_comp = cv2.normalize(hist_comp, hist_comp, 0,255, cv2.NORM_MINMAX)

        # Calculating distance and using the round function to round to 2 decimals.
        chi_sqr = round(cv2.compareHist(hist_target, hist_comp, cv2.HISTCMP_CHISQR))

        return chi_sqr
    
    # Defining short function for creating a pd data frame 
    '''
    Gets the total number of words from all the files
    Args:
        filenames (list): list of filenames
        distances (list): list containing chi-square distances for each images

    Returns:
        df (pd data frame): pd data frame containing all the cleaned data
    '''
    def get_dataframe(self, filenames, distances):

        # Initialising a pd dataframe with the required columns and adding the data generated by former functions to these collumns 
        df = pd.DataFrame(data= {'filename': filenames, 'distances': distances})

        return df


# Executing main function when script is run
if __name__ == '__main__':
    
    #Create an argument parser from argparse
    parser = argparse.ArgumentParser(description = "[INFO] Image similarity using color histograms",
                                formatter_class = argparse.RawTextHelpFormatter)
    
    # Creating argument variable for target image
    parser.add_argument('-ti',
                        metavar="--target_image",
                        type=str,
                        help=
                        "[DESCRIPTION] Name of the target image \n"
                        "[TYPE]        str \n"
                        "[DEFAULT]     image_0001 \n"
                        "[EXAMPLE]     -ti image_0001",
                        required=False)           

    main(parser.parse_args())