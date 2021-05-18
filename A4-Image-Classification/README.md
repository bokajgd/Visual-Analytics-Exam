# W7 - Digit Clasification 

# This repo is not yet finished - please check tomorrow! Thank you!

# Overview 

**Jakob Grøhn Damgaard, March 2021** <br/>
This repository contains the W5 assigmnent for the course *Visual Analytics*

# Code
The code to execute the tasks can be found in the file *W5-Edge-Detection.py*<br/>

# Data
The image files is located in the *data* folder <br/>
The script generates three .png images:
<br>
   - *image_with_ROI.png* - This is the original image with a green bounding box delimiting the ROI <br>
   - *only_ROI.png* - This is a cropped version of the original image containing only the ROI
   - *image_with_contours.png* - This is the final output image with all the contours drawn around the letters

# Download and Execute
To locally download a compressed zip version of this repository, one can zip the entire repository from GitHub by navigating back to the home page of the repository and clicking the *Code* button and then *Download ZIP*. <br/>
<br>
Before executing the .py file, open the terminal, navigate the directory to the folder directory and run the following code to install the requirements list in the *requirements.txt* file in a virtual environment:
<br>
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
deactivate
```

You can then proceed to run the script in the terminal by running the following line: 

```bash
python W5-Edge-Detection.py --input text_image.jpeg
```
I hope everything works! 

# License
Shield: [![CC BY-SA 4.0][cc-by-sa-shield]][cc-by-sa]

This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].

[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

