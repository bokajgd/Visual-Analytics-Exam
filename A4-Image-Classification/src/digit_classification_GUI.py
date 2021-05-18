# Import model scripts
import models.lr_mnist 
import models.cnn_mnist 

# Importing module 'tkinter' for graphical user interface 
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import PhotoImage
from PIL import Image, ImageTk

# Import numpy for easier data-transformations
import numpy as np

# Importing module 'cv2' for importing and modifying images
import cv2

# Importing module 'tensorflow' for neural network processing
import tensorflow as tf

# Importing other packages
import os
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

# Setting graphics directory
graphics_dir = Path.cwd() / 'W7-Image-Classification' / 'graphics'


# Defing main class
class MainApp(tk.Tk): # Defining the main-window as a class for back-end control of pages in the GUI
    def __init__(self, *args, **kwargs): # Have the class self-initiate all the following when the app is run
        tk.Tk.__init__(self, *args, **kwargs) # Initiate tkinter functions
        self.title("Digit Classifier") # Set the title of the main-app
        self.geometry("1000x500") # Set the size of the main-app window
        self.pack_propagate(0) # Set pack_propagate to zero to not allow widgets to control window-size
        self.resizable(0,0) # Don't allow resizing of windows

        box = tk.Frame(self) # create a box for all pages
        box.pack(side="top", fill="both", expand=True) # calling the container
        box.grid_rowconfigure(0, weight=1) # make the cell for rows and columns cover the entire defined window
        box.grid_columnconfigure(0, weight=1)

        self.frames = {} # create a dictionary of frames we want to navigate
        
        for F in (StartPage, CNN, LogReg, CNNclassImage): # for each page
            frame = F(box, self) # create the page
            self.frames[F] = frame  # store into frames
            frame.grid(row=0, column=0, sticky="nsew") # grid it to container

        self.showFrame(StartPage) # set StartPage to be the front page
    

    def showFrame(self, name): # defining a function for showing frames
        frame = self.frames[name] # specify which frame to show
        frame.tkraise() # raise the frame to top

    def addImage(self): # defining a function for choosing image
        global file 
        file = filedialog.askopenfilename(initialdir="/", title="Select File", 
                                            filetypes=(("JPG-images", "*.jpg"), ("all files", "*")))

    def openImage(self): # defining a function for pasting chosen image to frame
        global yourImgLabel
        global panel

        img = Image.open(file) # Opening image
        img = img.resize((170, 160)) # Uploaded image will be warped to a square like in the model
        img = ImageTk.PhotoImage(img) #  Image into PhotoImage format
        panel = tk.Label(self, image=img) # Turn image into label
        panel.image = img
        panel.place(relx=0.26, y=180) # Place label on frame

        # creating a label for header for white frame
        yourImgLabel = tk.Label(self, text="Your Image", relief="flat", background = "white",
                        foreground = 'black', width = 10, height = 1, font=("Arial", 18))
                        
        yourImgLabel.place(relx=0.29, y=145) # align it at the center of main-app

    def removeText(self): # defining a function for removing text labels when leaving a page
        predictionLabel.destroy() # remove the label defined as predictionLabel
        prediction_bar_plot_panel.destroy() 

    def removeImage(self): # defining a function for removing images when leaving a page
        panel.destroy() # remove the variable defined as panel

    def reset_cnn(CNN): # 
        trainModel.place_forget() # forget the variable defined as 'classifyButton'
        CNNresetButton.place(relx=0.21, rely=0.8, height=40, width=170) # place the button-variable defined as 'resetButton'

    def reset_lr(LogReg):
        trainModel.place_forget() # forget the variable defined as 'trainModels'
        LRresetButton.place(relx=0.2 , rely=0.8, height=40, width=170) # place the button-variable defined as 'resetButton'

    def cnn_replace(CNN, results_panel, nn_graph_panel, layers_choice, default, nodes_choice):
        CNNresetButton.place_forget() # forget the variable defined as 'resetButton'
        results_panel.place_forget() 
        nn_graph_panel.place_forget() 
        layers_choice.set(default) # restteting entry box and dropdown values
        nodes_choice.set("")
        trainModel.place(relx=0.21, rely=0.8, height=40, width=170) # place button on frame

    def lr_replace(LogReg, log_results_panel, lr_graph_panel, pen_choice, default, tol_choice):
        LRresetButton.place_forget() # forget the variable defined as 'resetButton'
        log_results_panel.place_forget() 
        lr_graph_panel.place_forget() 
        pen_choice.set(default) # restteting entry box and dropdown values
        tol_choice.set("")
        trainModel.place(relx=0.21, rely=0.8, height=40, width=170) # place button on frame

    def cnn_class_remove_predictions(self, prediction_bar_plot_panel, predictionLabel):
        prediction_bar_plot_panel.destroy() # forget the variable defined as 'resetButton'
        predictionLabel.destroy() 

    def cnn_class_remove_image(self, yourImgLabel, panel):
        yourImgLabel.destroy() 
        panel.destroy()
        
    def reset_cnn_classification(CNNclassImage):
        chooseImg.place_forget()
        CNNclassResetButton.place(relx=0.20, rely=0.8, height=40, width=170) # place button on frame

    def cnn_class_replace_chooseImage(CNNclassImage):
        CNNclassResetButton.place_forget() # forget the variable defined as 'resetButton'
        chooseImg.place(relx=0.20, rely=0.8, height=40, width=170) # place button on frame

    def preprocess(self,file): # defining a function for preprocessing new images
        global chosenImage # making the variable global
        imgSize = 28 # setting standard pixel width/height for all images
        chosenImage = cv2.imread(file, cv2.IMREAD_GRAYSCALE) / 255.0 # reading image using cv2 and standardizing
        chosenImage = cv2.resize(chosenImage, (imgSize, imgSize)) # resizing the image to standard size
        chosenImage = chosenImage.reshape(-1, imgSize, imgSize, 1) # reshaping image to allow model-prediction

    def runModel(self, n_layers, n_nodes): # defining function analysing preprocessed models
        global prediction # making the variable global
        model_path = Path.cwd() / 'W7-Image-Classification' / 'model_out' / f"{n_layers}-dense-{n_nodes}-nodes-CNN.model"
        model = tf.keras.models.load_model(str(model_path)) # loading the model from current directory
        prediction = model.predict([chosenImage]) # categorizing an image saving and saving the prediction

    def predCalculation(self, prediction): # defining model for pasting prediction
        global predictionLabel
        global prediction_bar_plot_panel
        global predictionLabel

        preds = [j for sub in prediction for j in sub] # Flatten prediction to 1d array using list comprehension
        classes = list(range(0,10))

        fig = plt.figure(figsize=(8, 5)) 
        sns.barplot(x=classes, y=preds)
        plt.ylabel("Probability")
        plt.xlabel("Class")
        plt.savefig(Path.cwd() / 'W7-Image-Classification' / 'model_out' / "latest-prediction.png")

        number_prediction = preds.index(max(preds)) 
        certainty = str(round(max(preds)*100, 2))
        predictionLabel = tk.Label(self, text=f"The image displays the number {number_prediction} \n Certainty: {certainty}%",
                                      relief="flat", background = "white",   
                                    foreground = 'black', width = 14, height = 3, font=("Arial", 13))
                                    # defining the predictionLabel variables as label
        predictionLabel.place(relx=0.6, rely=0.715,relwidth=0.2, anchor='s') # placing the prediction label on the frame

        prediction_bar_plot = Image.open(Path.cwd() / 'W7-Image-Classification' / 'model_out' / "latest-prediction.png") # Opening graph
        prediction_bar_plot = prediction_bar_plot.resize((230, 145)) # Uploaded image will be warped to a square like in the model
        prediction_bar_plot = ImageTk.PhotoImage(prediction_bar_plot) #  Image into PhotoImage format
        prediction_bar_plot_panel = tk.Label(self, image=prediction_bar_plot) # Turn image into label
        prediction_bar_plot_panel.image = prediction_bar_plot
        prediction_bar_plot_panel.place(relx = 0.485, rely = 0.33) # Place label on frame


    def trainCNN(self, layers_choice, choose_n_nodes):
        global results_panel
        global nn_graph_panel
        global n_layers
        global n_nodes_list
        global n_nodes

        n_layers = layers_choice.get() # Retrieve inputs from entry box and dropdown
        n_nodes = choose_n_nodes.get() 
        n_layers = int(n_layers)
        n_nodes_list = n_nodes.split(",") # Split input into a list
        n_nodes_list = [ int(x) for x in n_nodes_list] # Convert entries to integers

        if n_layers == len(n_nodes_list): # Only run if n_layers is identical to the lenght of the n_nodes list
            cm = models.cnn_mnist.cnn_mnist(n_layers, n_nodes_list) # Run nn model

            results_panel = tk.Label(self, text =  cm, font=('calibre',9)) # Create panel showing prediction results
            results_panel.place(x=320, y=160)

            nn_graph = Image.open(Path.cwd() / 'W7-Image-Classification' / 'model_out' / f"{n_layers}-dense-{n_nodes}-nodes-CNN-viz.png") # Opening graph
            nn_graph = nn_graph.resize((232, 232)) # Uploaded image will be warped to a square like in the model
            nn_graph = ImageTk.PhotoImage(nn_graph) #  Image into PhotoImage format
            nn_graph_panel = tk.Label(self, image=nn_graph) # Turn image into label
            nn_graph_panel.image = nn_graph
            nn_graph_panel.place(x=590, y=132) # Place label on frame

            self.reset_cnn() # Create reset button

    def trainLogReg(self, pen_choice, choose_tol):
        global log_results_panel
        global lr_graph_panel

        pen = pen_choice.get() # Retrieve inputs from entry box and dropdown
        tol = choose_tol.get() 
        tol = float(tol)# Split input into a list

        cm = models.lr_mnist.lr_mnist(pen, tol) # Run lr model

        log_results_panel = tk.Label(self, text =  cm, font=('calibre',9)) # Create panel showing prediction results
        log_results_panel.place(x=310, y=154)

        lr_graph = Image.open(Path.cwd() / 'W7-Image-Classification' / 'model_out' /  f"{pen}-penalty-{tol}-tol-nodes-LR-viz.png") # Opening graph
        lr_graph = lr_graph.resize((370, 235)) # Uploaded image will be warped to a square like in the model
        lr_graph = ImageTk.PhotoImage(lr_graph) #  Image into PhotoImage format
        lr_graph_panel = tk.Label(self, image=lr_graph) # Turn image into label
        lr_graph_panel.image = lr_graph
        lr_graph_panel.place(x=525, y=130) # Place label on frame

        self.reset_lr() # Create reset button


class StartPage(tk.Frame): # Defining a class for the start page inheriting functions from tk.Frame
    def __init__(self, parent, controller): # have the class self-initiate functions from itself and parent-class
        tk.Frame.__init__(self, parent)# initiate a frame for the page

        photo = tk.PhotoImage(file = graphics_dir / "bgHome.gif") # add a background image
        bg = tk.Label(self, image=photo) # create a label containing the specified photo
        bg.place(x=0, y=0, relwidth=1, relheight=1) # placing the bg filling the entire frame
        bg.image = photo # using the .image function of tkiner to call the background as the photo

        style = ttk.Style() # styling buttons using ttk module
        style.configure('Custom.TButton', font=("Arial Bold", 13),
                        foreground='#326DB2') # defining specifications for buttons

        NNmodelButton = ttk.Button(self, text="Logistic Regression", style = 'Custom.TButton',
                                command=lambda : controller.showFrame(LogReg))
                                # adding button to access the ModelPage
        NNmodelButton.place(relx=0.2, rely=0.8, height=40, width=170) # placing the button on the page

        LRmodelButton = ttk.Button(self, text="Neural Network", style = 'Custom.TButton',
                                command=lambda : controller.showFrame(CNN))
                                # adding button to access the InfoPage
        LRmodelButton.place(relx=0.64, rely=0.8, height=40, width=170) # placing the button on the page


class CNN(tk.Frame): # define a class for running the model on personally chosen picture
    def __init__(self, parent, controller): # have the class self-initiate functions from itself and parent-class as well as controller

        global trainLogReg
        global CNNresetButton

        tk.Frame.__init__(self, parent) # initiate a frame for the page

        background = tk.PhotoImage(file= graphics_dir / "bgStandard.gif") # set background
        bg = tk.Label(self, image=background) # defining bg as a label using defined image
        bg.place(x=0, y=0, relwidth=1, relheight=1) # placing the image as background
        bg.image = background # using the .image function of tkiner to call the background as the photo

        label = tk.Label(self, text="Convolutional Neural Network", relief="flat", background = "#42669C",
                        foreground = 'white', width = 450, height = 2, font=("Arial Bold", 28), highlightcolor="white") # add a label as header
        label.place(relx=0.286, rely=0.08, width = 450) # place label and align it at the center of main-app

        lowerFrame = tk.Frame(self, bg='white', bd=10, relief='raised', borderwidth = 5) #  adding white frame in middle of frame
        lowerFrame.place(relx=0.5, rely=0.25, relwidth=0.8, relheight=0.5, anchor='n') # placing the frame

        # Add dropdown
        layer_choices = [1, 2, 3]
        layers_choice = tk.StringVar(self)
        default = layer_choices[0]
        choose_n_layers = ttk.OptionMenu(self, layers_choice, default, *layer_choices)
        
        # Add entry box
        nodes_choice = tk.StringVar(self)
        choose_n_nodes = tk.Entry(self, textvariable = nodes_choice, font=('calibre',10,'normal'))
         
        # Add labels
        dropdown_label = tk.Label(self, text = 'Choose number of layers', font=('calibre',10, 'bold'))
        entry_label = tk.Label(self, text = 'Choose number of nodes', font=('calibre',10, 'bold'))

        # Placing stuff
        dropdown_label.place(rely = 0.38, relx = 0.12)
        choose_n_layers.place(rely = 0.42, relx = 0.12)
        entry_label.place(rely = 0.53, relx = 0.12)
        choose_n_nodes.place(rely = 0.57, relx = 0.12)

        style = ttk.Style() # styling buttons using ttk module
        style.configure('Custom.TButton', font=("Arial Bold", 13),
                            foreground='#326DB2') # defining specifications for buttons

        trainModel = ttk.Button(self, text="Train and Evaluate", style = 'Custom.TButton',
                                command=lambda : [controller.trainCNN(layers_choice, choose_n_nodes)])
                                # add button for adding and opening images
        trainModel.place(relx=0.21, rely=0.8, height=40, width=170) # place button on frame

        CNNstartButton = ttk.Button(self, text="Start Page", style = 'Custom.TButton',
                                command=lambda : [controller.showFrame(StartPage), controller.cnn_replace(results_panel, nn_graph_panel, layers_choice, default, nodes_choice)])
                                # add button for accessing StartPage and removing image
        CNNstartButton.place(relx=0.62, rely=0.8, height=40, width=170) # place button on frame

        CNNresetButton = ttk.Button(self, text="Reset", style = 'Custom.TButton',
                                command=lambda : [controller.cnn_replace(results_panel, nn_graph_panel, layers_choice, default, nodes_choice)])
                                # add button for running model and producing prediction            

        CNNclassImageButton = ttk.Button(self, text="Classify New Image", style = 'Custom.TButton',
                    command=lambda : [controller.showFrame(CNNclassImage), controller.cnn_replace(results_panel, nn_graph_panel, layers_choice, default, nodes_choice)])
                    # adding button to access the InfoPage
        CNNclassImageButton.place(relx=0.414, rely=0.8, height=40, width=170) # placing the button on the page


class LogReg(tk.Frame): # define a class for running the model on personally chosen picture
    def __init__(self, parent, controller): # have the class self-initiate functions from itself and parent-class as well as controller

        global trainModel
        global LRresetButton
        global style

        tk.Frame.__init__(self, parent) # initiate a frame for the page

        background = tk.PhotoImage(file= graphics_dir / "bgStandard.gif") # set background
        bg = tk.Label(self, image=background) # defining bg as a label using defined image
        bg.place(x=0, y=0, relwidth=1, relheight=1) # placing the image as background
        bg.image = background # using the .image function of tkiner to call the background as the photo

        label = tk.Label(self, text="Logistic Regression", relief="flat", background = "#42669C",
                        foreground = 'white', width = 450, height = 2, font=("Arial Bold", 28), highlightcolor="white") # add a label as header
        label.place(relx=0.287, rely=0.08, width = 450) # place label and align it at the center of main-app

        lowerFrame = tk.Frame(self, bg='white', bd=10, relief='raised', borderwidth = 5) #  adding white frame in middle of frame
        lowerFrame.place(relx=0.5, rely=0.25, relwidth=0.8, relheight=0.5, anchor='n') # placing the frame

        # Add dropdown
        pen_choices = ['l1', 'l2', 'elasticnet', 'none']
        pen_choice = tk.StringVar(self)
        default = pen_choices[0]
        choose_pen = ttk.OptionMenu(self, pen_choice, default, *pen_choices)
        
        # Add entry box
        tol_choice = tk.StringVar(self)
        choose_tol = tk.Entry(self, textvariable = tol_choice, font=('calibre',10,'normal'))
         
        # Add labels
        dropdown_label = tk.Label(self, text = 'Choose penalty', font=('calibre',12, 'bold'))
        entry_label = tk.Label(self, text = 'Choose tolerance', font=('calibre',12, 'bold'))

        # Placing stuff
        dropdown_label.place(rely = 0.38, relx = 0.12)
        choose_pen.place(rely = 0.42, relx = 0.12)
        entry_label.place(rely = 0.53, relx = 0.12)
        choose_tol.place(rely = 0.57, relx = 0.12)

        style = ttk.Style() # styling buttons using ttk module
        style.configure('Custom.TButton', font=("Arial Bold", 13),
                            foreground='#326DB2') # defining specifications for buttons

        trainModel = ttk.Button(self, text="Train and Evaluate", style = 'Custom.TButton',
                                command=lambda : [controller.trainLogReg(pen_choice, choose_tol)])
                                # add button for adding and opening images
        trainModel.place(relx=0.21, rely=0.8, height=40, width=170) # place button on frame

        LRstartButton = ttk.Button(self, text="Start Page", style = 'Custom.TButton',
                                command=lambda : [controller.showFrame(StartPage), controller.lr_replace(log_results_panel, lr_graph_panel, pen_choice, default, tol_choice)])
                                # add button for accessing StartPage and removing image
        LRstartButton.place(relx=0.59, rely=0.8, height=40, width=170) # place button on frame

        LRresetButton = ttk.Button(self, text="Reset", style = 'Custom.TButton',
                                command=lambda : [controller.lr_replace(log_results_panel, lr_graph_panel, pen_choice, default, tol_choice)])
                                # add button for running model and producing prediction

class CNNclassImage(tk.Frame): # define a class for classifying new image
    def __init__(self, parent, controller): # have the class self-initiate functions from itself and parent-class as well as controller
       
        global chooseImg
        global CNNclassResetButton

        tk.Frame.__init__(self, parent) # initiate a frame for the page

        background = tk.PhotoImage(file= graphics_dir / "bgStandard.gif") # set background
        bg = tk.Label(self, image=background) # defining bg as a label using defined image
        bg.place(x=0, y=0, relwidth=1, relheight=1) # placing the image as background
        bg.image = background # using the .image function of tkiner to call the background as the photo

        label = tk.Label(self, text="Classify New Image", relief="flat", background = "#42669C",
                        foreground = 'white', width = 450, height = 2, font=("Arial Bold", 28), highlightcolor="white") # add a label as header
        label.place(relx=0.32, rely=0.08, width = 350) # place label and align it at the center of main-app

        lowerFrame = tk.Frame(self, bg='white', bd=10, relief='raised', borderwidth = 5) #  adding white frame in middle of frame
        lowerFrame.place(relx=0.5, rely=0.25, relwidth=0.6, relheight=0.5, anchor='n') # placing the frame

        chooseImg = ttk.Button(self, text="Choose Image", style = 'Custom.TButton',
                                command=lambda : [controller.addImage(), controller.openImage()])
                                # add button for adding and opening images
        chooseImg.place(relx=0.20, rely=0.8, height=40, width=170) # place button on frame


        classifyButton = ttk.Button(self, text="Classify", style = 'Custom.TButton',
                                command=lambda : [controller.preprocess(file), controller.runModel(n_layers, n_nodes), 
                                                controller.predCalculation(prediction), controller.reset_cnn_classification()])
                                # add button for running model and producing prediction
        classifyButton.place(relx = 0.41, rely=0.8, height=40, width=170) # place button on frame

        CNNbackButton = ttk.Button(self, text="Go Back", style = 'Custom.TButton',
                                command=lambda : [controller.showFrame(CNN), 
                                                  controller.cnn_class_remove_predictions(prediction_bar_plot_panel, predictionLabel),
                                                  controller.cnn_class_remove_image(yourImgLabel, panel),
                                                  controller.cnn_class_replace_chooseImage()])
                                # add button for accessing StartPage and removing image
        CNNbackButton.place(relx=0.63, rely=0.8, height=40, width=170) # place button on frame

        CNNclassResetButton = ttk.Button(self, text="Reset", style = 'Custom.TButton',
                        command=lambda : [controller.removeImage(), controller.removeText(), 
                                          controller.cnn_class_remove_predictions(prediction_bar_plot_panel, predictionLabel),
                                          controller.cnn_class_remove_image(yourImgLabel, panel),
                                          controller.cnn_class_replace_chooseImage()])
                        # add button for running model and producing prediction
 
app = MainApp() # define variable 'app' as the MainApp() for running tkinter window
app.mainloop() # call the mainloop() function on app to activate all defined frames



