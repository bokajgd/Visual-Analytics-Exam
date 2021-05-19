# Import model scripts
import models.lr_mnist 
import models.cnn_mnist 

# Importing module 'tkinter' for graphical user interface 
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import Image, ImageTk

# Importing module 'cv2' for importing and modifying images
import cv2

# Importing module 'tensorflow' for neural network processing
import tensorflow as tf

# Importing other packages
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

#-----# Description #-----#
'''
'''

# Setting graphics directory
graphics_dir = Path.cwd() / 'A4-Image-Classification' / 'graphics'

#-----# Defing main class #-----#

# Defining the main-window as a class for back-end control of pages in the GUI
class MainApp(tk.Tk): 

    # Have the class self-initiate all the following when the app is run
    def __init__(self, *args, **kwargs):
        
        # Initiate tkinter functions
        tk.Tk.__init__(self, *args, **kwargs) 

        # Set the title of the main-app
        self.title("Digit Classifier") 

        # Set the size of the main-app window
        self.geometry("1000x500")

        # Set pack_propagate to zero to not allow widgets to control window-size
        self.pack_propagate(0) 

        # Don't allow resizing of windows
        self.resizable(0,0) 

        # Create a box for all pages
        box = tk.Frame(self) 

        # Call the box
        box.pack(side="top", fill="both", expand=True) 

        # Make the cell for rows and columns cover the entire defined window
        box.grid_rowconfigure(0, weight=1)

        box.grid_columnconfigure(0, weight=1)

        # Create a dictionary of frames we want to navigate
        self.frames = {} 
        
        # For each page
        for F in (StartPage, CNN, LogReg, CNNclassImage): 
            
            # Create the page
            frame = F(box, self) 

            
            self.frames[F] = frame 

            frame.grid(row=0, column=0, sticky="nsew") 

        # Set StartPage to be the front page
        self.showFrame(StartPage) 

    #-----# Utility functions for use in the other classes #-----#

    # Defining a function for showing frames
    def showFrame(self, name): 

        # Specify which frame to show
        frame = self.frames[name] 

        # Raise the frame to top
        frame.tkraise() 

    # Defining a function for choosing image
    def addImage(self): 

        global file 
        
        # Opening file finder window
        file = filedialog.askopenfilename(initialdir="/", title="Select File", 
                                            filetypes=(("JPG-images", "*.jpg"), ("all files", "*")))

    # Defining a function for pasting chosen image to frame
    def openImage(self): 

        global yourImgLabel
        global panel

         # Opening image
        img = Image.open(file)

        # Uploaded image will be warped to a square like in the model 
        img = img.resize((170, 160)) # kinda

        #  Image into PhotoImage format
        img = ImageTk.PhotoImage(img) 

        # Turn image into label
        panel = tk.Label(self, image=img) 

        panel.image = img

        # Place label on frame
        panel.place(relx=0.26, y=180) 

        # Creating a label for header for white frame
        yourImgLabel = tk.Label(self, text="Your Image", relief="flat", background = "white",
                        foreground = 'black', width = 10, height = 1, font=("Arial", 18))
        
        # Align it at the center of main-app and place
        yourImgLabel.place(relx=0.29, y=145) 

    # Defining a function for removing prediction label when leaving a page
    def removeText(self): 

        # Remove the label defined as predictionLabel
        predictionLabel.destroy() 

        prediction_bar_plot_panel.destroy() 

    # Defining a function for removing image when leaving a page
    def removeImage(self): 
        
        # Remove the variable defined as panel
        panel.destroy() 

    # Defining function for changing buttons on cnn page when train model is pressed
    def reset_cnn(CNN): 

        # Removing train model button
        trainModel.place_forget() 

        # Place the button-variable defined as 'resetButton'
        CNNresetButton.place(relx=0.21, rely=0.8, height=40, width=170) 

    # Defining function for chaing buttons on logReg page after train model is pressed
    def reset_lr(LogReg):

         # Forget the variable defined as 'trainModels'
        trainModel.place_forget()

        # Place the button-variable defined as 'resetButton'
        LRresetButton.place(relx=0.2 , rely=0.8, height=40, width=170) 

    # Defining function for actions taken when reset button is pressed on cnn page
    def cnn_replace(CNN, results_panel, nn_graph_panel, layers_choice, default, nodes_choice):

        # Forget buttons, results and fully connected layers graph
        CNNresetButton.place_forget() 

        results_panel.place_forget() 

        nn_graph_panel.place_forget() 

        # Restteting entry box and dropdown values
        layers_choice.set(default)

        nodes_choice.set("")

        # Place trainmodel button on frame
        trainModel.place(relx=0.21, rely=0.8, height=40, width=170)

    # Defining function for actions taken when reset button is pressed on logReg page
    def lr_replace(LogReg, log_results_panel, lr_graph_panel, pen_choice, default, tol_choice):

        # Forget buttons, results and graph panel
        LRresetButton.place_forget() 

        log_results_panel.place_forget() 

        lr_graph_panel.place_forget() 

        # Restteting entry box and dropdown values
        pen_choice.set(default) 

        tol_choice.set("")

        # Place button back on frame
        trainModel.place(relx=0.21, rely=0.8, height=40, width=170) 

    # Defining function for removal of predictions on cnn classify image page
    def cnn_class_remove_predictions(self, prediction_bar_plot_panel, predictionLabel):

        prediction_bar_plot_panel.destroy() 

        predictionLabel.destroy() 

    # Defining function for removal of images and label on cnn classify image page
    def cnn_class_remove_image(self, yourImgLabel, panel):

        yourImgLabel.destroy() 

        panel.destroy()
        
    # Defining function for changing buttons when image has been chosen
    def reset_cnn_classification(CNNclassImage):
        
        # Remove choose image button
        chooseImg.place_forget()

        # And replace with reset button
        CNNclassResetButton.place(relx=0.20, rely=0.8, height=40, width=170) 

    # Defining function for resetting buttons on cnn classify image page
    def cnn_class_replace_chooseImage(CNNclassImage):

        # Fsorget the variable defined as 'resetButton'
        CNNclassResetButton.place_forget() 

        # Place choose image button on frame
        chooseImg.place(relx=0.20, rely=0.8, height=40, width=170) 

    # Defining a function for preprocessing new images
    def preprocess(self,file): 

        global chosenImage 

        # Setting standard pixel width/height for all images
        imgSize = 28 

        # Reading image as a graysacel image using cv2 and standardizing
        chosenImage = cv2.imread(file, cv2.IMREAD_GRAYSCALE) / 255.0 

        # Resizing the image to standard size
        chosenImage = cv2.resize(chosenImage, (imgSize, imgSize)) 
        
        # Reshaping image to allow model-prediction
        chosenImage = chosenImage.reshape(-1, imgSize, imgSize, 1)

    # Defining function for running prediction on new image using model that has just been training
    def runModel(self, n_layers, n_nodes): 

        global prediction

        # Defining model path
        model_path = Path.cwd() / 'A4-Image-Classification' / 'output' / f"{n_layers}-dense-{n_nodes}-nodes-CNN.model"

        # Loading in the model from directory
        model = tf.keras.models.load_model(str(model_path)) 

        # Categorizing an image saving and saving the prediction
        prediction = model.predict([chosenImage]) 

    # Defining model for pasting prediction
    def predCalculation(self, prediction): 

        global predictionLabel
        global prediction_bar_plot_panel
        global predictionLabel

        # Flatten prediction to 1d array using list comprehension
        preds = [j for sub in prediction for j in sub] 

        classes = list(range(0,10))

        # Create and save bar plot
        plt.figure(figsize=(8, 5)) 

        sns.barplot(x=classes, y=preds)

        plt.ylabel("Probability")

        plt.xlabel("Class")

        # Call it 'latest-prediction.png'
        plt.savefig(Path.cwd() / 'A4-Image-Classification' / 'output' / "latest-prediction.png")

        # Get prediction 
        number_prediction = preds.index(max(preds))

        # Find model certainty 
        certainty = str(round(max(preds)*100, 2))

        # Create prediction label
        predictionLabel = tk.Label(self, text=f"The image displays the number {number_prediction} \n Certainty: {certainty}%",
                                      relief="flat", background = "white",   
                                    foreground = 'black', width = 14, height = 3, font=("Arial", 13))

        # Placing the prediction label on the frame
        predictionLabel.place(relx=0.6, rely=0.715,relwidth=0.2, anchor='s') 

        # Prininting bar plot showing model certainties 
        # Opening image that has just been created
        prediction_bar_plot = Image.open(Path.cwd() / 'A4-Image-Classification' / 'output' / "latest-prediction.png") 

        prediction_bar_plot = prediction_bar_plot.resize((230, 145)) 

        prediction_bar_plot = ImageTk.PhotoImage(prediction_bar_plot)

        prediction_bar_plot_panel = tk.Label(self, image=prediction_bar_plot) 

        prediction_bar_plot_panel.image = prediction_bar_plot

        # Place label on frame
        prediction_bar_plot_panel.place(relx = 0.485, rely = 0.33) 

    # Defining function for training and evaluating a cnn with the hyperparameters given (only control over the last fully connected part of nn)
    def trainCNN(self, layers_choice, choose_n_nodes):

        global results_panel
        global nn_graph_panel
        global n_layers
        global n_nodes_list
        global n_nodes

        # Retrieve inputs from entry box and dropdown
        n_layers = layers_choice.get() 

        n_nodes = choose_n_nodes.get() 

        # Changing number of layers to integer
        n_layers = int(n_layers)

        # Split input into a list
        n_nodes_list = n_nodes.split(",") 

        # Convert entries to integers
        n_nodes_list = [ int(x) for x in n_nodes_list] 

        # Only run if n_layers is identical to the lenght of the n_nodes list
        if n_layers == len(n_nodes_list): 
            
             # Run cnn model brough in from model script
            cm = models.cnn_mnist.cnn_mnist(n_layers, n_nodes_list)

            # Create and place panel showing prediction results 
            results_panel = tk.Label(self, text =  cm, font=('calibre',9), background='white') 

            results_panel.place(x=320, y=160)

            # Create and place network graph of the fully connected layers using hyperparameters as given by the users
            nn_graph = Image.open(Path.cwd() / 'A4-Image-Classification' / 'output' / f"{n_layers}-dense-{n_nodes}-nodes-CNN-viz.png") 

            nn_graph = nn_graph.resize((232, 232)) 

            nn_graph = ImageTk.PhotoImage(nn_graph) 

            nn_graph_panel = tk.Label(self, image=nn_graph, borderwidth=0)

            nn_graph_panel.image = nn_graph

            # Place label on frame
            nn_graph_panel.place(x=590, y=132) 

            # Run reset_cnn function
            self.reset_cnn()

    # Defining function for training and evaluating a lr with the hyperparameters given by user
    def trainLogReg(self, pen_choice, choose_tol):

        global log_results_panel
        global lr_graph_panel

        # Retrieve inputs from entry box and dropdown
        pen = pen_choice.get() 

        tol = choose_tol.get() 

        # Split input into a list
        tol = float(tol)

        # Run lr model
        cm = models.lr_mnist.lr_mnist(pen, tol) 
        
        # Create panel showing prediction results
        log_results_panel = tk.Label(self, text =  cm, font=('calibre',9), background='white') 

        log_results_panel.place(x=310, y=154)

        # Plot visualisations of most important inputs for each class
        lr_graph = Image.open(Path.cwd() / 'A4-Image-Classification' / 'output' /  f"{pen}-penalty-{tol}-tol-nodes-LR-viz.png")

        lr_graph = lr_graph.resize((370, 235))

        lr_graph = ImageTk.PhotoImage(lr_graph)

        lr_graph_panel = tk.Label(self, image=lr_graph, background="white")

        lr_graph_panel.image = lr_graph

        # Place label on frame
        lr_graph_panel.place(x=525, y=130)

        # Run reset_lr function
        self.reset_lr() 

#-----# Defing StartPage class #-----#

# Defining a class for the start page inheriting functions from tk.Frame
class StartPage(tk.Frame):

     # Have the class self-initiate functions from itself and parent-class
    def __init__(self, parent, controller):
        
        # Initiate a frame for the page
        tk.Frame.__init__(self, parent)

        # Add a background image
        photo = tk.PhotoImage(file = graphics_dir / "bgHome.gif") 

        # Create a label containing the specified photo
        bg = tk.Label(self, image=photo)

        # Placing the bg filling the entire frame
        bg.place(x=0, y=0, relwidth=1, relheight=1) 

        # Use the .image function of tkiner to call the background as the photo
        bg.image = photo 

        # Adding button to access the Log Reg page
        NNmodelButton = tk.Button(self, text="Logistic Regression", font=("Arial Bold", 13),
                        foreground='#326DB2', command=lambda : controller.showFrame(LogReg))
                                
        NNmodelButton.place(relx=0.2, rely=0.8, height=40, width=170)

        # Adding button to access the CNN page
        LRmodelButton = tk.Button(self, text="Neural Network", font=("Arial Bold", 13),
                        foreground='#326DB2', command=lambda : controller.showFrame(CNN))
                                
        LRmodelButton.place(relx=0.64, rely=0.8, height=40, width=170)

#-----# Defining CNN page class #-----#

# Defining class for page for training cnns
class CNN(tk.Frame):

    def __init__(self, parent, controller): 

        global trainLogReg
        global CNNresetButton

        tk.Frame.__init__(self, parent) 

         # Set background
        background = tk.PhotoImage(file= graphics_dir / "bgStandard.gif")
        bg = tk.Label(self, image=background)
        bg.place(x=0, y=0, relwidth=1, relheight=1)
        bg.image = background

        # Add a label as header
        label = tk.Label(self, text="Convolutional Neural Network", relief="flat", background = "#42669C",
                        foreground = 'white', width = 450, height = 2, font=("Arial Bold", 28), highlightcolor="white") 
        
        # Place label and align it at the center 
        label.place(relx=0.286, rely=0.08, width = 450) 

        # Adding white frame in middle of frame
        lowerFrame = tk.Frame(self, bg='white', bd=10, relief='raised', borderwidth = 5) 

        lowerFrame.place(relx=0.5, rely=0.25, relwidth=0.8, relheight=0.5, anchor='n')

        # Add dropdown
        # Number of possible layers in fully connected part
        layer_choices = [1, 2, 3]

        layers_choice = tk.StringVar(self)

        # Setting 1 as default number of fc layers
        default = layer_choices[0]

        choose_n_layers = ttk.OptionMenu(self, layers_choice, default, *layer_choices)
        
        # Add entry box
        nodes_choice = tk.StringVar(self)
        
        # Entry box for defining number of nodes in each layer e.g. 8,4 (number of integers must have same length as number of layers chosen)
        choose_n_nodes = tk.Entry(self, textvariable = nodes_choice, font=('calibre',10,'normal'), background='white')
         
        # Add labels
        dropdown_label = tk.Label(self, text = 'Choose number of layers', font=('calibre',10, 'bold'), background='white')

        entry_label = tk.Label(self, text = 'Choose number of nodes', font=('calibre',10, 'bold'), background='white')

        # Placing stuff
        dropdown_label.place(rely = 0.38, relx = 0.12)

        choose_n_layers.place(rely = 0.42, relx = 0.12)

        entry_label.place(rely = 0.53, relx = 0.12)

        choose_n_nodes.place(rely = 0.57, relx = 0.12)

        # Adding buttons
        trainModel = tk.Button(self, text="Train and Evaluate", font=("Arial Bold", 13),
                               foreground='#326DB2', command=lambda : [controller.trainCNN(layers_choice, choose_n_nodes)])
                                
        trainModel.place(relx=0.21, rely=0.8, height=40, width=170) 

        CNNstartButton = tk.Button(self, text="Start Page", font=("Arial Bold", 13),
                                   foreground='#326DB2', command=lambda : [controller.showFrame(StartPage), controller.cnn_replace(results_panel, nn_graph_panel, layers_choice, default, nodes_choice)])

        CNNstartButton.place(relx=0.62, rely=0.8, height=40, width=170) 

        CNNresetButton = tk.Button(self, text="Reset", font=("Arial Bold", 13),
                                   foreground='#326DB2', command=lambda : [controller.cnn_replace(results_panel, nn_graph_panel, layers_choice, default, nodes_choice)])

        CNNclassImageButton = tk.Button(self, text="Classify New Image", font=("Arial Bold", 13),
                                         foreground='#326DB2', command=lambda : [controller.showFrame(CNNclassImage), controller.cnn_replace(results_panel, nn_graph_panel, layers_choice, default, nodes_choice)])

        CNNclassImageButton.place(relx=0.414, rely=0.8, height=40, width=170)

#-----# Defining Log Reg page class #-----#

# Defining class for page for training a logistic regression
class LogReg(tk.Frame): 

    def __init__(self, parent, controller):

        global trainModel
        global LRresetButton

        tk.Frame.__init__(self, parent)

        # Set background
        background = tk.PhotoImage(file= graphics_dir / "bgStandard.gif") 

        bg = tk.Label(self, image=background)

        bg.place(x=0, y=0, relwidth=1, relheight=1)

        bg.image = background

        # Addiing header
        label = tk.Label(self, text="Logistic Regression", relief="flat", background = "#42669C",
                        foreground = 'white', width = 450, height = 2, font=("Arial Bold", 28), highlightcolor="white")
        
        label.place(relx=0.287, rely=0.08, width = 450)

        # Adding white frame in center
        lowerFrame = tk.Frame(self, bg='white', bd=10, relief='raised', borderwidth = 5) 

        lowerFrame.place(relx=0.5, rely=0.25, relwidth=0.8, relheight=0.5, anchor='n') 

        # Add dropdown
        # Different types of penalty
        pen_choices = ['l1', 'l2', 'elasticnet', 'none']

        pen_choice = tk.StringVar(self)

        default = pen_choices[0]

        choose_pen = ttk.OptionMenu(self, pen_choice, default, *pen_choices)
        
        # Add entry box for tolerance level
        tol_choice = tk.StringVar(self)

        choose_tol = tk.Entry(self, textvariable = tol_choice, font=('calibre',10,'normal'), background='white')

        # Add labels
        dropdown_label = tk.Label(self, text = 'Choose penalty', font=('calibre',12, 'bold'), background='white')

        entry_label = tk.Label(self, text = 'Choose tolerance', font=('calibre',12, 'bold'), background='white')

        # Placing stuff
        dropdown_label.place(rely = 0.38, relx = 0.12)

        choose_pen.place(rely = 0.42, relx = 0.12)

        entry_label.place(rely = 0.53, relx = 0.12)

        choose_tol.place(rely = 0.57, relx = 0.12)

        # Adding buttons
        trainModel = tk.Button(self, text="Train and Evaluate", font=("Arial Bold", 13),
                        foreground='#326DB2', command=lambda : [controller.trainLogReg(pen_choice, choose_tol)])

        trainModel.place(relx=0.21, rely=0.8, height=40, width=170)

        LRstartButton = tk.Button(self, text="Start Page", font=("Arial Bold", 13),
                                   foreground='#326DB2', command=lambda : [controller.showFrame(StartPage), controller.lr_replace(log_results_panel, lr_graph_panel, pen_choice, default, tol_choice)])

        LRstartButton.place(relx=0.59, rely=0.8, height=40, width=170)

        LRresetButton = tk.Button(self, text="Reset", font=("Arial Bold", 13),
                                  foreground='#326DB2', command=lambda : [controller.lr_replace(log_results_panel, lr_graph_panel, pen_choice, default, tol_choice)])

#-----# Defining for classifying new image using CNN #-----#

# Defining class for page for classifying self chosen image using the pre-trained cnn
class CNNclassImage(tk.Frame):
    def __init__(self, parent, controller): 
       
        global chooseImg
        global CNNclassResetButton

        tk.Frame.__init__(self, parent) 

        # Set background
        background = tk.PhotoImage(file=graphics_dir / "bgStandard.gif") 
        bg = tk.Label(self, image=background)
        bg.place(x=0, y=0, relwidth=1, relheight=1)
        bg.image = background

        # Adding header
        label = tk.Label(self, text="Classify New Image", relief="flat", background = "#42669C",
                        foreground = 'white', width = 450, height = 2, font=("Arial Bold", 28), highlightcolor="white")
        label.place(relx=0.32, rely=0.08, width = 350)

        # Adding white frame in middle of frame
        lowerFrame = tk.Frame(self, bg='white', bd=10, relief='raised', borderwidth = 5) 

        lowerFrame.place(relx=0.5, rely=0.25, relwidth=0.6, relheight=0.5, anchor='n')

        # Adding buttons
        chooseImg = tk.Button(self, text="Choose Image", font=("Arial Bold", 13),
                               foreground='#326DB2', command=lambda : [controller.addImage(), controller.openImage()])
                                
        chooseImg.place(relx=0.20, rely=0.8, height=40, width=170)


        classifyButton = tk.Button(self, text="Classify", font=("Arial Bold", 13),
                                   foreground='#326DB2', command=lambda : [controller.preprocess(file), controller.runModel(n_layers, n_nodes), 
                                                controller.predCalculation(prediction), controller.reset_cnn_classification()])
                                
        classifyButton.place(relx = 0.41, rely=0.8, height=40, width=170)

        CNNbackButton = tk.Button(self, text="Go Back", font=("Arial Bold", 13),
                                  foreground='#326DB2', command=lambda : [controller.showFrame(CNN), 
                                                  controller.cnn_class_remove_predictions(prediction_bar_plot_panel, predictionLabel),
                                                  controller.cnn_class_remove_image(yourImgLabel, panel),
                                                  controller.cnn_class_replace_chooseImage()])

        CNNbackButton.place(relx=0.63, rely=0.8, height=40, width=170)

        CNNclassResetButton = tk.Button(self, text="Reset", font=("Arial Bold", 13),
                                        foreground='#326DB2', command=lambda : [controller.removeImage(), controller.removeText(), 
                                          controller.cnn_class_remove_predictions(prediction_bar_plot_panel, predictionLabel),
                                          controller.cnn_class_remove_image(yourImgLabel, panel),
                                          controller.cnn_class_replace_chooseImage()])

 # Define variable 'app' as the MainApp() for running tkinter window
app = MainApp() 

# Call the mainloop() function on app to activate all defined frames
app.mainloop() 


