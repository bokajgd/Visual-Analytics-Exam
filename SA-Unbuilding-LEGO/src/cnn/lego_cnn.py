#-----# Importing packages #-----#
# Import packages
import numpy as np # Matrix maths
import tensorflow as tf # NN functions
import matplotlib.pyplot as plt # For drawing graph
from sklearn.metrics import classification_report
import keras

from pathlib import Path

#-----# Project desctiption #-----#

# Defining main function
def main():
    
    # Instantiate class
    cnn = legoCNN()

# Defining class
class legoCNN:
    def __init__(self):
        
        # Setting output directory
        self.out_dir = self.setting_output_directory()

        # Preproces data
        self.preprocess_data()  

         # Build model  
        self.build_model()

        # Train and evaluate model
        self.train_and_evaluate() 

        # Create plot over model learning history
        self.plot_history()

        return
    
    #-----# Utility functions #-----#

    # Defining function for setting directory for the raw data
    def setting_output_directory(self):

        root_dir = Path.cwd()  # Setting root directory

        out_dir = root_dir / 'output' / 'model_outputs' # Setting data directory

        return out_dir

    def preprocess_data(self):
        # Setting model output directory 
        self.model_out_dir = Path.cwd()  / 'data' 

        # Setting model data directory 
        self.model_data_dir = Path.cwd() / 'data' 


        self.train_data = tf.keras.preprocessing.image_dataset_from_directory(self.model_data_dir / 'train',
                                                                            image_size=(132, 132),
                                                                            batch_size=16)

        self.test_data = tf.keras.preprocessing.image_dataset_from_directory(self.model_data_dir / 'test',
                                                                          image_size=(132, 132),
                                                                          batch_size=16)

        # Gettiing names of each class (brick names)
        self.train_class_names = self.train_data.class_names

        train_classes = open(str(self.out_dir) + "/train_classes.txt", "w")

        train_classes.write(str(self.train_class_names))
        
        train_classes.close()
        
        self.test_class_names = self.test_data.class_names

        test_classes = open(str(self.out_dir) + "/test_classes.txt", "w")

        test_classes.write(str(self.test_class_names))
        
        test_classes.close()

        # Number of classes
        self.num_classes = len(self.train_class_names)


    # Defining cnn in a single function
    def build_model(self):
        
        # Image input shape
        input_shape = (132, 132, 3)

        # Build the model
        self.model = keras.models.Sequential()

        self.model.add(keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=input_shape))

        self.model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))

        self.model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

        self.model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        self.model.add(keras.layers.Dropout(0.25))

        self.model.add(keras.layers.Flatten())

        self.model.add(keras.layers.Dense(32, activation='relu'))

        self.model.add(keras.layers.Dense(16, activation='relu'))

        self.model.add(keras.layers.Dense(self.num_classes, activation='softmax'))

        # Compile the layers into one model
        # Loss function and optimizer needed (using SparseCategoricalCrossentropy and Adam)
        self.model.compile(
            optimizer = 'adam',
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # This allows more than two classes
            metrics = ['accuracy']
        )

        print(self.model.summary())

    # Defining function for trainings
    def train_and_evaluate(self):
         
        self.epochs = 2
        
        # Train the model
        self.history = self.model.fit(
            self.train_data,
            validation_data=self.test_data, 
            epochs = self.epochs # Number of iterations over the entire training dataset
        )

        # Saving model
        self.model.save(self.out_dir / f"lego-CNN.model") 

        # Evaluating model
        loss, acc = self.model.evaluate(self.test_data, verbose=1)

        # Print loss and accuracy
        print("loss: %.2f" % loss)

        print("acc: %.2f" % acc)

    #Function for plotting the models performance
    def plot_history(self):
        
        # Visualize training history
        plt.style.use("fivethirtyeight")

        fig = plt.figure()

        plt.plot(np.arange(0, self.epochs), self.history.history["loss"], label="train_loss")

        plt.plot(np.arange(0, self.epochs), self.history.history["val_loss"], label="val_loss")

        plt.plot(np.arange(0, self.epochs), self.history.history["accuracy"], label="train_acc")
        
        plt.plot(np.arange(0, self.epochs), self.history.history["val_accuracy"], label="val_acc")
        
        plt.title("Training Loss and Accuracy")
        
        plt.xlabel("Epoch #")
        
        plt.ylabel("Loss/Accuracy")
        
        plt.legend()
        
        plt.tight_layout()
        out_file = self.out_dir / "train_val_history.png"

        plt.savefig(out_file)


# Executing main function when script is run
if __name__ == '__main__':   
    main()