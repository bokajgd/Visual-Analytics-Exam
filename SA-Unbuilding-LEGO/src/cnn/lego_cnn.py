# Classifying impressionist paintings by painter

# Import packages
import numpy as np # Matrix maths
import tensorflow as tf # NN functions
import matplotlib.pyplot as plt # For drawing graph
from sklearn.metrics import classification_report
import keras

from pathlib import Path


# Defining main function
def main():
    
    # Instantiate class
    cnn = legoCNN()

    cnn.preprocess_data()  # Preproces data
    cnn.build_model() # Build model  
    cnn.train_and_evaluate() # Train and evaluate model

# Defining class
class legoCNN:
    def __init__(self):
        return

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

        # Preprocessing data for evaluation and classification report
        self.test_images = np.concatenate([images for images, labels in self.test_data], axis=0)

        self.test_labels = np.concatenate([labels for images, labels in self.test_data], axis=0)

        # Gettiing names of each class (brick names)
        self.train_class_names = self.train_data.class_names

        self.test_class_names = self.test_data.class_names

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
        # Train the model
        self.model.fit(
            self.train_data,
            validation_data=self.test_data, 
            epochs = 4, # Number of iterations over the entire training dataset
        )

        # Saving evaluation metrics
        predictions = self.model.predict(self.test_images, batch_size=16)
        eval_report = classification_report(self.test_labels,
                                            predictions.argmax(axis=1),
                                            target_names=self.val_class_names)

        print(eval_report)

# Executing main function when script is run
if __name__ == '__main__':   
    main()