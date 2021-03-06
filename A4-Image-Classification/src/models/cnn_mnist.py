# Classifying the MNIST handwritten digits (0-9)

# Import packages
import matplotlib.pyplot as plt # For drawing graph
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report
import keras
from pathlib import Path

# Import utility functions definied in utils.py
from models.model_utils.utils import draw_neural_net

# Defining cnn in a single function - the function takes arguments 'n_layers' and 'n_nodes' which are provided by GUI user
def cnn_mnist(n_layers, n_nodes):

    # Setting model output directory 
    model_out_dir = Path.cwd() / 'output' 

    # Load data
    mnist = keras.datasets.mnist

    # Split data into train and test
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizing the images / normalize the pixels values to values between [-0.5 ; 0.5]
    # This gives best conditions for parameters optimisation
    x_train = (x_train/255) - 0.5 

    x_test = (x_test/255) - 0.5

    # Flatting the images / Flatten from 28x28 matrix to 1x784 dimensional column vector
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    input_shape = (28, 28, 1)

    # Build the convolutional layers of the model + pooling layers and dropout
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                                input_shape=input_shape))

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(keras.layers.Dropout(0.25))

    model.add(keras.layers.Flatten())

    # Let user define hyperparams of fully connected layers
    # Add dense layers with the dimensions according to input by user
    for layer in range(n_layers):

        model.add(keras.layers.Dense(n_nodes[layer], activation='relu'))

    model.add(keras.layers.Dense(10, activation='softmax'))

    # Compile the layers into one model
    # Loss function and optimizer needed
    model.compile(
        optimizer = 'adam',
        loss = 'categorical_crossentropy', # This allows more than two classes
        metrics = ['accuracy']
    )

    # Train the model
    model.fit(
        x_train,
        to_categorical(y_train), # E.g. 2 is transformed into [0, 0, 1, 0, ... 0]
        epochs = 3, # Number of iterations over the entire training dataset
        batch_size = 64 # Number of samples per gradient update for training
    )

    # Evaluate the models
    model.evaluate(
        x_test,
        to_categorical(y_test)
    )

    # Saving evaluation metrics
    predictions = model.predict(x_test)

    predictions = predictions.argmax(axis=1)

    cm = classification_report(to_categorical(y_test).argmax(axis=1), predictions)

    # Saving figure
    nodes_str = str(n_nodes).replace('[','').replace(']','').replace(' ', '')

    model.save(model_out_dir / f"{n_layers}-dense-{nodes_str}-nodes-CNN.model") 

    # Visualising network

    # Styling title
    title = {'family': 'serif',
            'color': '#4A5B65',
            'weight': 'normal',
            'size': 24,
            }

    # Creating full network structure by adding the dimension of the output layer
    network_structure = n_nodes + [10] 
    
    # Creating figure
    fig = plt.figure(figsize=(7, 7)) 

    ax = fig.gca()

    ax.axis('off')

     # Running visualisation function
    draw_neural_net(ax, .1, .9, .1, .9, network_structure)
     
    # Setting unique titlie
    ax.set_title(f"{n_layers}-dense-{n_nodes}-nodes-CNN", fontdict = title, y = 0.9)

    # Saving figure
    plt.savefig(model_out_dir / f"{n_layers}-dense-{nodes_str}-nodes-CNN-viz.png") 

    # Return results
    return cm


