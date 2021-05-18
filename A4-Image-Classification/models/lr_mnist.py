# Importing packages
import sys,os
import numpy as np
from pathlib import Path
import argparse
# Import sklearn stuff
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Import utility functions definied in utils.py
from models.model_utils.utils import plot_coefs

# Defining logistic regression in a single function
def lr_mnist(pen, tol):
    # Setting model output directory 
    model_out_dir = Path.cwd() / 'W7-Image-Classification' / 'model_out' 

    X, y = fetch_openml('mnist_784', version=1, return_X_y=True) # Load data
    
    # Convert to arrays
    X = np.array(X) 
    y = np.array(y)
    
    #Predifine classes and number of classes
    classes = sorted(set(y))
    n_classes = len(classes)
    
    # Splitting data
    X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    train_size=7500, 
                                                    test_size=2500)
    
    # Scaling the features
    X_train_scaled = X_train/255.0
    X_test_scaled = X_test/255.0

    # Training the model
    clf = LogisticRegression(penalty=pen, 
                         tol=tol, 
                         solver='saga',
                         multi_class='multinomial').fit(X_train_scaled, y_train)

    # Evaluating the model
    y_pred = clf.predict(X_test_scaled)

    # Getting metrics 
    cm = metrics.classification_report(y_test, y_pred)

    # Creating image of coefs
    plot_coefs(clf.coef_, n_classes, model_out_dir / f"{pen}-penalty-{tol}-tol-nodes-LR-viz.png")

    return cm
