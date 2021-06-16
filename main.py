#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.utils import to_categorical
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.datasets import load_digits
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import load_model
from statistics import mean
from sympy import sympify
from time import sleep
from sympy import Symbol, sympify, factor, plot, solve, sin, Limit, Derivative, init_printing
from sympy import Integral, log
from collections import deque
from predict import predictions





# Creating a dictionary for 20 Devangiri characters as value and defining labels as their Key.


# Judges evaluatuon criterian code
def test():
    '''
    We will be using a similar template to test your code
    '''
    image_paths = [r"C:\Users\ASUS Vivobook\OneDrive\Desktop\Code\MOSAIC\Test images\pilot.jpg",r"C:\Users\ASUS Vivobook\OneDrive\Desktop\Code\MOSAIC\Test images\kamal_1.jpg",r"C:\Users\ASUS Vivobook\OneDrive\Desktop\Code\MOSAIC\Test images\bharat_1.jpg"]



    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path) # This input format wont change
        answer = predictions(image) # a list is expected
        print(''.join(answer))# will be the output string




    print('The final score of the participant is')
# In[294]:
# lets begin!!
if __name__ == "__main__":

    test()




# In[ ]:
