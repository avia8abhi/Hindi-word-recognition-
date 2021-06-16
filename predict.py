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
from keras.models import model_from_json

# Images from test() functions are fed here processed,manupulated and finally predicted
def predictions(image):
    #Loading previously trained model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    cv2.imshow(" image",image)
    # converting image from RGB to gray scale
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Removing unnecessary noise from the image by using kernel of size 5*5
    blur = cv2.GaussianBlur(gray,(5,5),0)

# Adaptive thresholding of image before sending it to further manupulation
    thresh =  cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, blockSize = 321, C = 38)
    count=0
    t=[]
    # Loops overs each column element of thresh image row to find out for pixel intensity 255(white) inorder to find for a row with maximum white pixel intensity
    #that will in turn be the header of our hindi word.
    for i in [*range(thresh.shape[0])]:

        for j in [*range(thresh.shape[1])]:
            if thresh[i,j]==255 :
                count+=1
            else:
                continue
        #keeps count for number of element with pixel int 255 has occured
        t.append(count)
        count=0

# gives out max value and index of a row with max white pixel intensity
    max_value = max(t)

    max_index = t.index(max_value)
# Reassigning 255 pixel value to 0 to remove the header which would be used further to find perfect contours and thus bounding box
#keeping range from max_index-8 to max_index+12 for replacing rows with 0 and thus removing header , the values can be changed according to the test image we..
    thresh[max_index-8:max_index+15,:]=0



    cv2.imshow("header removed ",thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    #################      Now finding Contours         ###################
# Finding contours from the updated image thresh
    contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
# Developing bounding box and doing simultaneous extraction of images in bounding box
    samples =  np.empty((0,1024))
    responses = []
    keys = [i for i in range(48,58)]

    key=0
    # init' 2D Array to collect the images detected in a bounding box
    A=np.zeros((6,1024))


    j=0
    #looping thorugh countours
    for cnt in contours:
# minimum value of cv2.contourArea(cnt ) can be varied further to get the best bounding box
        if cv2.contourArea(cnt)>150:
                [x,y,w,h] = cv2.boundingRect(cnt)

                if  h>28:
                    #j will be incremented as soon as "Enter" is pressed!
                    j+=1
                    cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
                    #extracted image in bounding box
                    roi = thresh[y:y+h,x:x+w]

                    # image resize to 32*32
                    roismall = cv2.resize(roi,(32,32))

                    cv2.imshow('norm',image)
                    key = cv2.waitKey(0)
                    # inserting images into A

                    A[j-1,:]=np.reshape(roismall,(1,1024))


                elif key == 27:  # (escape to quit)
                    sys.exit()
                elif key in keys:
                    responses.append(int(chr(key)))
                    sample = roismall.reshape((1,1024))

        print(j)

    print ("Extraction Complete!")


    #reshaping from single row to 2D Array of images
    testing_A = np.reshape(A, (A.shape[0],32,32))

    # Reshaping and adding extra layer in every images to make it capable of fitting in CNN Model
    testing_A = testing_A.reshape(testing_A.shape[0],testing_A.shape[1],testing_A.shape[2],1)
    # setting visualization for extracted images in a bounding box
    fig, axes = plt.subplots(3,3, figsize=(8,9))
    axes = axes.flatten()
    predictions=[]
    a=[]
    # feeding model with testing_A for prediction of labels
    predictions = loaded_model.predict(testing_A)
    z={1:'क',2:'ौ',3: 'भ',4: 'च',5: 'ड',6: 'ग',7: 'घ',8: 'ज्ञ',9: 'ह',10: 'क',11:'ल',12:'म',13:'प',14:'फ',15: 'र',16:'स',17: 'त',18: 'ट',19: 'व',20:'य'}
    #d={1:'aa',2:'auu',3: 'bha',4: 'cha',5: 'da',6: 'ga',7: 'gha',8: 'gnya',9: 'ha',10: 'ka',11:'la',12:'ma',13:'pa',14:'pha',15: 'ra',16:'sa',17: 'ta',18: 'tta',19: 'va',20:'ya'}
    for i,ax in enumerate(axes):
        print(i)
        if i<testing_A.shape[0]:
            print(testing_A.shape)
            img = np.reshape(testing_A[i], (32,32))

            ax.imshow(img, cmap="Greys")
        else:
            break
        # index corresponding to max value is extracted
        pred = np.argmax(predictions[i])
        # appending corresponding "value" for the key pred
        a.append(z[pred])
        ax.set_title(z[pred])
        ax.grid()
# since it starts from detecting from right most side image predicted should be reversed
    a.reverse()

    #Final answer::
    return a
