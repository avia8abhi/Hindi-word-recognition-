# Creating a dictionary for 20 Devangiri characters as value and defining labels as their Key
d={1:'aa',2:'auu',3: 'bha',4: 'cha',5: 'da',6: 'ga',7: 'gha',8: 'gnya',9: 'ha',10: 'ka',11:'la',12:'ma',13:'pa',14:'pha',15: 'ra',16:'sa',17: 'ta',18: 'tta',19: 'va',20:'ya'}

# Creating 2D Array of zeros of X & y to be used to insert training images and their corresponding labels
X=np.zeros((30610,1024))
y=np.zeros((30610,1))

# Function load_images will load every images from each of the folders named under their devangiri characters
def load_image_process(folder):
    count=0
    k=0
    images = []
    for filename in os.listdir(folder):
        k+=1

        for i in os.listdir(os.path.join(folder,filename)):

            img=cv2.imread(os.path.join(folder,filename,i),cv2.IMREAD_GRAYSCALE)

            if img is not None:
                images.append(img)
                X[count,:]=np.reshape(images[count],(1,1024))
                y[count,0]=k


                count+=1

            else:
                continue


# splitting inserted images into training and test set to get accuracy for unseen data (20% of the training set will split into test set)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size = 0.2)

    #reshaping from single row to 2D Array of images
    print(train_x.shape[0])

    train_x = np.reshape(train_x, (train_x.shape[0], 32,32))
    test_x = np.reshape(test_x, (test_x.shape[0], 32,32))
    print(train_x.shape[0])
    # Reshaping and adding extra layer in every images to make it capable of fitting in CNN Model
    train_X = train_x.reshape(train_x.shape[0],train_x.shape[1],train_x.shape[2],1)
    print("New shape of train data: ", train_X.shape)
    test_X = test_x.reshape(test_x.shape[0], test_x.shape[1], test_x.shape[2],1)
    print("New shape of tra data: ", test_X.shape)


# visualization of random shuffled training images just to make sure evrything is going rigth

    shuff = shuffle(train_x[:100])
    fig, ax = plt.subplots(3,3, figsize = (10,10))
    axes = ax.flatten()
    for i in range(9):
        _, shu = cv2.threshold(shuff[i], 30, 200, cv2.THRESH_BINARY)
        axes[i].imshow(np.reshape(shuff[i], (32,32)), cmap="Greys")
    plt.show()

# Converting labels of training and test images into categorical values so as to fit in CNN Model

    train_yOHE = to_categorical(train_y, num_classes = 21, dtype='int')
    print("New shape of train labels: ", train_yOHE.shape)
    test_yOHE = to_categorical(test_y, num_classes = 21, dtype='int')
    print("New shape of test labels: ", test_yOHE.shape)

    return train_X,test_X,train_yOHE,test_yOHE
# In[290]:
# Defininf Model :: We iterated thorugh various combination of filters size , kernel size & strides and at the end cam up with this optimized combination
# wherein the model has 3 Convulation layer along with 3 corresponding MaxPool layer followed by a flattening and dense layer
def learn_model(train_X,test_X,train_yOHE,test_yOHE):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32,32,1)))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'))
    model.add(MaxPool2D(pool_size=(2, 2), strides=2))
    model.add(Flatten())
    model.add(Dense(64,activation ="relu"))
    model.add(Dense(128,activation ="relu"))
    model.add(Dense(21,activation ="softmax"))


# Inserting our training sets into model to begin learning
    model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(train_X, train_yOHE, epochs=1,  validation_data = (test_X,test_yOHE))

# Brief summary of our model
    model.summary()
#printing all the accuracies we achieved so far
    print("The validation accuracy is :", history.history['val_accuracy'])
    print("The training accuracy is :", history.history['accuracy'])
    print("The validation loss is :", history.history['val_loss'])
    print("The training loss is :", history.history['loss'])
