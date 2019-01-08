import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

name = ["asuka","mai","erika","nanami","nanase"]

# Labeling the training data
train_dir="./Input_Data/train/"
X_train = []
Y_train = []
for i in range(len(name)):
    img_file_name_list=os.listdir(train_dir+name[i])
    print('Found {} training images for {}'.format(len(img_file_name_list), name[i]))
    for j in range(0, len(img_file_name_list)-1):
        n=os.path.join(train_dir+name[i]+"/", img_file_name_list[j])
        img = cv2.imread(n)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        # refactoring the image
        img = np.divide(img, 255)
        X_train.append(img)
        Y_train.append(i)

# Labeling the validation data
validation_dir="./input_data/test/"
X_test = []
Y_test = []
for i in range(len(name)):
    img_file_name_list=os.listdir(validation_dir+name[i])
    print('Found {} testing images for {}'.format(len(img_file_name_list), name[i]))
    for j in range(0, len(img_file_name_list)-1):
        n=os.path.join(validation_dir+name[i]+"/", img_file_name_list[j])
        img=cv2.imread(n)
        b,g,r = cv2.split(img)
        img = cv2.merge([r,g,b])
        # Refactoring the images
        img = np.divide(img, 255)
        X_test.append(img)
        Y_test.append(i)
X_train=np.array(X_train)
X_test=np.array(X_test)

from keras.utils.np_utils import to_categorical
y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

from keras import layers, optimizers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), input_shape=(64,64,3), strides=(1,1), padding='same'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (2, 2), strides=(1,1), padding='same'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32, (2, 2), strides=(1,1), padding='same'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128, (2, 2), strides=(1,1), padding='same'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='sigmoid'))
model.add(layers.Dense(128, activation='sigmoid'))
model.add(layers.Dense(5, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
             optimizer='sgd',
             metrics=['accuracy'])

history = model.fit(X_train, y_train, 
                    batch_size=32,
                    epochs=100,
                    validation_data=(X_test, y_test),
                    verbose=1
                    )

model.save('BasicCNN.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Plot the accuracy
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('BasicCNN_acc.png')

plt.figure()

# Plot the loss value
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('BasicCNN_loss.png')