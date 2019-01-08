print("=== Importing Libraries ===")
import os
import glob
import math
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras import optimizers

print('=== Preprocessing Data ===')
name = ["asuka","mai","erika","nanami","nanase"]
# Labeling the training data
train_dir="./input_data2/train/"
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
validation_dir="./input_data2/test/"
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

y_train = to_categorical(Y_train)
y_test = to_categorical(Y_test)

print('=== Importing VGGFace Net ===')
conv_base = VGGFace(include_top=False, input_shape=(150, 150, 3))

print('=== Creating Model ===')
#custom parameters
nb_class = 5
hidden_dim = 512

last_layer = conv_base.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(5, activation='softmax', name='fc8')(x)
model = Model(conv_base.input, out)

model.summary()

print("=== Freezing the Convolutional Base ===")
# Freezing the Network
print('This is the number of trainable weights '
      'before freezing the conv base:', len(model.trainable_weights))
  
conv_base.trainable = False
for layer in conv_base.layers:
  layer.trainable = False

print('This is the number of trainable weights '
      'after freezing the conv base:', len(model.trainable_weights))

model.compile(loss='categorical_crossentropy',
             optimizer='sgd',
             metrics=['accuracy'])

print('=== Training the Model ===')
# Using the batch generator to fit the model to the data
history = model.fit(X_train, y_train, 
                    batch_size=29,
                    epochs=50,
                    validation_data=(X_test, y_test),
                    verbose=1
                    )
                  
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

print('=== Plotting Results ===')
# Plot the accuracy
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('./pret_acc.png')

plt.figure()

# Plot the loss value
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('./pret_loss.png')

print('=== Unfreezing Network ===')
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
        if layer.name == 'conv4_1':
                set_trainable = True
        if set_trainable:
                layer.trainable = True
        else:
                layer.trainable = False

print('=== Fine Tuning ===')
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-5), metrics=['acc'])

history = model.fit(X_train, y_train, 
                    batch_size=29,
                    epochs=50,
                    validation_data=(X_test, y_test),
                    verbose=1
                    )

print('=== Plotting Results ===')
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

# Plot the accuracy
plt.figure()

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('./pret_acc_ft.png')

plt.figure()

# Plot the loss value
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('./pret_loss_ft.png')