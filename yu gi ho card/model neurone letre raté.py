import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


mnist=tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test)=mnist.load_data()
x_train=tf.keras.utils.normalize(x_train,axis=1)
x_test=tf.keras.utils.normalize(x_test,axis=1)
x_train=x_train/255
x_test=x_test/255
x_train_flaten=x_train.reshape(len(x_train),28*28)
x_test_flaten=x_test.reshape(len(x_test),28*28)

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(128,activation='relu'))
model.add(tf.keras.layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3)
model.save('yugi')

model=tf.keras.models.load_model('yugi')

loss,accuracy=model.evaluate(x_test,y_test)
print(accuracy)
print(loss)

img=cv2.imread(r"C:\Users\9820937G\OneDrive - SNCF\Bureau\yugi.png")
img=np.array([img])
plt.imshow(img[0])
plt.show()
img=np.resize(img,(1,28,28))
img=img.reshape(len(img),28*28)
img.shape




predictt=model.predict(img)
