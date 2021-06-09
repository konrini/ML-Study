#!/usr/bin/env python
# coding: utf-8

# In[5]:


from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


# In[7]:


import tensorflow as tf


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


# In[9]:


print(x_train)
print(y_train)


# In[10]:


print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[18]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import numpy as np

model = Sequential()
model.add(LSTM(16, return_sequences=True, input_shape = (28, 28)))
model.add(LSTM(128))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.summary()


# In[19]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[21]:


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

MODEL_DIR = './mnist_rnn/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath = MODEL_DIR + "{epoch:02d}-{val_loss:.4f}.hdf5"
checkpointer = ModelCheckpoint(filepath=modelpath, monitor='val_loss',
                              verbose=1, save_best_only=True)
esc = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(x_train, y_train, epochs=100, batch_size=300,
                    validation_split=0.2, callbacks=[esc, checkpointer], verbose=1)

print(model.evaluate(x_test, y_test))


# In[22]:


import matplotlib.pyplot as plt
y_vloss = history.history['val_loss']
y_loss = history.history['loss']
x_len = np.arange(len(y_vloss))
plt.plot(x_len, y_vloss, marker='.', c='red', label='testset')
plt.plot(x_len, y_loss, marker='.', c='blue', label='trainset')
plt.legend(loc='best')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# In[24]:


from tensorflow.keras.models import load_model
model = load_model('./mnist_rnn/22-0.0657.hdf5')


# In[37]:


y_hat = model.predict(x_test)
y_hatt = np.argmax(y_hat, axis=1)
y_testt = np.argmax(y_test, axis=1)
plt.plot(range(30), y_testt[70:100], 'ro-', label='real')
plt.plot(range(30), y_hatt[70:100], 'bs--', label='predict')
plt.legend(loc='best')
plt.show()

