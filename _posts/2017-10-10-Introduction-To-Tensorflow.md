---
layout: post
title:  "Introduction to Tensorflow!"
date:   2017-10-14 17:10:21 +0530
categories: Tensorflow
---

# Introduction to Tensorflow

- **Installation.**
- **Architecture.**
- **API.**
- **Model Building.**
- **Training & Validation.**
- **Model Deployment.**

In this post we are going to explore the most popular Machine Learning library Tensorflow. we will checkout how to install Tensorflow. we will build a model using it and deploy the model using a plugin called tensorflow-model-server. 
		
- ###Installation: 

Tensorflow can be installed using following commands.

Requirements: Ubuntu 16.04, CUDA9,Cudnn (for GPU support),python=3.6
using pip:
			
			##CPU
			pip install tensorflow
			##GPU
			pip install tensorflow-gpu

using conda
			
			##CPU
			conda install -c anaconda tensorflow
			##GPU
			conda install -c anaconda tensorflow-gpu 

to install the CUDA dn Cudnn for the GPU version of tensorflow to work check [here](entet a URL later). 


### Architecture:

![Architecture image](/assets/layers.png)

the architecture of tensorflow follows the standard Client-Master model. Computations are designed as a dataflow graph in a client language, will be executed as a session using a C API. a distributed master layer will then process the graphs. the processing involves pruning a subgraph from a graph as defined in Session.run(), partitioning the subgraphs into pieces to run on differenct processes and devices, and distribute to worker services.  the worker services will then run the graph pieces on GPU/CPU's using the **Kernal Implementations** layer where the computation for the individual graphs happen. the worker services also communicate with other worker services.

Note that the Distributed Master and Worker Service only exist in distributed TensorFlow. The single-process version of TensorFlow includes a special Session implementation that does everything the distributed master does but only communicates with devices in the local process.

![Client image](/assets/graph_client.svg)


in the above figure, the client creates a session which sends the graph to distributed master. when client evaluates a node, the evaluation triggers the master to initiate the computation. the above graph applies weights w to a feature vector x and adds a bias term and saves them to a variable s. the distributed master even applies standard optimization tehniques to sub graphs such as common  subexpression elimination and constant folding. the distributed master prunes the sub graphs and partition them and cache the peices to use in subsequent steps. the pieces are distributed into nodes and parameters servers.

The **Kernel Implementations** includes standard operations including mathematical, array manipulation, control flow, and state management operations. The kernel implementations are optimized for a variety of devices(CPU,GPU). we can even write a kernel implementation and register them in the tensorflow.

### API:

Tensorflow provides a two kinds of API, one a high level API for easily building and training deep learning and machine learning models and other a low level API to access the core features of Tensorflow and build applications outside high level API.

 
 Tensorflow provides the following High level API:
		
	1. Keras
	2. Eager Execution
	3. Importing Data
	4. Estimators
and the following low levev API

	1. Variables
	2. Tensors
	3. Graph and Sessions
	4. Save and Restore Models
	

### Model Building:

let's create 2 models using high level and low level API.

##### High Level API:
```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(Convolution2D(32, 3, 3, activation='relu', input_shape=(1,28,28)))
model.add(Convolution2D(32, 3, 3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
 
# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
              
# Configure a model for mean-squared error regression.
model.compile(optimizer=tf.train.AdamOptimizer(0.01),
              loss='mse',       # mean squared error
              metrics=['mae'])  # mean absolute error

# Configure a model for categorical classification.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),
              loss=keras.losses.categorical_crossentropy,
              metrics=[keras.metrics.categorical_accuracy])
              
import numpy as np

# Load pre-shuffled MNIST data into train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()
 
#  Preprocess input data
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
 
#  Preprocess class labels
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
dataset = dataset.batch(32).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
val_dataset = val_dataset.batch(32).repeat()

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)
```

**Evaluate and predict:**
The tf.keras.Model.evaluate and tf.keras.Model.predict methods can use NumPy data and a tf.data.Dataset.

To evaluate the inference-mode loss and metrics for the data provided:
```python
model.evaluate(x, y, batch_size=32)

model.evaluate(dataset, steps=30)
And to predict the output of the last layer in inference for the data provided, as a NumPy array:

model.predict(x, batch_size=32)

model.predict(dataset, steps=30)
```

**Callbacks:**

```python
callbacks = [
  # Interrupt training if `val_loss` stops improving for over 2 epochs
  keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
  # Write TensorBoard logs to `./logs` directory
  keras.callbacks.TensorBoard(log_dir='./logs')
]
model.fit(X_train, Y_train, batch_size=32, epochs=5, callbacks=callbacks,
          validation_data=(val_data, val_targets))
```

By default, this saves the model's weights in the TensorFlow checkpoint file format. Weights can also be saved to the Keras HDF5 format (the default for the multi-backend implementation of Keras):
```python
# Save weights to a HDF5 file
model.save_weights('my_model.h5', save_format='h5')

# Restore the model's state
model.load_weights('my_model.h5')
```

**Entire model:**
The entire model can be saved to a file that contains the weight values, the model's configuration, and even the optimizer's configuration. This allows you to checkpoint a model and resume training later—from the exact same state—without access to the original code.
```python
# Create a trivial model
model = keras.Sequential([
  keras.layers.Dense(10, activation='softmax', input_shape=(32,)),
  keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, targets, batch_size=32, epochs=5)


# Save entire model to a HDF5 file
model.save('my_model.h5')

# Recreate the exact same model, including weights and optimizer.
model = keras.models.load_model('my_model.h5')
```

**Estimators:**
The Estimators API is used for training models for distributed environments. This targets industry use cases such as distributed training on large datasets that can export a model for production.
```python
A tf.keras.Model can be trained with the tf.estimator API by converting the model to an tf.estimator.Estimator object with tf.keras.estimator.model_to_estimator. See Creating Estimators from Keras models.

model = keras.Sequential([layers.Dense(10,activation='softmax'),
                          layers.Dense(10,activation='softmax')])

model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

estimator = keras.estimator.model_to_estimator(model)
```

### Model Deployment:

we can use Tensorflow plugin tensorflow-model-serving to deploy the models in a scalable and efficient way. the deployed models can be accessed as a REST API. a full tutorial can be found [here](write something here)