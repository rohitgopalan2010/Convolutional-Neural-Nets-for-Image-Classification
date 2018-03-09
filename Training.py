"""
Created on Thu Nov 30 16:25:13 2017

@author: Rohit
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from tensorflow.python.saved_model import builder as saved_model_builder
import numpy as np
import cv2
import os
import tensorflow as tf
import shutil

sess = tf.Session()

# two dictionaries that map integers to images, i.e., 2D numpy array.
TRAIN_IMAGE_DATA = {}
TEST_IMAGE_DATA1 = {}
TEST_IMAGE_DATA2= {}


# the train target is an array of 1's
TRAIN_TARGET = []
# the set target is an array of 0's.
TEST_TARGET1  = []

# the set target is an array of 0's.
TEST_TARGET2  = []

### Global counters for train and test samples
NUM_TRAIN_SAMPLES = 0
NUM_TEST_SAMPLES  = 0

## define the root directory 
ROOT_DIR = 'C:/Users/rohit/Downloads/nn_train/nn_train/'

## read the single bee train images
YES_BEE_TRAIN = ROOT_DIR + 'single_bee_train'

for root, dirs, files in os.walk(YES_BEE_TRAIN):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TRAIN_IMAGE_DATA[NUM_TRAIN_SAMPLES] = img
            TRAIN_TARGET.append(int(1))
        NUM_TRAIN_SAMPLES +=1

## read the single bee test images
YES_BEE_TEST = ROOT_DIR + 'single_bee_test'

for root, dirs, files in os.walk(YES_BEE_TEST):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            # print img.shape
            TEST_IMAGE_DATA1[NUM_TEST_SAMPLES] = img
            TEST_TARGET1.append(int(1))
        NUM_TEST_SAMPLES += 1

## read the no-bee train images
NO_BEE_TRAIN = ROOT_DIR + 'no_bee_train'

for root, dirs, files in os.walk(NO_BEE_TRAIN):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TRAIN_IMAGE_DATA[NUM_TRAIN_SAMPLES] = img
            TRAIN_TARGET.append(int(0))
        NUM_TRAIN_SAMPLES += 1
        
# read the no-bee test images
NO_BEE_TEST = ROOT_DIR + 'no_bee_test'

for root, dirs, files in os.walk(NO_BEE_TEST):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TEST_IMAGE_DATA2[NUM_TEST_SAMPLES] = img
            TEST_TARGET2.append(int(0))
        NUM_TEST_SAMPLES += 1

train1 = TRAIN_IMAGE_DATA.values()
test1 = TEST_IMAGE_DATA1.values()
test2= TEST_IMAGE_DATA2.values()

N_DIGITS = 2  # Number of digits.
X_FEATURE = 'x'  # Name of the input feature.

def conv_model(features, labels, mode):
  # Reshape feature
  feature = tf.reshape(features[X_FEATURE], [-1, 32, 32, 3])
  # First conv layer will compute 32 features for each 5x5 patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = tf.layers.conv2d(
        feature,
        filters=32,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    h_pool1 = tf.layers.max_pooling2d(
        h_conv1, pool_size=2, strides=2, padding='same')
    
  # Second conv layer will compute 64 features for each 5x5 patch.
  with tf.variable_scope('conv_layer2'):
    h_conv2 = tf.layers.conv2d(
        h_pool1,
        filters=64,
        kernel_size=[5, 5],
        padding='same',
        activation=tf.nn.relu)
    h_pool2 = tf.layers.max_pooling2d(
        h_conv2, pool_size=2, strides=2, padding='same')
    
    #reshape tensor into a batch of vectors
    h_pool2_flat = tf.contrib.layers.flatten(h_pool2)
    
  #Densely connected layer with 1024 neurons.
  h_fc1 = tf.layers.dense(h_pool2_flat, 1024, activation=tf.nn.relu)
  if mode == tf.estimator.ModeKeys.TRAIN:
    h_fc1 = tf.layers.dropout(h_fc1, rate=0.5)
    
  #Compute logits (1 per class) and compute loss.
  logits = tf.layers.dense(h_fc1, N_DIGITS, activation=None)
  
  #Compute predictions.
  predicted_classes = tf.argmax(logits, 1)
  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {
        'class': predicted_classes,
        'prob': tf.nn.softmax(logits)
    }
    return tf.estimator.EstimatorSpec(mode, predictions=predictions)

  #Compute loss.
  onehot_labels = tf.one_hot(tf.cast(labels, tf.int32), N_DIGITS, 1, 0)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)
  
  # Create training op.
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

  #Compute evaluation metrics.
  eval_metric_ops = {
      'accuracy': tf.metrics.accuracy(
          labels=labels, predictions=predicted_classes)
  }
  return tf.estimator.EstimatorSpec(
      mode, loss=loss, eval_metric_ops=eval_metric_ops)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
x={X_FEATURE: np.array(list(train1)).astype(np.float32)},
y=np.array(TRAIN_TARGET).astype(np.int32),
batch_size=100,
num_epochs=None,
shuffle=True)

test_input_fn1 = tf.estimator.inputs.numpy_input_fn(
x={X_FEATURE: np.array(list(test1)).astype(np.float32)},
y=np.array(TEST_TARGET1).astype(np.int32),
num_epochs=1,
shuffle=False)

test_input_fn2 = tf.estimator.inputs.numpy_input_fn(
x={X_FEATURE: np.array(list(test2)).astype(np.float32)},
y=np.array(TEST_TARGET2).astype(np.int32),
num_epochs=1,
shuffle=False)

classifier = tf.estimator.Estimator(model_fn=conv_model)
classifier.train(input_fn=train_input_fn, steps=50000)
scores1 = (classifier.evaluate(input_fn=test_input_fn1))

scores2 = (classifier.evaluate(input_fn=test_input_fn2))

print('Yes-bee Accuracy (conv_model): {0:f}'.format(scores1['accuracy']))
print('No- Bee Accuracy (conv_model): {0:f}'.format(scores2['accuracy']))


init = tf.global_variables_initializer()
sess.run(init)

tf.add_to_collection("nn_model", classifier)

# Add ops to save and restore all the variables

'''try:
    shutil.rmtree("model")
except:
   pass

builder = saved_model_builder.SavedModelBuilder("model")
builder.add_meta_graph_and_variables(sess, ["nn"])
builder.save()'''

print("Model saved in file")

import pickle
with open("C:/Users/rohit/Downloads/nn_train/nn_train/finalmodel.pkl",'wb') as f:
    pickle.dump(classifier,f)
    
