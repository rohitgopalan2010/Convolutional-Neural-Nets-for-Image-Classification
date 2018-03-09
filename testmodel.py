# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 23:52:51 2017

@author: rohit
"""
from tensorflow.python.saved_model import builder as saved_model_builder
import numpy as np
import cv2
import os
import tensorflow as tf
#import shutil
    
def testNet(netpath, dirpath):

    
    TEST_IMAGE_DATA2 = {}
    TEST_TARGET2 = []
    
    #sess = tf.Session()
    
    NO_BEE_TEST = dirpath
    NUM_TEST_SAMPLES=0
    for root, dirs, files in os.walk(NO_BEE_TEST):
        for item in files:
            if item.endswith('.png'):
                ip = os.path.join(root, item)
                img = (cv2.imread(ip)/float(255))
                TEST_IMAGE_DATA2[NUM_TEST_SAMPLES] = img
                TEST_TARGET2.append(int(0))
            NUM_TEST_SAMPLES += 1
    Image=np.array(list(TEST_IMAGE_DATA2.values()))
    Labels=np.array(TEST_TARGET2)
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
    
    
    #classifier = tf.estimator.Estimator(model_fn=conv_model)
    import pickle
    with open(netpath,'rb') as f:
        classifier1 = pickle.load(f)
    
    
    test_input_fn2 = tf.estimator.inputs.numpy_input_fn(
    x={X_FEATURE: np.array(list(Image)).astype(np.float32)},
    y=np.array(Labels).astype(np.int32),
    num_epochs=1,
    shuffle=False)
    scores1 = (classifier1.evaluate(input_fn=test_input_fn2))
    return scores1

x=testNet("C:/Users/rohit/Downloads/nn_train/nn_train/finalmodel.pkl", "C:/Users/rohit/Downloads/nn_train/nn_train/single_bee_test")
print (x)


