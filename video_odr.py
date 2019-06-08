#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# In[3]:

tf.logging.set_verbosity(tf.logging.INFO)


# In[4]:


from sklearn.preprocessing import OneHotEncoder


# In[5]:


tf.reset_default_graph()


# In[6]:


# Load training and test data
((train_data, train_labels),(test_data, test_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data/np.float32(255)
test_data = test_data/np.float32(255)

ohe=OneHotEncoder(sparse=False, categories='auto')
train_labels=ohe.fit_transform(train_labels.reshape(-1,1))
test_labels=ohe.fit_transform(test_labels.reshape(-1,1))


# In[7]:


x = tf.placeholder(tf.float32, shape=[None,28,28], name='X')
x_img = tf.reshape(x, [-1,28,28,1])
y_label = tf.placeholder(tf.float32, shape=[None, 10], name='y_label')
y_true = tf.argmax(y_label, axis=1)

rate = tf.placeholder(tf.float32, name='rate')


# # Layer Functions

# In[8]:


def conv_layer(inp, num_inp_channels, filter_size, num_filters):
    shape = [filter_size, filter_size, num_inp_channels, num_filters]
    weights = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[num_filters]))
    layer = tf.nn.conv2d(input=inp, filter=weights, strides=[1,1,1,1], padding='SAME')
    return tf.add(layer, biases)


# In[9]:


def pool_layer(inp):
    layer = tf.nn.max_pool(value=inp, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    return layer


# In[10]:


def fc_layer(inp, num_inps, num_outs):
    weights = tf.Variable(tf.truncated_normal([num_inps, num_outs], stddev=0.1))
    biases = tf.Variable(tf.constant(0.1, shape=[num_outs]))
    return tf.add(tf.matmul(inp,weights), biases)


# # Create CNN

# In[11]:


conv1 = tf.nn.relu(conv_layer(inp=x_img, num_inp_channels=1, filter_size=5, num_filters=6))
pool1 = pool_layer(conv1)
conv2 = tf.nn.relu(conv_layer(inp=pool1, num_inp_channels=6, filter_size=5, num_filters=16))
pool2 = pool_layer(conv2)

num_features = pool2.shape[1:].num_elements()
flat_layer = tf.reshape(pool2, [-1, num_features])

fc1 = tf.nn.relu(fc_layer(flat_layer, num_inps=num_features, num_outs=128))
fc2 = fc_layer(fc1, num_inps=128, num_outs=10)

y_pred = tf.nn.softmax(fc2)
y_pred_val = tf.argmax(y_pred, axis=1)

saver = tf.train.Saver()

import cv2
# In[33]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess,"Model/model")
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        ret, img_th = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY_INV)

        ctrs,hier = cv2.findContours(img_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]

        for rect in rects:
            cv2.rectangle(img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 2) 
            y_val = rect[1]
            x_val = rect[0]
            th_x=rect[3]//8
            th_y=rect[3]//10
            roi = img_th[y_val-th_y:y_val+rect[3]+th_y, x_val-th_x:x_val+rect[2]+th_x]
            try:
                roi = cv2.resize(roi, (20, 20))
                roi = cv2.dilate(roi, (6, 6))
                roi = cv2.copyMakeBorder(roi,4,4,4,4,cv2.BORDER_CONSTANT,(255,255,255))
                roi = (roi.reshape(1,28,28))/np.float32(255)
                feed_dict = {x:roi}
                out = sess.run(y_pred, feed_dict=feed_dict)
                nbr=np.argmax(out[0])
                cv2.putText(img, str(nbr), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1)
            except:
                pass

        cv2.imshow("Results", img)
        if cv2.waitKey(1) & 0xff == 27:
            break
    cam.release()
    cv2.destroyAllWindows()