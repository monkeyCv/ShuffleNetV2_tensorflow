# -*- coding: utf-8 -*-
"""
Created on Mon Aug 20 20:07:23 2018

@author: Yuxi1989
"""

import tensorflow as tf

def unit_c(l,cns,is_training,regular,initial,idx):#NHWC
    with tf.variable_scope('unit_c{}'.format(idx)):
        left,right=tf.split(l,2,axis=3)
        s=left.get_shape().as_list()
        H=s[1]
        W=s[2]
        left_c=s[3]
        conv1=tf.layers.conv2d(right,cns//2,1,padding='SAME',
                               kernel_initializer=initial,
                               kernel_regularizer=regular)
        bn1=tf.layers.batch_normalization(conv1,training=is_training)
        relu1=tf.nn.relu(bn1)
        depth_conv_kernel=tf.get_variable('dconv',shape=[3,3,cns//2,1],initializer=initial,regularizer=regular)
        conv2=tf.nn.depthwise_conv2d(relu1,filter=depth_conv_kernel,
                                     strides=[1,1,1,1],padding='SAME')
        bn2=tf.layers.batch_normalization(conv2,training=is_training)
        conv3=tf.layers.conv2d(bn2,cns-left_c,1,padding='SAME',
                               kernel_initializer=initial,
                               kernel_regularizer=regular)
        bn3=tf.layers.batch_normalization(conv3,training=is_training)
        relu3=tf.nn.relu(bn3)
        out=tf.concat([left,relu3],axis=3)
        out=tf.reshape(out,shape=[-1,H,W,cns//2,2])#shuffle
        out=tf.transpose(out,perm=[0,1,2,4,3])
        out=tf.reshape(out,shape=[-1,H,W,cns])
        return out
    
def unit_d(l,cns,is_training,regular,initial):
    with tf.variable_scope('unit_d'):
        left,right=l,l
        s=l.get_shape().as_list()
        H=s[1]//2
        W=s[2]//2
        depth_conv_kernel1=tf.get_variable('dconv1',shape=[3,3,cns//2,1],initializer=initial,regularizer=regular)
        lconv1=tf.nn.depthwise_conv2d(left,filter=depth_conv_kernel1,
                                      strides=[1,2,2,1],padding='SAME')
        lbn1=tf.layers.batch_normalization(lconv1,training=is_training)
        lconv2=tf.layers.conv2d(lbn1,cns//2,1,padding='SAME',
                                kernel_initializer=initial,
                                kernel_regularizer=regular)
        lbn2=tf.layers.batch_normalization(lconv2,training=is_training)
        lrelu2=tf.nn.relu(lbn2)
        rconv1=tf.layers.conv2d(right,cns//2,1,padding='SAME',
                                kernel_initializer=initial,
                                kernel_regularizer=regular)
        rbn1=tf.layers.batch_normalization(rconv1,training=is_training)
        rrelu1=tf.nn.relu(rbn1)
        depth_conv_kernel2=tf.get_variable('dconv2',shape=[3,3,cns//2,1],initializer=initial,regularizer=regular)
        rconv2=tf.nn.depthwise_conv2d(rrelu1,filter=depth_conv_kernel2,
                                      strides=[1,2,2,1],padding='SAME')
        rbn2=tf.layers.batch_normalization(rconv2,training=is_training)
        rconv3=tf.layers.conv2d(rbn2,cns//2,1,padding='SAME',
                                kernel_initializer=initial,
                                kernel_regularizer=regular)
        rbn3=tf.layers.batch_normalization(rconv3,training=is_training)
        rrelu3=tf.nn.relu(rbn3)
        out=tf.concat([lrelu2,rrelu3],axis=3)
        out=tf.reshape(out,shape=[-1,H,W,cns//2,2])
        out=tf.transpose(out,perm=[0,1,2,4,3])
        out=tf.reshape(out,shape=[-1,H,W,cns])
        return out
    
def stage_1(img,cns,regular,initial):
    with tf.variable_scope('stage_1'):
        conv=tf.layers.conv2d(img,cns,3,strides=2,padding='SAME',
                              kernel_initializer=initial,
                              kernel_regularizer=regular)
        pool=tf.layers.max_pooling2d(conv,3,strides=2,padding='SAME')
        return pool

def stage_2(l,cns,is_training,regular,initial):
    with tf.variable_scope('stage_2'):
        l=unit_d(l,cns,is_training,regular,initial)
        for i in range(3):
            l=unit_c(l,cns,is_training,regular,initial,i)
    return l

def stage_3(l,cns,is_training,regular,initial):
    with tf.variable_scope('stage_3'):
        l=unit_d(l,cns,is_training,regular,initial)
        for i in range(7):
            l=unit_c(l,cns,is_training,regular,initial,i)
    return l

def stage_4(l,cns,is_training,regular,initial):
    with tf.variable_scope('stage_4'):
        l=unit_d(l,cns,is_training,regular,initial)
        for i in range(3):
            l=unit_c(l,cns,is_training,regular,initial,i)
    return l

def stage_5(l,cns,cls,regular,initial):
    with tf.variable_scope('stage_5'):
        conv=tf.layers.conv2d(l,cns,1,padding='SAME',
                              kernel_initializer=initial,
                              kernel_regularizer=regular)
        pool=tf.reduce_mean(conv,axis=[1,2])
        fc=tf.layers.dense(pool,cls)
        return fc
    
def model(img,cns,cls,is_training,regular,initial):
    with tf.variable_scope('model',reuse=not is_training):
        o1=stage_1(img,cns[0],regular,initial)
        o2=stage_2(o1,cns[1],is_training,regular,initial)
        o3=stage_3(o2,cns[2],is_training,regular,initial)
        o4=stage_4(o3,cns[3],is_training,regular,initial)
        logits=stage_5(o4,cns[4],cls,regular,initial)
        return logits