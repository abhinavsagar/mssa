import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import os, sys

def Upsampling(inputs,scale):
    return tf.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*scale,  tf.shape(inputs)[2]*scale])

def ConvUpscaleBlock(inputs, n_filters, kernel_size=[3, 3], scale=2):
    net = tf.nn.relu(slim.batch_norm(inputs, fused=True))
    net = slim.conv2d_transpose(net, n_filters, kernel_size=[3, 3], stride=[scale, scale], activation_fn=None)
    return net

def ConvBlock(inputs, n_filters, kernel_size=[3, 3], strides=1):
    net = slim.conv2d(inputs, n_filters, kernel_size, stride=[strides, strides], activation_fn=None, normalizer_fn=None)
    net = tf.nn.relu(slim.batch_norm(net, fused=True))
    return net

def AttentionModule(inputs, n_filters):
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = slim.batch_norm(net, fused=True)
    net = tf.sigmoid(net)
    net = tf.multiply(inputs, net)
    return net

def FeatureFusionModule(input_1, input_2, n_filters):
    inputs = tf.concat([input_1, input_2], axis=-1)
    inputs = ConvBlock(inputs, n_filters=n_filters, kernel_size=[3, 3])
    net = tf.reduce_mean(inputs, [1, 2], keep_dims=True)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.nn.relu(net)
    net = slim.conv2d(net, n_filters, kernel_size=[1, 1])
    net = tf.sigmoid(net)
    net = tf.multiply(inputs, net)
    net = tf.add(inputs, net)
    return net


def build_mssa(inputs, num_classes, preset_model='mssa', frontend="ResNet101", weight_decay=1e-5, is_training=True, pretrained_dir="models"):
    spatial_net = ConvBlock(inputs, n_filters=64, kernel_size=[3, 3], strides=2)
    spatial_net = ConvBlock(spatial_net, n_filters=128, kernel_size=[3, 3], strides=2)
    spatial_net = ConvBlock(spatial_net, n_filters=256, kernel_size=[3, 3], strides=2)
    logits, end_points, frontend_scope, init_fn  = frontend_builder.build_frontend(inputs, frontend, pretrained_dir=pretrained_dir, is_training=is_training)

    net_4 = AttentionModule(end_points['pool4'], n_filters=512)
    net_5 = AttentionModule(end_points['pool5'], n_filters=2048)

    global_channels = tf.reduce_mean(net_5, [1, 2], keep_dims=True)
    net_5_scaled = tf.multiply(global_channels, net_5)
    net_4 = Upsampling(net_4, scale=2)
    net_5_scaled = Upsampling(net_5_scaled, scale=4)
    context_net = tf.concat([net_4, net_5_scaled], axis=-1)

    net = FeatureFusionModule(input_1=spatial_net, input_2=context_net, n_filters=num_classes)
    net = Upsampling(net, scale=8)
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='logits')
    return net, init_fn
