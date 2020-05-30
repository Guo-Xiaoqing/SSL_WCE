import tensorflow as tf
from utils import conv2layer, pool2layer
from tensorflow.contrib.framework import arg_scope
from ops import *
from utils import *
import utils
from tensorflow.contrib.layers.python.layers import layers
########################################
############hyper parameters############
########################################
#init lr=0.01, wach 30 epochs *0.1
########################################


def make_png(att, scale):
    #att_current = up_sample_bilinear1(att, scale_factor=scale)
    att_current = tf.image.resize_bilinear(att, size=[128, 128])
    att_current = tf.nn.relu(att_current)
    att_current = tf.reduce_mean(att_current,axis=-1)
    att_current = tf.stack([att_current, att_current, att_current])
    att_current = tf.transpose(att_current, perm=[1, 2, 3, 0])
    return att_current

def attention_cross(x, channels, sn=False, de=4, scope='attention_cross'):
    with tf.variable_scope(scope):
        f = conv(x, channels // de, kernel=1, stride=1, sn=sn, scope='f_conv') # [bs, h, w, c']
        g = conv(x, channels // de, kernel=1, stride=1, sn=sn, scope='g_conv') # [bs, h, w, c']
        h = conv(x, channels, kernel=1, stride=1, sn=sn, scope='h_conv') # [bs, h, w, c]
        
        f1 = atrous_conv2d(x, channels // de, kernel=3, rate=2, sn=sn, scope='f1_conv') # [bs, h, w, c']
        g1 = atrous_conv2d(x, channels // de, kernel=3, rate=2, sn=sn, scope='g1_conv') # [bs, h, w, c']

        # N = h * w
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True) # # [bs, N, N]
        s1 = tf.matmul(hw_flatten(g1), hw_flatten(f1), transpose_b=True) # # [bs, N, N]

        beta_a = tf.nn.softmax(s, dim=-1)  # attention map
        beta_a1 = tf.nn.softmax(s1, dim=-1)  # attention map

        o = tf.matmul(beta_a, hw_flatten(h)) # [bs, N, C]
        o1 = tf.matmul(beta_a1, hw_flatten(h)) # [bs, N, C]
        
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        gamma1 = tf.get_variable("gamma1", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=x.shape) # [bs, h, w, C]
        o1 = tf.reshape(o1, shape=x.shape) # [bs, h, w, C]
        att = gamma * o + gamma1 * o1
        x = att + x

    return x

def vgg16(images, n_classes, trainable=False, reuse=None, scope=None):
    keep_prob = 1.0
    end_points = {}
    with tf.variable_scope(scope, reuse=reuse) as sc:
        with arg_scope([conv2layer,pool2layer], stride=1, padding_mode='SAME'):
            with arg_scope([conv2layer], activation='relu', bn=True, trainning=trainable):
                net = conv2layer(images=images, kernel=[3, 3], output_channel=16, scope='conv1')

                net = conv2layer(images=net, kernel=[3, 3], output_channel=16, scope='conv2')

                net = pool2layer(images=net, kernel=[3, 3], stride=2, scope='pooling1')

                net = conv2layer(images=net, kernel=[3, 3], output_channel=32, scope='conv3')
                net = conv2layer(images=net, kernel=[3, 3], output_channel=32, scope='conv4')
                net = pool2layer(images=net, kernel=[3, 3], stride=2, scope='pooling2')

                #net = attention_cross(net, int(net.shape[-1]), sn=False, de=4, scope="attention0")
                end_points['attention0'] = make_png(net, 4)   
                print('attention0', net)
                net = conv2layer(images=net, kernel=[3, 3], output_channel=64, scope='conv5')
                net = conv2layer(images=net, kernel=[3, 3], output_channel=64, scope='conv6')
                #net = conv2layer(images=net, kernel=[3, 3], output_channel=256, scope='conv7')
                net = pool2layer(images=net, kernel=[3, 3], stride=2, scope='pooling3')

                #net = attention_cross(net, int(net.shape[-1]), sn=False, de=4, scope="attention1")
                end_points['attention1'] = make_png(net, 8)           
                print('attention1', net)
                net = conv2layer(images=net, kernel=[3, 3], output_channel=128, scope='conv8')
                net = conv2layer(images=net, kernel=[3, 3], output_channel=128, scope='conv9')
                #net = conv2layer(images=net, kernel=[3, 3], output_channel=512, scope='conv10')
                net = pool2layer(images=net, kernel=[3, 3], stride=2, scope='pooling4')

                #net = conv2layer(images=net, kernel=[3, 3], output_channel=512, scope='conv11')
                #net = conv2layer(images=net, kernel=[3, 3], output_channel=512, scope='conv12')
                #net = conv2layer(images=net, kernel=[3, 3], output_channel=512, scope='conv13')
                #net = pool2layer(images=net, kernel=[3, 3], stride=2, scope='pooling5')

                net = conv2layer(images=net, kernel=[8, 8], output_channel=512, padding_mode='VALID', scope='fc1')
                net = tf.nn.dropout(net, keep_prob=keep_prob)
                net = conv2layer(images=net, kernel=[1, 1], output_channel=512, scope='fc2')
                net = tf.nn.dropout(net, keep_prob=keep_prob)
                
                feature = tf.squeeze(net)
                print(feature)
                end_points['feature'] = feature
                
                net = conv2layer(images=net, kernel=[1, 1], output_channel=n_classes, activation=None, scope='logits')
                net = tf.squeeze(net, axis=[1, 2], name='squeeze_logits')

                return net, end_points