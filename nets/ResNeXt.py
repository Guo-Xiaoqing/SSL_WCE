import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope
import numpy as np
from ops import *
########################################
############hyper parameters############
########################################
# initial learning rate net0=0.0025, net1=0.005
# learning rate each 20 epoches *0.9
# baseline init lr=0.1, wach 30 epochs *0.1
########################################

cardinality = 2 # how many split ?
blocks = 4 # res_block ! (split + transition)
depth = 32 # out channel

def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Average_pooling(x, pool_size=[2,2], stride=2, padding='SAME'):
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Batch_Normalization(x, training, scope):
    '''with arg_scope([batch_norm],
                   scope=scope,
                   updates_collections=None,
                   decay=0.9,
                   center=True,
                   scale=True,
                   zero_debias_moving_mean=True) :
        return tf.cond(training,
                       lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                       lambda : batch_norm(inputs=x, is_training=training, reuse=True))'''
    return batch_norm(x, is_training=training, scope=scope)

def Relu(x):
    return tf.nn.relu(x)

def Concatenation(layers) :
    return tf.concat(layers, axis=3)

def Linear(x) :
    return tf.layers.dense(inputs=x, use_bias=False, units=class_num, name='linear')

def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    test_pre_index = 0
    add = 1000

    for it in range(test_iteration):
        test_batch_x = test_x[test_pre_index: test_pre_index + add]
        test_batch_y = test_y[test_pre_index: test_pre_index + add]
        test_pre_index = test_pre_index + add

        test_feed_dict = {
            x: test_batch_x,
            label: test_batch_y,
            learning_rate: epoch_learning_rate,
            training_flag: False
        }

        loss_, acc_ = sess.run([cost, accuracy], feed_dict=test_feed_dict)

        test_loss += loss_
        test_acc += acc_

    test_loss /= test_iteration # average loss
    test_acc /= test_iteration # average accuracy

    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary

class ResNeXt():
    def __init__(self, x, num_classes, training=True, reuse=True, scope=None):
        self.num_classes = num_classes
        self.training = training
        self.reuse = reuse
        self.scope = scope
        self.model = self.Build_ResNext(x)

    def first_layer(self, x, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=32, kernel=[3, 3], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)
            x = Average_pooling(x)

            return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope) :
            x = conv_layer(x, filter=depth, kernel=[1,1], stride=stride, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            x = Relu(x)

            x = conv_layer(x, filter=depth, kernel=[3,3], stride=1, layer_name=scope+'_conv2')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch2')
            x = Relu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=[1,1], stride=1, layer_name=scope+'_conv1')
            x = Batch_Normalization(x, training=self.training, scope=scope+'_batch1')
            # x = Relu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name) :
            layers_split = list()
            for i in range(cardinality) :
                splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def residual_layer(self, input_x, out_dim, layer_num, res_block=blocks):
        # split + transform(bottleneck) + transition + merge

        for i in range(res_block):
            # input_dim = input_x.get_shape().as_list()[-1]
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1
            x = self.split_layer(input_x, stride=stride, layer_name='split_layer_'+layer_num+'_'+str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_'+layer_num+'_'+str(i))

            if flag is True :
                pad_input_x = Average_pooling(input_x)
                pad_input_x = tf.pad(pad_input_x, [[0, 0], [0, 0], [0, 0], [channel, channel]]) # [?, height, width, channel]
            else :
                pad_input_x = input_x

            input_x = Relu(x + pad_input_x)

        return input_x 


    def Build_ResNext(self, input_x):
        end_points = {}
        with tf.variable_scope(self.scope, reuse=self.reuse):
            # only cifar10 architecture

            input_x = self.first_layer(input_x, scope='first_layer')
            print('1',input_x)

            x = self.residual_layer(input_x, out_dim=32, layer_num='1', res_block=3)
            print('2',x)

            x = self.residual_layer(x, out_dim=64, layer_num='2', res_block=4)
            if self.scope == 'dmlnet_0':
                x = attention_cross(x, int(x.shape[-1]), sn=False, de=4, scope="attention0")
                end_points['attention0'] = make_png(x, 4)             
            print('3',x)

            x = self.residual_layer(x, out_dim=128, layer_num='3', res_block=6)
            if self.scope == 'dmlnet_0':
                x = attention_cross(x, int(x.shape[-1]), sn=False, de=4, scope="attention1")
                end_points['attention1'] = make_png(x, 8)             
            print('4',x)

            x = self.residual_layer(x, out_dim=256, layer_num='4', res_block=3)           
            print('5',x)

            x = Global_Average_Pooling(x)
            print('6',x)
            x = flatten(x)
            print('7',x)
            end_points['feature'] = x
            #x =tf.layers.dense(inputs=x, use_bias=False, units=self.num_classes, name='linear')
        return x, end_points
    
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