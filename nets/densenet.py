from ops import *
from utils import *
import utils
from tensorflow.contrib.layers.python.layers import layers

FLAGS = tf.app.flags.FLAGS

SN = True

def conv2pool(input, filters, kernel, decay, stride, scope, training, reuse=True):
    x = conv(input, channels=filters, kernel=kernel, stride=stride, pad=1, sn=SN, use_bias=False, scope=scope)
    x = batch_norm(x, is_training=training, scope=scope + '_batch1')
    x = tf.nn.relu(x)
    #x = lrelu(x, 0.2)
    x = tf.contrib.layers.max_pool2d(inputs=x, kernel_size=[2, 2], stride=2, padding='VALID')
    return x

def bottleneck_layer_2d(input, filters, drop_rate, decay, scope, training, reuse):
    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        x = batch_norm(input, is_training=training, scope=scope + '_batch1')
        x = tf.nn.relu(x)
        #x = lrelu(x, 0.2)
        x = conv(x, channels=4 * filters, kernel=1,stride=1,pad=0,sn=SN, use_bias=False, scope=scope+'_conv1')
        #x = tf.contrib.layers.dropout(inputs=x, keep_prob=drop_rate, is_training=training)
        
        x = batch_norm(x, is_training=training, scope=scope + '_batch2')
        x = tf.nn.relu(x)
        #x = lrelu(x, 0.2)
        x = conv(x, channels=filters, kernel=3,stride=1,pad=1,sn=SN, use_bias=False, scope=scope+'_conv2')
        #x = tf.contrib.layers.dropout(inputs=x, keep_prob=drop_rate, is_training=training)
        return x

def transition_layer_2d(input, filters, drop_rate, decay, scope, training, reuse):
    with tf.variable_scope(name_or_scope=scope, reuse=reuse):
        x = batch_norm(input, is_training=training, scope=scope + '_batch')
        x = tf.nn.relu(x)
        #x = lrelu(x, 0.2)
        x = conv(x, channels=filters, kernel=1,stride=1,pad=0,sn=SN, use_bias=False, scope=scope+'_conv')
        #x = tf.contrib.layers.dropout(x, keep_prob=drop_rate, is_training=training)
        x = tf.contrib.layers.avg_pool2d(inputs=x, kernel_size=[2, 2], stride=2, padding='VALID')
        return x

def dense_block_2d(input, filters, nb_layers, drop_rate, decay, training, reuse, scope):
    with tf.name_scope(scope):
        layers_concat = list()
        layers_concat.append(input)
        x = input
        for i in range(nb_layers):
            x = bottleneck_layer_2d(x, filters, drop_rate, decay, training=True,
                                    reuse=reuse, scope=scope + '_bottleN_' + str(i+1))
            layers_concat.append(x)
            x = tf.concat(layers_concat, axis=-1)
        return x

def AAA(x, channels, sn=False, de=4, scope='attention_cross', reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
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

def make_png(att, scale):
    att_current = up_sample_bilinear(att, scale_factor=scale)
    att_current = tf.nn.relu(att_current)
    att_current = tf.reduce_mean(att_current,axis=-1)
    att_current = tf.stack([att_current, att_current, att_current])
    att_current = tf.transpose(att_current, perm=[1, 2, 3, 0])
    return att_current
            
def discriminator(image, num_classes, drop_rate=1.0, decay=0.9, growth_k=12, trainable=True, reuse=False, scope='dis'):
    layer_num = int(np.log2(int(image.shape[1]))) - 3
    with tf.variable_scope(scope, reuse=reuse):
        print('model_name:densenet')
        end_points = {}
        #################################################################################
        ###conv pool  input: 128  output: 64
        #################################################################################
        logits = conv2pool(image, filters=2*growth_k, kernel=3, stride=1, decay=decay,
                      training=trainable, reuse=reuse, scope='conv2pool_1')
        print(logits)
            
        #################################################################################
        ###dense_block1  &&  trans_layer1  input: 64  output: 32
        #################################################################################
        logits = dense_block_2d(logits, growth_k, nb_layers=6, drop_rate=drop_rate, decay=decay,training=trainable, reuse=reuse, scope='dense_block_1')
        print(logits)
        logits = transition_layer_2d(logits, filters=0.5*int(logits.shape[-1]), drop_rate=drop_rate, decay=decay,
                                     training=trainable, reuse=reuse,
                                     scope='trans_layer_1')#96
        print(logits)
        if scope == 'dmlnet_0':
            logits = AAA(logits, int(logits.shape[-1]), sn=SN, de=4, scope="attention0", reuse=reuse)
            print(logits)
            end_points['attention0'] = make_png(logits, 4)           
            
        #################################################################################
        ###dense_block2  &&  trans_layer2  input: 32  output: 16
        #################################################################################
        logits = dense_block_2d(logits, growth_k, nb_layers=12, drop_rate=drop_rate, decay=decay,training=trainable, reuse=reuse, scope='dense_block_2')
        print(logits)
        logits = transition_layer_2d(logits, filters=0.5*int(logits.shape[-1]),  drop_rate=drop_rate, decay=decay,
                                     training=trainable, reuse=reuse, scope='trans_layer_2')
        print(logits)
        if scope == 'dmlnet_0':
            logits = AAA(logits, int(logits.shape[-1]), sn=SN, de=4, scope="attention1", reuse=reuse)
            print(logits)
            end_points['attention1'] = make_png(logits, 8)          
            
        #################################################################################
        ###dense_block3  &&  trans_layer3  input: 16  output: 8
        #################################################################################
        logits = dense_block_2d(logits, growth_k, nb_layers=24,  drop_rate=drop_rate, decay=decay,training=trainable, reuse=reuse, scope='dense_block_3')
        print(logits)
        logits = transition_layer_2d(logits, filters=0.5*int(logits.shape[-1]),  drop_rate=drop_rate, decay=decay,
                                     training=trainable, reuse=reuse, scope='trans_layer_3')
        print(logits)

        #################################################################################
        ###dense_block4  input: 8  output: 8
        #################################################################################
        logits = dense_block_2d(logits, growth_k, nb_layers=16, decay=decay,  drop_rate=drop_rate,training=trainable, reuse=reuse, scope='dense_block_4')
        print(logits) 
#        end_points['feature_map'] = logits           
            
        logits = global_avg_pool(logits, name='Global_avg_pooling_pool')
        feature = tf.squeeze(logits)
        print(feature)
        end_points['feature'] = feature
        #logits = tf.layers.dense(inputs=logits, units=num_classes, name='fc2')
        #logits = fully_conneted(logits, num_classes, use_bias=True, sn=SN, scope='fc2')
        #print(logits)
        #end_points['Logits'] = logits
        #end_points['Predictions'] = layers.softmax(logits, scope='predictions')
        
        return feature, end_points
