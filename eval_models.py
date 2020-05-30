from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import dataset_factory
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
from datasets.utils import *
import numpy as np
import time
import utils
from ops import *
from utils import *
slim = tf.contrib.slim
FLAGS = tf.app.flags.FLAGS
from tensorflow.python.ops import array_ops
from tensorflow.contrib.layers.python.layers import layers
import math

def _average_gradients(tower_grads, catname=None):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(input=g, axis=0)
            # print(g)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads, name=catname)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def JS_loss_compute(logits1, logits2):
    """ JS loss
    """
    pred1 = tf.nn.softmax(logits1)
    pred2 = tf.nn.softmax(logits2)
    ave = (pred1 + pred2) / 2
    loss = 0.5*tf.reduce_mean(tf.reduce_sum(pred2 * tf.log((1e-8 + pred2) / (ave + 1e-8)), 1)) + 0.5*tf.reduce_mean(tf.reduce_sum(pred1 * tf.log((1e-8 + pred1) / (ave + 1e-8)), 1))
    return loss

###########################################################################
########our final loss function
###########################################################################
def DA_loss(Features, labels, out_num, s=64., m=0.5, k = 0.3, scope = 'arc_loss', is_cross=True, reuse=True):
    '''
    :param embedding: the input embedding vectors
    :param labels:  the input labels, the shape should be eg: (batch_size, 1)
    :param s: scalar value default is 64
    :param out_num: output class num
    :param m: the margin value, default is 0.5
    :return: the final cacualted output, this output is send into the tf.nn.softmax directly
    '''
    cos_m = math.cos(m)
    sin_m = math.sin(m)
    mm = sin_m * m  # issue 1
    threshold = math.cos(math.pi - m)
    with tf.variable_scope(scope+'arc_loss', reuse=reuse):
        # inputs and weights norm
        embedding_norm = tf.norm(Features, axis=1, keep_dims=True)
        embedding = tf.div(Features, embedding_norm+10e-6, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=tf.orthogonal_initializer, dtype=tf.float32)
        print(embedding, weights)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights = tf.div(weights, weights_norm+10e-6, name='norm_weights')
        # cos(theta+m)
        arccos = tf.acos(tf.matmul(embedding, weights, name='cos_t'))
        cos_t = activation_function(arccos, k)
        pred = cos_t
        output = cos_t
        
        if is_cross:
            arccos_mt = tf.acos(tf.matmul(embedding, weights, name='cos_t')) + m
            cos_mt =  s*activation_function(arccos_mt, k)

            # this condition controls the theta+m should in range [0, pi]
            #      0<=theta+m<=pi
            #     -m<=theta<=pi-m
            cond_v = cos_t - threshold
            cond = tf.cast(tf.nn.relu(cond_v, name='if_else'), dtype=tf.bool)

            keep_val = s*(cos_t - mm)
            cos_mt_temp = tf.where(cond, cos_mt, keep_val)

            mask = tf.one_hot(labels, depth=out_num, name='one_hot_mask')
            # mask = tf.squeeze(mask, 1)
            inv_mask = tf.subtract(1., mask, name='inverse_mask')

            s_cos_t = tf.multiply(s, cos_t, name='scalar_cos_t')

            output = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask), name='DA_loss_output')
            
    with tf.variable_scope(scope+'center_loss', reuse=reuse):
        alpha = 0.5
        len_features = Features.get_shape()[1]
        #centers = tf.get_variable('centers', [out_num, len_features], dtype=tf.float32,
        #    initializer=tf.constant_initializer(0), trainable=False)
        centers = tf.get_variable('centers', dtype=tf.float32, initializer=tf.transpose(weights), trainable=False)
        
        ######regularization term calculate##########
        centers_norm = tf.norm(centers, axis=1, keep_dims=True)
        centers_normed = tf.div(centers, centers_norm+10e-6, name='norm_weights')
        A = [1] * out_num
        exclude = tf.to_float(tf.matrix_diag(A))
        zeros = array_ops.zeros_like(exclude, dtype=exclude.dtype)
        reg = tf.matmul(centers_normed, weights)
        print(reg)
        reg = tf.where(exclude>0.0, zeros, reg)
        regularization = tf.reduce_sum(reg) / ((out_num-1) * out_num)
        ######regularization term calculate##########

        Labels = tf.reshape(labels, [-1])
    
        centers_batch = tf.gather(centers, Labels)
        numerator = tf.norm(Features - centers_batch, axis=-1)
        f = tf.expand_dims(Features, axis=1)
        f = tf.tile(f,[1,centers.shape[0],1])
        denominator = tf.norm(f - centers, axis=-1)
        denominator = 1e-8 + tf.reduce_sum(denominator, axis=-1) - numerator
        loss_weight = (out_num-1) * numerator/denominator
    
        diff = centers_batch - Features
    
        unique_label, unique_idx, unique_count = tf.unique_with_counts(Labels)
        appear_times = tf.gather(unique_count, unique_idx)
        appear_times = tf.reshape(appear_times, [-1, 1])
    
        diff = diff / tf.cast((1 + appear_times), tf.float32)
        diff = alpha * diff    
        centers_update_op = tf.scatter_sub(centers, Labels, diff)        
            
    return pred, output, regularization, loss_weight, centers_update_op

def activation_function(arccos, q):
    q = 3.0
    p = 1.0
    cos_t = 1-2*(p+tf.exp(-arccos*q+math.pi/2.0*q))**(-1)
    return cos_t

def _tower_loss(network_fn, images, labels, is_cross = True, reuse=False, is_training=False):
    """Calculate the total loss on a single tower running the reid model.""" 
    scale = 4.
    margin = 0.5
    net_features, net_logits, net_endpoints, net_raw_loss, net_pred, net_features, c_update_op = {}, {}, {}, {}, {}, {}, {}
    weight_loss, net_regularization, net_pred = {}, {}, {}
    for i in range(FLAGS.num_networks):
        net_features["{0}".format(i)], net_endpoints["{0}".format(i)] = \
            network_fn["{0}".format(i)](images, reuse=reuse, is_training=is_training, scope=('dmlnet_%d' % i))
        #net_pred["{0}".format(i)] = net_endpoints["{0}".format(i)]['Predictions']                
        
        if is_cross:
            net_pred["{0}".format(i)], net_logits["{0}".format(i)], net_regularization["{0}".format(i)], weight, c_update_op["{0}".format(i)] = DA_loss(net_features["{0}".format(i)], tf.argmax(labels, axis=1), FLAGS.num_classes, s=scale, m=margin, scope = ('dmlnet_%d' % i), is_cross=is_cross, reuse=reuse)
            net_pred["{0}".format(i)] = layers.softmax(net_pred["{0}".format(i)], scope='predictions')
            
            raw_loss = tf.nn.softmax_cross_entropy_with_logits(logits=net_logits["{0}".format(i)], labels=labels)
            weighted_loss = tf.multiply(weight, raw_loss)
            #weight_loss["{0}".format(i)] = tf.reduce_mean(weight)
            net_raw_loss["{0}".format(i)] = tf.reduce_mean(raw_loss)
            #net_raw_loss["{0}".format(i)] = tf.losses.softmax_cross_entropy(
            #    logits=net_logits["{0}".format(i)], onehot_labels=labels,
            #    label_smoothing=FLAGS.label_smoothing, weights=1.0)
            kl_weight = 1.0

        else:
            print('semi_data!')
            net_pred["{0}".format(i)], _, _, _, _ = DA_loss(net_features["{0}".format(i)], tf.argmax(labels, axis=1), FLAGS.num_classes, s=scale, m=margin, scope = ('dmlnet_%d' % i), is_cross=is_cross, reuse=reuse)
            net_pred["{0}".format(i)]  = layers.softmax(net_pred["{0}".format(i)], scope='predictions')
            ## if the maximum probability of semi data is larger than threshold, update the feature centers.
            softmax_logits = net_pred["{0}".format(i)]   
            ones = array_ops.ones_like(softmax_logits, dtype=softmax_logits.dtype)
            zeros = array_ops.zeros_like(softmax_logits, dtype=softmax_logits.dtype)
            threshold = 0.7 * array_ops.ones_like(softmax_logits, dtype=softmax_logits.dtype)#threshold is set as 0.85
            threshold_softmax_logits = array_ops.where(softmax_logits > threshold, ones, zeros)
            threshold_softmax_logits = tf.reduce_max(threshold_softmax_logits, axis=-1)
            idx = tf.where(threshold_softmax_logits > 0.95)
            feats = tf.gather_nd(net_endpoints["{0}".format(i)]['feature'], idx)    
            argmax_logits = tf.gather_nd(tf.argmax(softmax_logits, axis=1), idx)    
            _, _, _, _, c_update_op["{0}".format(i)] = DA_loss(feats, argmax_logits, FLAGS.num_classes, s=scale, m=margin, scope = ('dmlnet_%d' % i), is_cross=True, reuse=reuse)
                
            net_raw_loss["{0}".format(i)] = tf.constant(0.0)
            net_regularization["{0}".format(i)] = tf.constant(0.0)
            kl_weight = 1.0    
       
        if i == 0:
            attention0 = net_endpoints["{0}".format(i)]['attention0']
            attention1 = net_endpoints["{0}".format(i)]['attention1']
            #images = 0.5 * (images + 0.5 * images * attention0 + 0.5 * images * attention1)
            images = (images + images * attention0 + images * attention1) / 3.0
            
    # Add KL loss if there are more than one network
    net_loss, kl_loss, net_reg_loss, net_total_loss, net_loss_averages, net_loss_averages_op = {}, {}, {}, {}, {}, {}
    
    for i in range(FLAGS.num_networks):
        net_loss["{0}".format(i)] = net_raw_loss["{0}".format(i)] + net_regularization["{0}".format(i)]
        for j in range(FLAGS.num_networks):
            if i != j:
                kl_loss["{0}{0}".format(i, j)] = JS_loss_compute(net_pred["{0}".format(i)], net_pred["{0}".format(j)])
                net_loss["{0}".format(i)] += kl_weight*kl_loss["{0}{0}".format(i, j)]
                #tf.summary.scalar('kl_loss_%d%d' % (i, j), kl_loss["{0}{0}".format(i, j)])

        net_reg_loss["{0}".format(i)] = tf.add_n([FLAGS.weight_decay * tf.nn.l2_loss(var) for var in tf.trainable_variables() if 'dmlnet_%d' % i in var.name])
        net_total_loss["{0}".format(i)] = net_loss["{0}".format(i)] + net_reg_loss["{0}".format(i)]
        
    return net_total_loss, c_update_op, net_pred, attention0, attention1, images

def to_heat(input_image):
    input_image[input_image<0] = 0
    heatmap = input_image
    print(np.min(heatmap))
    heatmap = (heatmap-np.min(heatmap))/(np.max(heatmap)-np.min(heatmap))
    heatmap = np.uint8(255 * heatmap) 
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET) 
    return heatmap

def train():
    if not FLAGS.dataset_dir:
        raise ValueError('You must supply the dataset directory with --dataset_dir')

    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.Graph().as_default():
        #######################
        # Config model_deploy #
        #######################
        deploy_config = model_deploy.DeploymentConfig(
            num_clones=FLAGS.num_clones,
            clone_on_cpu=FLAGS.clone_on_cpu,
            replica_id=FLAGS.task,
            num_replicas=FLAGS.worker_replicas,
            num_ps_tasks=FLAGS.num_ps_tasks)

        # Create global_step
        with tf.device(deploy_config.variables_device()):
            global_step = tf.train.create_global_step()

        ######################
        # Select the network and #
        ######################
        network_fn = {}
        model_names = [net.strip() for net in FLAGS.model_name.split(',')]
        for i in range(FLAGS.num_networks):
            network_fn["{0}".format(i)] = nets_factory.get_network_fn(
                model_names[i],
                num_classes=FLAGS.num_classes,
                weight_decay=FLAGS.weight_decay)

        #####################################
        # Select the preprocessing function #
        #####################################
        preprocessing_name = FLAGS.preprocessing_name  # or FLAGS.model_name
        image_preprocessing_fn = preprocessing_factory.get_preprocessing(
            preprocessing_name,
            is_training=True)

        ##############################################################
        # Create a dataset provider that loads data from the dataset #
        ##############################################################            
        test_image_batch, test_label_batch = utils.get_image_label_batch(FLAGS, shuffle=False, name='test4')
 
        test_label_batch = slim.one_hot_encoding(test_label_batch, FLAGS.num_classes)
            
        precision, test_precision, test_predictions, net_var_list, net_grads, net_update_ops = {}, {}, {}, {}, {}, {}
        semi_net_grads = {}

        with tf.name_scope('tower') as scope:
            with tf.variable_scope(tf.get_variable_scope()):
                test_net_loss, _, test_net_pred, test_attention0, test_attention1, test_second_input = _tower_loss(network_fn, test_image_batch, test_label_batch, is_cross = True, reuse=False, is_training=False)

                test_truth = tf.argmax(test_label_batch, axis=1)

                # Reuse variables for the next tower.
                #tf.get_variable_scope().reuse_variables()

                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                var_list = tf.trainable_variables()

                for i in range(FLAGS.num_networks):
                    test_predictions["{0}".format(i)] = tf.argmax(test_net_pred["{0}".format(i)], axis=1)
                    test_precision["{0}".format(i)] = tf.reduce_mean(tf.to_float(tf.equal(test_predictions["{0}".format(i)], test_truth)))
                    #test_predictions["{0}".format(i)] = test_net_pred["{0}".format(i)]
                net_pred = (test_net_pred["{0}".format(0)] + test_net_pred["{0}".format(1)])/2.0
                net_pred = tf.argmax(net_pred, axis=1)
                precision_mean = tf.reduce_mean(tf.to_float(tf.equal(net_pred, test_truth)))
                

                    # Add a summary to track the training precision.
                    #summaries.append(tf.summary.scalar('precision_%d' % i, precision["{0}".format(i)]))
                    #summaries.append(tf.summary.scalar('test_precision_%d' % i, test_precision["{0}".format(i)]))
          
        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        #summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5),
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        
        load_fn = slim.assign_from_checkpoint_fn(os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-0'),tf.global_variables(),ignore_missing_vars=True)
        #load_fn = slim.assign_from_checkpoint_fn('./WCE_densenet4/checkpoint/model.ckpt-20',tf.global_variables(),ignore_missing_vars=True)
        load_fn(sess)
        
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        #summary_writer = tf.summary.FileWriter(
        #    os.path.join(FLAGS.log_dir),
        #    graph=sess.graph)

        net_loss_value, test_precision_value, test_predictions_value, precision_value = {}, {}, {}, {}

        parameters = utils.count_trainable_params()
        print("Total training params: %.1fM \r\n" % (parameters / 1e6))
                  
        start_time = time.time() 
        counter = 0
        infile = open(os.path.join(FLAGS.attention_map, 'log.txt'),'w')
        batch_count = np.int32(FLAGS.dataset_size / FLAGS.batch_size)  
        precision_value["{0}".format(0)] = []
        precision_value["{0}".format(1)] = []
        precision_value["{0}".format(2)] = []
        feature0 = []
        feature1 = []
        for batch_idx in range(batch_count):
            #for i in range(FLAGS.num_networks):
            test_predictions_value["{0}".format(0)], test_predictions_value["{0}".format(1)], truth, predictions, test_precision_value["{0}".format(0)], test_precision_value["{0}".format(1)], prec, test, test_att0, test_att1, test_sec = sess.run([test_predictions["{0}".format(0)], test_predictions["{0}".format(1)], test_truth, net_pred, test_precision["{0}".format(0)], test_precision["{0}".format(1)], precision_mean, test_image_batch, test_attention0, test_attention1, test_second_input])
                
            precision_value["{0}".format(0)].append(test_precision_value["{0}".format(0)])
            precision_value["{0}".format(1)].append(test_precision_value["{0}".format(1)])
            precision_value["{0}".format(2)].append(prec)
                
            #predictions = test_predictions_value["{0}".format(1)]
            #infile.write(str(np.around(predictions[:,0], decimals=3))+' '+str(np.around(predictions[:,1], decimals=3))+'\n')
            #infile.write(str(np.around(predictions[:,2], decimals=3))+' '+str(truth)+'\n')
            infile.write(str(test_predictions_value["{0}".format(0)])+' '+str(test_predictions_value["{0}".format(1)])+'\n')
            infile.write(str(predictions)+' '+str(truth)+'\n')
            infile.write(str(np.float32(test_precision_value["{0}".format(0)]))+' '+str(np.float32(test_precision_value["{0}".format(1)]))+' '+str(np.float32(prec))+'\n')
            format_str = 'batch_idx: [%3d] [%3d/%3d] time: %4.4f, net0_test_acc = %.4f,      net1_test_acc = %.4f,      net_test_acc = %.4f'
            print(format_str % (batch_idx, batch_idx,batch_count, time.time()-start_time, np.float32(test_precision_value["{0}".format(0)]),np.float32(test_precision_value["{0}".format(1)]),np.float32(prec)))                    
                    
                #train, att0, att1, sec, semi, semi_att0, semi_att1, semi_sec, test, test_att0, test_att1, test_sec  = sess.run([train_image_batch, attention0, attention1, second_input, semi_image_batch, semi_attention0, semi_attention1, semi_second_input, test_image_batch, test_attention0, test_attention1, test_second_input])
                #train, att0, att1, sec, test, test_att0, test_att1, test_sec  = sess.run([train_image_batch, attention0, attention1, second_input, test_image_batch, test_attention0, test_attention1, test_second_input])
            
            #test, test_att0, test_att1, test_sec, test_att0_0, test_att0_1, test_att1_0, test_att1_1  = sess.run([test_image_batch, test_attention0, test_attention1, test_second_input, att0_0, att0_1, att1_0, att1_1])
            #feature0.append(netendpoints["{0}".format(0)]['feature'])
            #feature1.append(netendpoints["{0}".format(1)]['feature'])
            
            for index in range(test.shape[0]):
                test1 = test[index,:,:,:]
                test_att01 = test_att0[index,:,:,:]
                test_att11 = test_att1[index,:,:,:]
                test_sec1 = test_sec[index,:,:,:]
                #test_att0_01 = test_att0_0[index,:,:,:]
                #test_att0_11 = test_att0_1[index,:,:,:]
                #test_att1_01 = test_att1_0[index,:,:,:]
                #test_att1_11 = test_att1_1[index,:,:,:]

#                 test_att01 = to_heat(test_att01)
#                 test_att11 = to_heat(test_att11)
                scipy.misc.imsave(os.path.join(FLAGS.attention_map, str(batch_idx)+'_'+str(index)+'test.jpg'), test1[:,:,:])
                scipy.misc.imsave(os.path.join(FLAGS.attention_map, str(batch_idx)+'_'+str(index)+'test_att0.jpg'), test_att01[:,:,:])
                scipy.misc.imsave(os.path.join(FLAGS.attention_map, str(batch_idx)+'_'+str(index)+'test_att1.jpg'), test_att11[:,:,:])
                #scipy.misc.imsave(os.path.join(FLAGS.attention_map, str(batch_idx)+'_'+str(index)+'test_att0_0.jpg'), test_att0_01[:,:,:])
                #scipy.misc.imsave(os.path.join(FLAGS.attention_map, str(batch_idx)+'_'+str(index)+'test_att0_1.jpg'), test_att0_11[:,:,:])
                #scipy.misc.imsave(os.path.join(FLAGS.attention_map, str(batch_idx)+'_'+str(index)+'test_att1_0.jpg'), test_att1_01[:,:,:])
                #scipy.misc.imsave(os.path.join(FLAGS.attention_map, str(batch_idx)+'_'+str(index)+'test_att1_1.jpg'), test_att1_11[:,:,:])
                scipy.misc.imsave(os.path.join(FLAGS.attention_map, str(batch_idx)+'_'+str(index)+'test_sec.jpg'), test_sec1[:,:,:])

        #scipy.io.savemat(os.path.join(FLAGS.attention_map, 'feature0.mat'), {'feature_map0': feature0}) 
        #scipy.io.savemat(os.path.join(FLAGS.attention_map, 'feature1.mat'), {'feature_map1': feature1}) 
        
        for i in range(FLAGS.num_networks):
            print(np.mean(np.array(precision_value["{0}".format(i)])))
        print(np.mean(np.array(precision_value["{0}".format(2)])))
        infile.write(str(np.mean(np.array(precision_value["{0}".format(0)])))+' '+str(np.mean(np.array(precision_value["{0}".format(1)])))+' '+str(np.mean(np.array(precision_value["{0}".format(2)])))+'\n')
        infile.close()
