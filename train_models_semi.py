"""
Created on Sat May 30 2020
@author: Guo Xiaoqing
"""

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
        embedding = tf.div(Features, embedding_norm, name='norm_embedding')
        weights = tf.get_variable(name='embedding_weights', shape=(embedding.get_shape().as_list()[-1], out_num),
                                  initializer=tf.orthogonal_initializer, dtype=tf.float32)
        print(embedding, weights)
        weights_norm = tf.norm(weights, axis=0, keep_dims=True)
        weights_normed = tf.div(weights, weights_norm, name='norm_weights')
        # cos(theta+m)
        arccos = tf.acos(tf.matmul(embedding, weights_normed, name='cos_t'))
        cos_t = activation_function(arccos, k)
        pred = cos_t
        output = cos_t
        
        if is_cross:
            arccos_mt = tf.acos(tf.matmul(embedding, weights_normed, name='cos_t')) + m
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
        centers = tf.get_variable('centers', dtype=tf.float32, initializer=tf.transpose(weights_normed), trainable=False)
        
        ######regularization term calculate##########
        centers_norm = tf.norm(centers, axis=1, keep_dims=True)
        centers_normed = tf.div(centers, centers_norm, name='norm_weights')
        A = [1] * out_num
        exclude = tf.to_float(tf.matrix_diag(A))
        zeros = array_ops.zeros_like(exclude, dtype=exclude.dtype)
        reg = tf.matmul(centers_normed, weights_normed)
        print(reg)
        reg = tf.where(exclude>0.1, zeros, reg)
        regularization = tf.reduce_sum(reg) / ((out_num-1) * out_num)
        ######regularization term calculate##########

        Labels = tf.reshape(labels, [-1])
    
        centers_norm = tf.norm(centers, axis=1, keep_dims=True)
        centers_normed = tf.div(centers, centers_norm, name='centers_weights')
        centers_batch = tf.gather(centers_normed, Labels)
        Features_norm = tf.norm(Features, axis=1, keep_dims=True)
        Features_normed = tf.div(Features, Features_norm, name='Features_weights')
        centers_batch = tf.gather(centers_normed, Labels)
        numerator = tf.reduce_sum(tf.multiply(Features_normed, centers_batch), axis=-1)
        numerator = tf.acos(numerator)

        f = tf.expand_dims(Features_normed, axis=1)
        f = tf.tile(f,[1,centers_normed.shape[0],1])
        denominator = tf.reduce_sum(tf.multiply(f, centers_normed), axis=-1)
        zeros = array_ops.zeros_like(denominator, dtype=denominator.dtype)
        denominator = tf.acos(denominator)
        denominator = array_ops.where(tf.squeeze(slim.one_hot_encoding(Labels, 3)) > 0.1, zeros, denominator)
        denominator = tf.reduce_sum(denominator, axis=-1)
        
        #denominator = 1e-8 + tf.reduce_sum(denominator, axis=-1) - numerator
        loss_weight = (out_num-1) * numerator/denominator
        diff = tf.gather(centers, Labels) - Features
    
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
    scale = 64.
    scale_pred = 4.
    margin = 0.5
    net_features, net_logits, net_endpoints, net_raw_loss, net_pred, net_features, c_update_op = {}, {}, {}, {}, {}, {}, {}
    weight_loss, net_regularization, net_pred, net_predict = {}, {}, {}, {}
    for i in range(FLAGS.num_networks):
        net_features["{0}".format(i)], net_endpoints["{0}".format(i)] = \
            network_fn["{0}".format(i)](images, reuse=reuse, is_training=is_training, scope=('dmlnet_%d' % i))
        
        if is_cross:
            net_predict["{0}".format(i)], net_logits["{0}".format(i)], net_regularization["{0}".format(i)], weight, c_update_op["{0}".format(i)] = DA_loss(net_features["{0}".format(i)], tf.argmax(labels, axis=1), FLAGS.num_classes, s=scale, m=margin, scope = ('dmlnet_%d' % i), is_cross=is_cross, reuse=reuse)
            
            net_raw_loss["{0}".format(i)] = tf.losses.softmax_cross_entropy(
                logits=net_logits["{0}".format(i)], onehot_labels=labels,
                label_smoothing=FLAGS.label_smoothing, weights=1.0) + tf.reduce_mean(weight)
            kl_weight = 1.0

        else:
            print('semi_data!')
            net_predict["{0}".format(i)], _, _, _, _ = DA_loss(net_features["{0}".format(i)], tf.argmax(labels, axis=1), FLAGS.num_classes, s=scale, m=margin, scope = ('dmlnet_%d' % i), is_cross=is_cross, reuse=reuse)
            net_pred["{0}".format(i)]  = layers.softmax(scale_pred*net_predict["{0}".format(i)], scope='predictions')
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
        net_pred["{0}".format(i)]  = layers.softmax(scale_pred*net_predict["{0}".format(i)], scope='predictions')
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
            
        #########################################
        # Configure the optimization procedure. #
        #########################################
        with tf.device(deploy_config.optimizer_device()):
            net_opt, semi_net_opt = {}, {}
            for i in range(FLAGS.num_networks):
                net_opt["{0}".format(i)] = tf.train.AdamOptimizer(FLAGS.learning_rate,
                                                                  beta1=FLAGS.adam_beta1,
                                                                  beta2=FLAGS.adam_beta2,
                                                                  epsilon=FLAGS.opt_epsilon)
               
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
        train_image_batch, train_label_batch = utils.get_image_label_batch(FLAGS, shuffle=True, name='train4')
        semi_image_batch, semi_label_batch = utils.get_image_label_batch(FLAGS, shuffle=True, name='semi')
        test_image_batch, test_label_batch = utils.get_image_label_batch(FLAGS, shuffle=False, name='test4')
 
        train_label_batch = slim.one_hot_encoding(train_label_batch, FLAGS.num_classes)
        semi_label_batch = slim.one_hot_encoding(semi_label_batch, FLAGS.num_classes)
        test_label_batch = slim.one_hot_encoding(test_label_batch, FLAGS.num_classes)
            
        precision, test_precision, net_var_list, net_grads, net_update_ops = {}, {}, {}, {}, {}
        semi_net_grads = {}

        with tf.name_scope('tower') as scope:
            with tf.variable_scope(tf.get_variable_scope()):
                net_loss, tc_update_op, net_pred, attention0, attention1, second_input = _tower_loss(network_fn, train_image_batch, train_label_batch, is_cross = True, reuse=False, is_training=True)
                semi_net_loss, sc_update_op, semi_net_pred, semi_attention0, semi_attention1, semi_second_input = _tower_loss(network_fn, semi_image_batch, semi_label_batch, is_cross = False, reuse=True, is_training=True)
                test_net_loss, _, test_net_pred, test_attention0, test_attention1, test_second_input = _tower_loss(network_fn, test_image_batch, test_label_batch, is_cross = True, reuse=True, is_training=False)

                truth = tf.argmax(train_label_batch, axis=1)
                test_truth = tf.argmax(test_label_batch, axis=1)

                # Reuse variables for the next tower.
                #tf.get_variable_scope().reuse_variables()

                # Retain the summaries from the final tower.
                summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                var_list = tf.trainable_variables()

                for i in range(FLAGS.num_networks):
                    predictions = tf.argmax(net_pred["{0}".format(i)], axis=1)
                    test_predictions = tf.argmax(test_net_pred["{0}".format(i)], axis=1)
                    precision["{0}".format(i)] = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
                    test_precision["{0}".format(i)] = tf.reduce_mean(tf.to_float(tf.equal(test_predictions, test_truth)))

                    # Add a summary to track the training precision.
                    #summaries.append(tf.summary.scalar('precision_%d' % i, precision["{0}".format(i)]))
                    #summaries.append(tf.summary.scalar('test_precision_%d' % i, test_precision["{0}".format(i)]))

                    net_update_ops["{0}".format(i)] = \
                                tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=('%sdmlnet_%d' % (scope, i)))

                    net_var_list["{0}".format(i)] = \
                                    [var for var in var_list if 'dmlnet_%d' % i in var.name]

                    net_grads["{0}".format(i)] = net_opt["{0}".format(i)].compute_gradients(
                                    net_loss["{0}".format(i)], var_list=net_var_list["{0}".format(i)])
                
                    semi_net_grads["{0}".format(i)] = net_opt["{0}".format(i)].compute_gradients(
                                    semi_net_loss["{0}".format(i)], var_list=net_var_list["{0}".format(i)])

        # Add histograms for histogram and trainable variables.
        #for i in range(FLAGS.num_networks):
        #    for grad, var in net_grads["{0}".format(i)]:
        #        if grad is not None:
        #            if 'gamma' in var.name:
        #                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        #for i in range(FLAGS.num_networks):
        #    for grad, var in semi_net_grads["{0}".format(i)]:
        #        if grad is not None:
        #            summaries.append(tf.summary.histogram(var.op.name + '/semi_gradients', grad))

        #for var in tf.trainable_variables():
        #    if 'gamma' in var.name:
        #        summaries.append(tf.summary.histogram(var.op.name, var))

        #################################
        # Configure the moving averages #
        #################################

        if FLAGS.moving_average_decay:
            moving_average_variables = {}
            all_moving_average_variables = slim.get_model_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                FLAGS.moving_average_decay, global_step)
            for i in range(FLAGS.num_networks):
                moving_average_variables["{0}".format(i)] = \
                    [var for var in all_moving_average_variables if 'dmlnet_%d' % i in var.name]
                net_update_ops["{0}".format(i)].append(
                    variable_averages.apply(moving_average_variables["{0}".format(i)]))

        # Apply the gradients to adjust the shared variables.
        net_grad_updates, net_train_op, semi_net_grad_updates, semi_net_train_op = {}, {}, {}, {}
        for i in range(FLAGS.num_networks):
            net_grad_updates["{0}".format(i)] = net_opt["{0}".format(i)].apply_gradients(
                net_grads["{0}".format(i)], global_step=global_step)
            semi_net_grad_updates["{0}".format(i)] = net_opt["{0}".format(i)].apply_gradients(
                semi_net_grads["{0}".format(i)], global_step=global_step)
            net_update_ops["{0}".format(i)].append(net_grad_updates["{0}".format(i)])
            net_update_ops["{0}".format(i)].append(semi_net_grad_updates["{0}".format(i)])
            # Group all updates to into a single train op.
            net_train_op["{0}".format(i)] = tf.group(*net_update_ops["{0}".format(i)])
            
        '''# Apply the gradients to adjust the shared variables.
        net_train_op, semi_net_train_op = {}, {}
        for i in range(FLAGS.num_networks):
            net_train_op["{0}".format(i)] = net_opt["{0}".format(i)].minimize(net_loss["{0}".format(i)], global_step=global_step, var_list=net_var_list["{0}".format(i)])
            #semi_net_train_op["{0}".format(i)] = semi_net_opt["{0}".format(i)].minimize(semi_net_loss["{0}".format(i)],global_step=global_step, var_list=net_var_list["{0}".format(i)])'''

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        #summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85),
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        
        #load_fn = slim.assign_from_checkpoint_fn('./WCE_densenet55/checkpoint/model.ckpt-95',tf.global_variables(),ignore_missing_vars=True)
        
        #load_fn = slim.assign_from_checkpoint_fn(os.path.join(FLAGS.checkpoint_dir, 'model.ckpt-95'),tf.global_variables(),ignore_missing_vars=True)
        #load_fn(sess)
        
        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        #summary_writer = tf.summary.FileWriter(
        #    os.path.join(FLAGS.log_dir),
        #    graph=sess.graph)

        net_loss_value, test_net_loss_value, precision_value, Accuracy = {}, {}, {}, {}

        parameters = utils.count_trainable_params()
        print("Total training params: %.1fM \r\n" % (parameters / 1e6))
                  
        start_time = time.time() 
        counter = 0
        best_acc = 0.
        infile = open(os.path.join(FLAGS.log_dir, 'log.txt'),'w')
        batch_count = np.int32(FLAGS.dataset_size * 8 / FLAGS.batch_size)
        for epoch in range(1, 1+FLAGS.max_number_of_epochs):
            if (epoch) % 30 == 0: 
                FLAGS.learning_rate = FLAGS.learning_rate * 0.1
                
            for batch_idx in range(batch_count):
                counter += 1
                for i in range(FLAGS.num_networks):
                    _, net_loss_value["{0}".format(i)], precision_value["{0}".format(i)], _, _ = \
                        sess.run([net_train_op["{0}".format(i)], net_loss["{0}".format(i)],
                              precision["{0}".format(i)], tc_update_op["{0}".format(i)], sc_update_op["{0}".format(i)]])
                    assert not np.isnan(net_loss_value["{0}".format(i)]), 'Model diverged with loss = NaN'
                    #if epoch >= 20:
                    #    _ = sess.run([sc_update_op["{0}".format(i)]])
                    
                if batch_idx == 0:
                    Accuracy["{0}".format(0)] = 0.
                    Accuracy["{0}".format(1)] = 0.
                    for test_batch in range(np.int32(FLAGS.testing_dataset_size * 8 / FLAGS.batch_size)):
                        for i in range(FLAGS.num_networks):
                            test_precision_value = sess.run([test_precision["{0}".format(i)]])
                            Accuracy["{0}".format(i)] += np.float32(test_precision_value)
                    for i in range(FLAGS.num_networks):
                        Accuracy["{0}".format(i)] = Accuracy["{0}".format(i)]/np.float32(FLAGS.testing_dataset_size * 8 / FLAGS.batch_size)
                                          
                    format_str = 'Epoch:[%2d][%3d/%3d] time:%4.2f,net0_loss = %.3f, net0_acc = %.3f, net0_test_acc = %.3f,    net1_loss = %.3f, net1_acc = %.3f, net1_test_acc = %.3f'
                    print(format_str % (epoch, batch_idx,batch_count, time.time()-start_time, net_loss_value["{0}".format(0)], precision_value["{0}".format(0)], Accuracy["{0}".format(0)],net_loss_value["{0}".format(1)],precision_value["{0}".format(1)], Accuracy["{0}".format(1)]))
                    
                    infile.write(format_str % (epoch, batch_idx,batch_count, time.time()-start_time, net_loss_value["{0}".format(0)], precision_value["{0}".format(0)], np.float32(Accuracy["{0}".format(0)]),net_loss_value["{0}".format(1)],precision_value["{0}".format(1)], np.float32(Accuracy["{0}".format(1)])))
                    infile.write('\n')
                    
                    #format_str = 'Epoch: [%3d] [%3d/%3d] time: %4.4f, net0_loss = %.5f, net0_acc = %.4f, net0_test_loss = %.5f, net0_test_acc = %.4f'
                    #print(format_str % (epoch, batch_idx,batch_count, time.time()-start_time, net_loss_value["{0}".format(0)],precision_value["{0}".format(0)],test_net_loss_value["{0}".format(0)], np.float32(test_precision_value["{0}".format(0)])))
                    
                    #format_str = 'Epoch: [%3d] [%3d/%3d] time: %4.4f, net0_loss = %.5f, net0_acc = %.4f'
                    #print(format_str % (epoch, batch_idx,batch_count, time.time()-start_time, net_loss_value["{0}".format(1)],
                    #     precision_value["{0}".format(1)]))
                    if Accuracy["{0}".format(1)] > best_acc and epoch >= 75:
                        checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=0)
                        best_acc = Accuracy["{0}".format(1)]

                if batch_idx == 0:
                    #train, att0, att1, sec, semi, semi_att0, semi_att1, semi_sec, test, test_att0, test_att1, test_sec  = sess.run([train_image_batch, attention0, attention1, second_input, semi_image_batch, semi_attention0, semi_attention1, semi_second_input, test_image_batch, test_attention0, test_attention1, test_second_input])
                    #train, att0, att1, sec, test, test_att0, test_att1, test_sec  = sess.run([train_image_batch, attention0, attention1, second_input, test_image_batch, test_attention0, test_attention1, test_second_input])
                    test, test_att0, test_att1, test_sec  = sess.run([test_image_batch, test_attention0, test_attention1, test_second_input])

                    tot_num_samples = FLAGS.batch_size
                    manifold_h = int(np.floor(np.sqrt(tot_num_samples)))
                    manifold_w = int(np.floor(np.sqrt(tot_num_samples)))
                    '''save_images(train[:manifold_h * manifold_w, :,:,:],[manifold_h, manifold_w],os.path.join(FLAGS.attention_map, str(epoch)+'_'+str(batch_idx)+'.jpg'))
                    save_images(att0[:manifold_h * manifold_w, :,:,:],[manifold_h, manifold_w],os.path.join(FLAGS.attention_map, str(epoch)+'_'+str(batch_idx)+'att0.jpg'))       
                    #for aa1 in range(att1.shape[-1]):
                    #    a1 = att1[:manifold_h * manifold_w, :,:,aa1]
                    #    a1 = np.stack([a1, a1, a1]).transpose([1, 2, 3, 0])
                    #    save_images(a1,[manifold_h, manifold_w],os.path.join(FLAGS.attention_map, str(epoch)+'_'+str(batch_idx)+'_'+str(aa1)+'att1.jpg')) 
                    save_images(att1[:manifold_h * manifold_w, :,:,:],[manifold_h, manifold_w],os.path.join(FLAGS.attention_map, str(epoch)+'_'+str(batch_idx)+'att1.jpg')) 
                    save_images(sec[:manifold_h * manifold_w, :,:,:],[manifold_h, manifold_w],os.path.join(FLAGS.attention_map, str(epoch)+'_'+str(batch_idx)+'sec.jpg'))       
                    
                    save_images(semi[:manifold_h * manifold_w, :,:,:],[manifold_h, manifold_w],os.path.join(FLAGS.attention_map, str(epoch)+'_'+str(batch_idx)+'semi.jpg'))
                    save_images(semi_att0[:manifold_h * manifold_w, :,:,:],[manifold_h, manifold_w],os.path.join(FLAGS.attention_map, str(epoch)+'_'+str(batch_idx)+'semi_att0.jpg'))       
                    save_images(semi_att1[:manifold_h * manifold_w, :,:,:],[manifold_h, manifold_w],os.path.join(FLAGS.attention_map, str(epoch)+'_'+str(batch_idx)+'semi_att1.jpg'))       
                    save_images(semi_sec[:manifold_h * manifold_w, :,:,:],[manifold_h, manifold_w],os.path.join(FLAGS.attention_map, str(epoch)+'_'+str(batch_idx)+'semi_sec.jpg'))'''
                    
                    save_images(test[:manifold_h * manifold_w, :,:,:],[manifold_h, manifold_w],os.path.join(FLAGS.attention_map, str(epoch)+'_'+str(batch_idx)+'test.jpg'))
                    save_images(test_att0[:manifold_h * manifold_w, :,:,:],[manifold_h, manifold_w],os.path.join(FLAGS.attention_map, str(epoch)+'_'+str(batch_idx)+'test_att0.jpg'))       
                    save_images(test_att1[:manifold_h * manifold_w, :,:,:],[manifold_h, manifold_w],os.path.join(FLAGS.attention_map, str(epoch)+'_'+str(batch_idx)+'test_att1.jpg'))       
                    save_images(test_sec[:manifold_h * manifold_w, :,:,:],[manifold_h, manifold_w],os.path.join(FLAGS.attention_map, str(epoch)+'_'+str(batch_idx)+'test_sec.jpg'))
                    
                #summary_str = sess.run(summary_op)
                #summary_writer.add_summary(summary_str, counter)
                
            # Save the model checkpoint periodically.
            #if epoch % FLAGS.ckpt_steps == 0 or epoch == FLAGS.max_number_of_epochs:
            #    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, 'model.ckpt')
            #    saver.save(sess, checkpoint_path, global_step=epoch)

