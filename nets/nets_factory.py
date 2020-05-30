"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import tensorflow as tf
from nets import densenet
from nets import ResNeXt
from nets import resnet_v2
from nets import vgg
from nets import SE_ResNeXt

slim = tf.contrib.slim

networks_map = {'densenet': densenet.discriminator}

def get_network_fn(name, num_classes, weight_decay=0.0):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature:
          logits, end_points = network_fn(images)
    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]

    @functools.wraps(func)
    def network_fn(images, reuse=False, is_training=False, scope=None):
        return densenet.discriminator(images, num_classes, trainable=is_training, reuse=reuse, scope=scope)
        #return vgg.vgg16(images, num_classes, trainable=is_training, reuse=reuse, scope=scope)
        #return resnet_v2.resnet_v2_50(images, num_classes=num_classes, is_training=is_training, reuse=reuse, scope=scope)
        #return ResNeXt.ResNeXt(images, num_classes, training=is_training, reuse=reuse, scope=scope).model        
        #return SE_ResNeXt.SE_ResNeXt(images, num_classes, training=is_training, reuse=reuse, scope=scope).model        

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
