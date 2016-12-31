'''Restores a model from checkpoint and evaluates it on CIFAR-10'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import time
from datetime import datetime
import data_helpers
import two_layer_fc

# Basic model parameters as external flags.
flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('hidden1', 120, 'Number of units in hidden layer 1.')
flags.DEFINE_string('train_dir', 'tf_logs',
  'Directory to put the training data.')
flags.DEFINE_float('reg_constant', 0.1, 'Regularization constant.')

FLAGS._parse_flags()
print('\nParameters:')
for attr, value in sorted(FLAGS.__flags.items()):
  print('{} = {}'.format(attr.upper(), value))
print()

IMAGE_PIXELS = 3072
CLASSES = 10

beginTime = time.time()

data_sets = data_helpers.load_data()

images_placeholder = tf.placeholder(tf.float32, shape=[None, IMAGE_PIXELS],
  name='images')

labels_placeholder = tf.placeholder(tf.int64, shape=[None], name='image-labels')

logits = two_layer_fc.inference(images_placeholder, IMAGE_PIXELS,
  FLAGS.hidden1, CLASSES, reg_constant=FLAGS.reg_constant)

global_step = tf.Variable(0, name="global_step", trainable=False)

accuracy = two_layer_fc.evaluation(logits, labels_placeholder)

saver = tf.train.Saver()

with tf.Session() as sess:
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and ckpt.model_checkpoint_path:
    print('Restoring variables from checkpoint')
    saver.restore(sess, ckpt.model_checkpoint_path)
    current_step = tf.train.global_step(sess, global_step)
    print('Current step: {}'.format(current_step))

  print('Test accuracy {:g}'.format(accuracy.eval(
    feed_dict={ images_placeholder: data_sets['images_test'],
                labels_placeholder: data_sets['labels_test']}
  )))

endTime = time.time()
print('Total time: {:5.2f}s'.format(endTime - beginTime))
