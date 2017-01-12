# logistic regression

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, convolution2d, flatten, batch_norm, max_pool2d, dropout
from tensorflow.python.ops.nn import relu, elu, relu6, sigmoid, tanh, softmax
from tensorflow.python.ops.nn import dynamic_rnn, sparse_softmax_cross_entropy_with_logits

import data

import custom_ops
import conditional_random_fields as crf

tb_log_freq = 50
save_freq = 50
valid_every = 50
max_to_keep = 200
batch_size=64
number_inputs=42
number_outputs=8
num_iterations = 5001
learning_rate = 0.001
clip_norm = 1

num_units_encoder = 400
num_units_l1 = 200
num_units_l2 = 200

data_gen = data.gen_data(num_iterations=num_iterations, batch_size=batch_size)

def model(crf_on):
    print("building model ...")
    with tf.variable_scope('train'):
        print("building train ...")
        # setup
        X_input = tf.placeholder(tf.float32, shape=[None, None, number_inputs], name='X_input')
        X_length = tf.placeholder(tf.int32, shape=[None,], name='X_length')
        t_input = tf.placeholder(tf.int32, shape=[None, None], name='t_input')
        t_input_hot = tf.one_hot(t_input, number_outputs)
        t_mask = tf.placeholder(tf.float32, shape=[None, None], name='t_mask')
        is_training_pl = tf.placeholder(tf.bool)
        # model
        l1 = fully_connected(X_input, num_units_l1)
        l1 = tf.concat(2, [X_input, l1])
        cell_fw = tf.nn.rnn_cell.GRUCell(num_units_encoder)
        cell_bw = tf.nn.rnn_cell.GRUCell(num_units_encoder)
        enc_outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=cell_fw, cell_bw=cell_bw, inputs=l1,
                                                         sequence_length=X_length, dtype=tf.float32)
        enc_outputs = tf.concat(2, enc_outputs)
        enc_outputs = dropout(enc_outputs, is_training=is_training_pl)
        with tf.variable_scope('l2'):
            l2 = fully_connected(enc_outputs, num_units_l2)
        with tf.variable_scope('f'):
            l_f = fully_connected(l2, number_outputs, activation_fn=None)
            f = l_f
        if crf_on:
            print("CRF ON!")
            with tf.variable_scope('g'):
                l_g = fully_connected(l2, number_outputs**2, activation_fn=None)
                batch_size_shp = tf.shape(enc_outputs)[0]
                seq_len_shp = tf.shape(enc_outputs)[1]
                l_g = tf.reshape(l_g, [batch_size_shp, seq_len_shp, number_outputs, number_outputs])
                g = tf.slice(l_g, [0, 0, 0, 0], [-1, seq_len_shp-1, -1, -1])
            nu_alp = crf.forward_pass(f, g, X_length)
            nu_bet = crf.backward_pass(f, g, X_length)
            prediction = crf.log_marginal(nu_alp, nu_bet)
        else:
            print("CRF OFF!")
            prediction = f
        tf.contrib.layers.summarize_variables()

    with tf.variable_scope('metrics'):
        print("building metrics ...")
        if crf_on:
            print("CRF ON!")
            sum_mask = tf.reduce_sum(t_mask, axis=1)
            mean_mask = tf.reduce_mean(sum_mask)
            loss = -crf.log_likelihood(t_input_hot, f, g, nu_alp, nu_bet, X_length) / mean_mask
        else:
            print("CRF OFF!")
            loss = custom_ops.sequence_loss(prediction, t_input, t_mask)
        argmax = tf.to_int32(tf.argmax(prediction, 2))
        correct = tf.to_float(tf.equal(argmax, t_input)) * t_mask
        accuracy = tf.reduce_sum(correct) / tf.reduce_sum(t_mask)
        tf.summary.scalar('train/loss', loss)
        tf.summary.scalar('train/accuracy', accuracy)

    with tf.variable_scope('optimizer'):
        print("building optimizer ...")
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss)
        gradients, variables = zip(*grads_and_vars)
        clipped_gradients, global_norm = tf.clip_by_global_norm(gradients, clip_norm)
        grads_and_vars = zip(clipped_gradients, variables)
        tf.summary.scalar('train/global_gradient_norm', global_norm)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    return X_input, X_length, t_input, t_input_hot, t_mask, is_training_pl, prediction, loss, accuracy, train_op, global_step


def setup_validation_summary():
    acc = tf.placeholder(tf.float32)
    valid_summaries = [
        tf.summary.scalar('validation/acc', acc),
    ]
    return tf.summary.merge(valid_summaries), acc

if __name__ == '__main__':
    model = model()
