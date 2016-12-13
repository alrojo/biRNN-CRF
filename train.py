# train

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import importlib
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
import os

import time
import utils

if len(sys.argv) != 2:
    sys.exit("Usage: python train.py <config_name>")

config_name = sys.argv[1]

config = importlib.import_module("configurations.%s" % config_name)
print("Using configurations: '%s'" % config_name)

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
experiment_id = "%s-%s" % (config_name, timestamp)
metadata_path = "metadata/dump_%s" % experiment_id

print("Experiment id: %s" % experiment_id)

print("setting up tensorboard and weight saving")

SAVER_PATH = {
    'base': 'train/',
    'checkpoint': 'checkpoints/',
    'log': 'log/',
    'test': 'test/'
}

local_path = os.path.join(SAVER_PATH['base'], config_name)
summary_path = os.path.join(local_path, SAVER_PATH['log'])

X_input, X_length, t_input, t_input_hot, t_mask, is_training_pl, prediction, loss, accuracy, train_op, global_step = config.model()
print("Model loaded")

if config.save_freq:
    checkpoint_saver = tf.train.Saver(max_to_keep=config.max_to_keep)
    checkpoint_path = os.path.join(local_path, SAVER_PATH['checkpoint'])
    checkpoint_file_path = os.path.join(checkpoint_path, 'checkpoint')
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_path)
    print("Saving checkpoints at: %s" % checkpoint_path)
else:
    print("WARNING: Not saving checkpoints!")

data_gen = config.data_gen
train_losses = []
train_accs = []

with tf.Session() as sess:
    if config.tb_log_freq and config_name:
        if not os.path.exists(summary_path) and config.tb_log_freq:
            os.makedirs(summary_path)
        summary_writer = tf.train.SummaryWriter(summary_path, sess.graph)
    if latest_checkpoint:
        checkpoint_saver.restore(sess, latest_checkpoint)
    else:
        sess.run(tf.initialize_all_variables())

    # prepare summary operations and summary writer
    summaries = tf.merge_all_summaries()
    val_summaries, valid_accs_pl = config.setup_validation_summary()

    combined_time = 0.0
    for idx, batch in enumerate(data_gen.gen_train()):
        train_fetches = [train_op, loss, accuracy, summaries] # which ops to run
        train_feed_dict = {X_input: batch['X'], t_input: batch['t'], X_length: batch['length'], t_mask: batch['mask'], is_training_pl: True} # data input point in the graph
        start_time = time.time()
        res = tuple(sess.run(fetches=train_fetches, feed_dict=train_feed_dict))
        elapsed = time.time() - start_time
        combined_time += elapsed
        _, train_loss, train_acc, train_sums = res
        if summary_writer and idx % config.tb_log_freq == 0:
            print(" train_loss,", train_loss)
            print(" train_acc,", train_acc)
            summary_writer.add_summary(train_sums, idx)

        if checkpoint_saver and idx % config.save_freq == 0:
            checkpoint_saver.save(sess, checkpoint_file_path, global_step)

        if (idx % config.valid_every) == 0:
            def validate(sess):
                gen = data_gen.gen_valid
                valid_masks = []
                valid_outs = []
                valid_targets = []
                sum = 0
                for batch, i in gen():
                    valid_fetches = [prediction]
                    valid_feed_dict = {X_input: batch['X'], t_input: batch['t'], X_length: batch['length'], t_mask: batch['mask'], is_training_pl: False}
                    valid_out = sess.run(fetches=valid_fetches, feed_dict=valid_feed_dict)[0]
                    h_out = np.zeros((i, 700, 8), dtype="float32")
                    h_out[:, :valid_out.shape[1], :] = valid_out
                    h_mask = np.zeros((i, 700), dtype="float32")
                    h_mask[:, :valid_out.shape[1]] = batch['mask']
                    h_targets = np.zeros((i, 700), dtype="int32")
                    h_targets[:, :valid_out.shape[1]] = batch['t']
                    valid_masks.append(h_mask)
                    valid_targets.append(h_targets)
                    valid_outs.append(h_out)
                    sum += i
                valid_outs = np.concatenate(valid_outs, axis=0)[:sum]
                valid_targets = np.concatenate(valid_targets, axis=0)[:sum]
                valid_masks = np.concatenate(valid_masks, axis=0)[:sum]
                valid_accs = utils.proteins_acc(valid_outs, valid_targets, valid_masks)
                print(" valid_accs,", valid_accs)
                sum_fetches = [val_summaries, global_step]
                sum_feed_dict = {
                    valid_accs_pl: valid_accs,
                }
                summaries, i = sess.run(sum_fetches, sum_feed_dict)
                summary_writer.add_summary(summaries, i)
            print("Validating!")
            validate(sess)
            print("Continue training..")
