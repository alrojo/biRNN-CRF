import numpy as np
import os.path
import subprocess

import utils

TRAIN_PATH = 'data/cullpdb+profile_6133_filtered.npy.gz'
TEST_PATH = 'data/cb513+profile_split1.npy.gz'
##### TRAIN DATA #####

def get_train(seq_len=None):
  if not os.path.isfile(TRAIN_PATH):
    print("Train path is not downloaded ...")
    subprocess.call("./download_train.sh", shell=True)
  else:
    print("Train path is downloaded ...")
  print("Loading train data ...")
  X_in = utils.load_gz(TRAIN_PATH)
  X = np.reshape(X_in,(5534,700,57))
  del X_in
  X = X[:,:,:]
  labels = X[:,:,22:30]
  mask = X[:,:,30] * -1 + 1

  a = np.arange(0,21)
  b = np.arange(35,56)
  c = np.hstack((a,b))
  X = X[:,:,c]

  # getting meta
  num_seqs = np.size(X,0)
  seqlen = np.size(X,1)
  d = np.size(X,2)
  num_classes = 8

  #### REMAKING LABELS ####
  X = X.astype("float32")
  mask = mask.astype("float32")
  # Dummy -> concat
  vals = np.arange(0,8)
  labels_new = np.zeros((num_seqs,seqlen))
  for i in xrange(np.size(labels,axis=0)):
    labels_new[i,:] = np.dot(labels[i,:,:], vals)
  labels_new = labels_new.astype('int32')
  labels = labels_new

  print("Loading splits ...")
  ##### SPLITS #####
  # getting splits (cannot run before splits are made)
  #split = np.load("data/split.pkl")

  seq_names = np.arange(0,num_seqs)
  #np.random.shuffle(seq_names)

  X_train = X[seq_names[0:5278]]
  X_valid = X[seq_names[5278:5534]]
  labels_train = labels[seq_names[0:5278]]
  labels_valid = labels[seq_names[5278:5534]]
  mask_train = mask[seq_names[0:5278]]
  mask_valid = mask[seq_names[5278:5534]]
  num_seq_train = np.size(X_train,0)
  num_seq_valid = np.size(X_valid,0)
  if seq_len is not None:
    X_train = X_train[:, :seq_len]
    X_valid = X_valid[:, :seq_len]
    labels_train = labels_train[:, :seq_len]
    labels_valid = labels_valid[:, :seq_len]
    mask_train = mask_train[:, :seq_len]
    mask_valid = mask_valid[:, :seq_len]
  len_train = np.sum(mask_train, axis=1)
  len_valid = np.sum(mask_valid, axis=1)
  return X_train, X_valid, labels_train, labels_valid, mask_train, \
      mask_valid, len_train, len_valid, num_seq_train
#del split
##### TEST DATA #####

def get_test(seq_len=None):
  if not os.path.isfile(TEST_PATH):
    subprocess.call("./download_test.sh", shell=True)
  print("Loading test data ...")
  X_test_in = utils.load_gz(TEST_PATH)
  X_test = np.reshape(X_test_in,(514,700,57))
  del X_test_in
  X_test = X_test[:,:,:].astype("float32")
  labels_test = X_test[:,:,22:30].astype('int32')
  mask_test = X_test[:,:,30].astype("float32") * -1 + 1

  a = np.arange(0,21)
  b = np.arange(35,56)
  c = np.hstack((a,b))
  X_test = X_test[:,:,c]

  # getting meta
  seqlen = np.size(X_test,1)
  d = np.size(X_test,2)
  num_classes = 8
  num_seq_test = np.size(X_test,0)
  del a, b, c

  ## DUMMY -> CONCAT ##
  vals = np.arange(0,8)
  labels_new = np.zeros((num_seq_test,seqlen))
  for i in xrange(np.size(labels_test,axis=0)):
    labels_new[i,:] = np.dot(labels_test[i,:,:], vals)
  labels_new = labels_new.astype('int32')
  labels_test = labels_new

  ### ADDING BATCH PADDING ###

  X_add = np.zeros((126,seqlen,d))
  label_add = np.zeros((126,seqlen))
  mask_add = np.zeros((126,seqlen))

  X_test = np.concatenate((X_test,X_add), axis=0).astype("float32")
  labels_test = np.concatenate((labels_test, label_add), axis=0).astype('int32')
  mask_test = np.concatenate((mask_test, mask_add), axis=0).astype("float32")
  if seq_len is not None:
    X_test = X_test[:, :seq_len]
    labels_test = labels_test[:, :seq_len]
    mask_test = mask_test[:, :seq_len]
  len_test = np.sum(mask_test, axis=1)
  return X_test, mask_test, labels_test, num_seq_test, len_test
