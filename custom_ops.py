from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops

def sequence_loss(logits, targets, weights,
                  average_across_timesteps=True, average_across_batch=True,
                  softmax_loss_function=None, name=None):
  """Weighted cross-entropy loss for a sequence of logits (per example).
  Args:
    logits: 3D Tensor of shape [batch_size x sequence_length x num_decoder_symbols].
    targets: 2D Tensor of shape [batch_size x sequence_length].
    weights: 2D Tensor of shape [batch_size x sequence_length].
    average_across_timesteps: If set, divide the returned cost by the total
      label weight.
    average_across_batch: If set, divide the returned cost by the batch size.
    softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
      to be used instead of the standard softmax (the default if this is None).
    name: Optional name for this operation, defaults to "sequence_loss".
  Returns:
    A scalar float Tensor: The average log-perplexity per symbol (weighted).
  Raises:
    ValueError: logits does not have 3 dimensions or targets does not have 2
                dimensions or weights does not have 2 dimensions.
  """
  with ops.name_scope(name, "sequence_loss", [logits, targets, weights]):
    num_classes = array_ops.shape(logits)[2]
    probs_flat = array_ops.reshape(logits, [-1, num_classes])
    targets = array_ops.reshape(targets, [-1])
    if softmax_loss_function is None:
      crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
        probs_flat, targets)
    else:
      crossent = softmax_loss_function(probs_flat, targets)
      crossent = crossent * tf.reshape(weights, [-1])
    if average_across_timesteps and average_across_batch:
      crossent = math_ops.reduce_sum(crossent)
      total_size = math_ops.reduce_sum(weights)
      total_size += 1e-12 # to avoid division by 0 for all-0 weights
      crossent /= total_size
    if average_across_timesteps and not average_across_batch:
      crossent = math_ops.reduce_sum(crossent, reduction_indices=[1])
      total_size = math_ops.reduce_sum(weights, reduction_indices=[1])
      total_size += 1e-12 # to avoid division by 0 for all-0 weights
      crossent /= total_size
    if not average_across_timesteps and average_across_batch:
      crossent = math_ops.reduce_sum(crossent, reduction_indices=[0])
      total_size = math_ops.reduce_sum(weights, reduction_indices=[0])
      total_size += 1e-12 # to avoid division by 0 for all-0 weights
      crossent /= total_size
    return crossent
