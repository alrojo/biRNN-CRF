import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops as ta_ops


def log_sum_exp(z, axis=1, name=None):
    with ops.name_scope(name, "log_sum_exp", [z, axis]):
        zmax = tf.reduce_max(z, axis=axis, name="zmax")
        return zmax + tf.log(tf.reduce_sum(
            tf.exp(z - tf.expand_dims(zmax, axis)), axis))


def forward_step(nu, f, g, axis, name=None):
    with ops.name_scope(name, "forward_step", [nu, f, g, axis]):
        return f + log_sum_exp(g + tf.expand_dims(nu, axis+1), axis=axis)


def forward_pass(f, g, sequence_length=None, time_major=False, name=None):
    with ops.name_scope(name, "forward_pass", [f, g, time_major]):
        if sequence_length is None:
            sequence_length = tf.to_int32(tf.reduce_sum(
                tf.ones((tf.shape(f)[:2])), axis=1))
        if not time_major:
            # [batch_size, seq_len, ...] -> [seq_len, batch_size, ...]
            f = tf.transpose(f, perm=[1, 0, 2])
            g = tf.transpose(g, perm=[1, 0, 2, 3])

        # setup loop variables
        f_seq_len = array_ops.shape(f)[0]
        g_seq_len = array_ops.shape(g)[0]
        f_ta = ta_ops.TensorArray(tf.float32, size=f_seq_len, name="f_ta")
        g_ta = ta_ops.TensorArray(tf.float32, size=g_seq_len, name="g_ta")
        f_ta = f_ta.unstack(f)
        g_ta = g_ta.unstack(g)


        max_sequence_length = tf.reduce_max(sequence_length)
        min_sequence_length = tf.reduce_min(sequence_length)
        nu_ta = ta_ops.TensorArray(tf.float32, size=max_sequence_length, name="nu_ta")

        def forward_cond(time, nu_state, nu_ta_t):
            return tf.less(time, max_sequence_length)
    
        def forward_body(time, nu_state, nu_ta_t):
            def zero_state():
                return nu_state
            def normal():
                return forward_step(nu_state, f_ta.read(time),
                                    g_ta.read(time-1), axis=1)
            new_nu_state = tf.cond(
                tf.greater(time, 0),
                normal,
                zero_state, name="new_nu_state")

            def updateall():
                return new_nu_state

            def updatesome():
                return tf.where(tf.less(time, sequence_length),
                                 new_nu_state,
                                 tf.zeros(tf.shape(new_nu_state),
                                          dtype=tf.float32))

            proposed_state = tf.cond(tf.less(time, min_sequence_length),
                                     updateall, updatesome)
            nu_ta_t = nu_ta_t.write(time, proposed_state)

            return (time+1, new_nu_state, nu_ta_t)

        time = tf.constant(0, name="time")
        loop_vars = [time, f_ta.read(tf.constant(0)), nu_ta]

        time, state, nu_ta = tf.while_loop(forward_cond, forward_body,
                                           loop_vars)
        nu = nu_ta.stack()

        if not time_major:
            nu = tf.transpose(nu, perm=[1, 0, 2])

        return nu


def backward_step(nu, f, g, axis, name=None):
    with ops.name_scope(name, "backward_step", [nu, f, g, axis]):
        return log_sum_exp(tf.expand_dims(f + nu, axis) + g, axis=axis+1)


def backward_pass(f, g, sequence_length=None, time_major=False, name=None):
    with ops.name_scope(name, "backward_pass", [f, g, time_major]):
        if sequence_length is None:
            sequence_length = tf.to_int32(tf.reduce_sum(
                tf.ones((tf.shape(f)[:2])), axis=1))
        f = tf.reverse_sequence(f, sequence_length, 1)
        g = tf.reverse_sequence(g, sequence_length-1, 1)
        if not time_major:
            # [batch_size, seq_len, ...] -> [seq_len, batch_size, ...]
            f = tf.transpose(f, perm=[1, 0, 2])
            g = tf.transpose(g, perm=[1, 0, 2, 3])

        # setup loop variables
        f_seq_len = array_ops.shape(f)[0]
        g_seq_len = array_ops.shape(g)[0]
        f_ta = ta_ops.TensorArray(tf.float32, size=f_seq_len, name="f_ta")
        g_ta = ta_ops.TensorArray(tf.float32, size=g_seq_len, name="g_ta")
        f_ta = f_ta.unstack(f)
        g_ta = g_ta.unstack(g)
        max_sequence_length = tf.reduce_max(sequence_length)
        min_sequence_length = tf.reduce_min(sequence_length)
        nu_ta = ta_ops.TensorArray(tf.float32, size=max_sequence_length, name="nu_ta")

        def backward_cond(time, nu_state, nu_ta_t):
            return tf.less(time, max_sequence_length)

        def backward_body(time, nu_state, nu_ta_t):
            def zero_state():
                return nu_state
            def normal():
                back = backward_step(nu_state,
                                     f_ta.read(time-tf.constant(1)),
                                     g_ta.read(time-tf.constant(1)),
                                     axis=1)
                back.set_shape(nu_state.get_shape())
                return back

            new_nu_state = tf.cond(
                tf.greater(time, 0),
                normal,
                zero_state, name="new_nu_state")

            def updateall():
                return new_nu_state
            def updatesome():
                return tf.where(tf.less(time, sequence_length),
                                new_nu_state,
                                tf.zeros_like(new_nu_state, dtype=tf.float32))

            proposed_state = tf.cond(tf.less(time, min_sequence_length),
                                     updateall, updatesome)
            nu_ta_t = nu_ta_t.write(time, proposed_state)

            return (time+1, new_nu_state, nu_ta_t)

        time = tf.constant(0, name="time")
        loop_vars = [time, tf.zeros(tf.shape(f)[1:]), nu_ta]

        time, state, nu_ta = tf.while_loop(backward_cond, backward_body,
                                           loop_vars)
        nu = nu_ta.stack()

        if not time_major:
            nu = tf.transpose(nu, perm=[1, 0, 2])
        nu = tf.reverse_sequence(nu, sequence_length, 1)

        return nu


def logZ(nu_alp, nu_bet, index=0, time_major=False, name=None):
    with ops.name_scope(name, "logZ", [nu_alp, nu_bet, index, time_major]):
        if not time_major:
            # [batch_size, seq_len, ...] -> [seq_len, batch_size, ...]
            nu_alp = tf.transpose(nu_alp, perm=[1, 0, 2])
            nu_bet = tf.transpose(nu_bet, perm=[1, 0, 2])
        nu_alp_slice = tf.squeeze(tf.slice(nu_alp, [index, 0, 0],
                                           [1, -1, -1]), squeeze_dims=0)
        nu_bet_slice = tf.squeeze(tf.slice(nu_bet, [index, 0, 0],
                                           [1, -1, -1]), squeeze_dims=0)
        sum_slice = nu_alp_slice+nu_bet_slice
        return log_sum_exp(sum_slice, axis=1)


def log_likelihood(y, f, g, nu_alp, nu_bet, sequence_length=None,
                   mean_batch=True, time_major=False, name=None):
    with ops.name_scope(name, "log_likelihood",
                        [y, f, g, nu_alp, nu_bet, mean_batch, time_major]):
        if sequence_length is None:
            sequence_length = tf.to_int32(tf.reduce_sum(
                tf.ones((tf.shape(f)[:2])), axis=1))
        mask = tf.expand_dims(tf.sequence_mask(sequence_length,
                                               dtype=tf.float32), dim=2)
        if not time_major:
            # [batch_size, seq_len, ...] -> [seq_len, batch_size, ...]
            mask = tf.transpose(mask, perm=[1, 0, 2])
            y = tf.transpose(y, perm=[1, 0, 2])
            f = tf.transpose(f, perm=[1, 0, 2])
            g = tf.transpose(g, perm=[1, 0, 2, 3])
        y = y * mask
        f_term = tf.reduce_sum(f * y, axis=(0, 2))
        f_seq_len = tf.shape(f)[0]
        y_i = tf.expand_dims(tf.slice(y, [0, 0, 0], [f_seq_len-1, -1, -1]),
                             dim=3)
        y_plus = tf.expand_dims(tf.slice(y, [1, 0, 0], [-1, -1, -1]), dim=2)
        g_term = tf.reduce_sum(g * y_i * y_plus, axis=(0, 2, 3))
        z_term = logZ(nu_alp, nu_bet)
        log_like = f_term + g_term - z_term
        if mean_batch:
            log_like = tf.reduce_mean(log_like)
        return log_like


def log_marginal(nu_alp, nu_bet, index_start=None, num_index=None,
                 time_major=False, name=None):
    with ops.name_scope(name, "log_marginal",
                        [nu_alp, nu_bet, index_start, num_index, time_major]):
        if not time_major:
            # [batch_size, seq_len, ...] -> [seq_len, batch_size, ...]
            nu_alp = tf.transpose(nu_alp, perm=[1, 0, 2])
            nu_bet = tf.transpose(nu_bet, perm=[1, 0, 2])
        z_term = tf.expand_dims(logZ(nu_alp, nu_bet, time_major=True), dim=1)
        if index_start is not None or num_index is not None:
            if index_start is not None and num_index is not None:
                nu_alp = tf.slice(nu_alp, [index_start, 0, 0],
                                  [num_index, -1, -1])
                nu_bet = tf.slice(nu_bet, [index_start, 0, 0],
                                  [num_index, -1, -1])
            else:
                raise ValueError("Both index_start and num_index must both be"
                                 " defined or both be None")
        res = nu_alp + nu_bet - z_term
        if not time_major:
            if len(res.get_shape()) == 3:
                res = tf.transpose(res, [1, 0, 2])
        return res


def forward_step_max(nu, f, g, axis):
    return f + tf.reduce_max(g + tf.expand_dims(nu, axis+1),
                             axis=axis)


def viterbi(f, g, time_major=False, name=None):
    with ops.name_scope(name, "viterbi", [f, g, time_major]):
        if not time_major:
            # [batch_size, seq_len, ...] -> [seq_len, batch_size, ...]
            f = tf.transpose(f, perm=[1, 0, 2])
            g = tf.transpose(g, perm=[1, 0, 2, 3])
        axis = 1
        f_sequence_length = tf.shape(f)[0]
        batch_size = tf.shape(f)[1]
        classes = tf.shape(f)[2]
        g_sequence_length = array_ops.shape(g)[0]
        f_ta = ta_ops.TensorArray(tf.float32, size=f_sequence_length,
                                  name="f_ta")
        g_ta = ta_ops.TensorArray(tf.float32, size=g_sequence_length,
                                  name="g_ta")
        f_ta = f_ta.unstack(f)
        g_ta = g_ta.unstack(g)
        nu_ta = ta_ops.TensorArray(tf.float32, size=f_sequence_length,
                                   name="nu_ta")
        nu_label_ta = ta_ops.TensorArray(tf.int32, size=f_sequence_length,
                                         name="nu_label_ta")
        max_time = f_sequence_length
        def forward_cond(time, nu_state, nu_ta_t, nu_label_ta_t):
            return tf.less(time, max_time)

        def forward_body(time, nu_state, nu_ta_t, nu_label_ta_t):
            def zero_state():
                return nu_state, array_ops.zeros([batch_size, classes],
                                                 dtype=tf.int32)
            def normal():
                f_term = f_ta.read(time)
                g_term = g_ta.read(time-1)
                p1 = forward_step_max(nu_state, f_term, g_term, axis=1)
                p2 = tf.cast(tf.argmax(g_term + 
                        tf.expand_dims(nu_state, dim=axis+1), axis=axis),
                             dtype=tf.int32)
                return p1, p2

            new_nu_state, new_nu_label = tf.cond(
                tf.greater(time, 0),
                normal,
                zero_state, name="new_nu_state")
            nu_ta_t = nu_ta_t.write(time, new_nu_state)
            nu_label_ta_t = nu_label_ta_t.write(time, new_nu_label)

            return (time+1, new_nu_state, nu_ta_t, nu_label_ta_t)


        time_forward = tf.constant(0, name="time_forward")
        loop_vars = [time_forward, f_ta.read(tf.constant(0)), nu_ta,
                     nu_label_ta]

        time, state, nu_ta, nu_label_ta = tf.while_loop(forward_cond,
            forward_body, loop_vars)

        viterbi_seq_ta = ta_ops.TensorArray(tf.float32,
                                            size=f_sequence_length,
                                            name="viterbi_seq_ta")

        def viterbi_cond(time, viterbi_state, viterbi_seq_ta_t):
            return tf.less(time, max_time)

        def viterbi_body(time, viterbi_state, viterbi_seq_ta_t):
            def zero_state():
                return viterbi_state
            def normal():
                p1 = tf.cast(nu_label_ta.read(max_time-time),
                             dtype=tf.float32)
                p2 = tf.one_hot(tf.cast(viterbi_state, dtype=tf.int32),
                                classes)
                res = tf.reduce_sum(p1 * p2, axis=axis)
                return res

            new_viterbi_state = tf.cond(
                tf.greater(time, 0),
                normal,
                zero_state, name="new_viterbi_state")
            viterbi_seq_ta_t = viterbi_seq_ta_t.write(
                max_time-time-tf.constant(1), new_viterbi_state)

            return (time+1, new_viterbi_state, viterbi_seq_ta_t)

        time_viterbi = tf.constant(0, name="time_viterbi")
        loop_vars = [time_viterbi,
                     tf.cast(tf.argmax(nu_ta.read(max_time-tf.constant(1)),
                                       axis=axis), dtype=tf.float32),
                     viterbi_seq_ta]

        time, state, viterbi_seq_ta = tf.while_loop(viterbi_cond,
            viterbi_body, loop_vars)
        viterbi_seq = viterbi_seq_ta.stack()

        if not time_major:
            viterbi_seq = tf.transpose(viterbi_seq, perm=[1, 0])

        return viterbi_seq
