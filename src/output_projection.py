import tensorflow as tf
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope
def output_projection_layer(num_units, num_items, name="decoder/output_projection"):
    def output_fn(outputs):
        return layers.linear(outputs, num_items, scope=name)

    def sampled_sequence_loss(outputs, targets, masks, scope=None):
        with variable_scope.variable_scope(name if scope is None else scope):
            weights = tf.transpose(tf.get_variable("weights", [num_units, num_items]))
            bias = tf.get_variable("biases", [num_items])

            local_prob = tf.nn.softmax(tf.einsum('aij,kj->aik', outputs, weights) + bias)
            local_labels = tf.reshape(targets, [-1])
            local_masks = tf.reshape(masks, [-1])

            #[batch_size*length, num_items]
            y_log_prob = tf.reshape(tf.log(local_prob+1e-18), [-1, num_items])
            #[batch_size*length, num_items]
            labels_onehot = tf.clip_by_value(tf.one_hot(local_labels, num_items), 0.0, 1.0)
            #[batch_size*length]
            local_loss = tf.reduce_sum(-labels_onehot * y_log_prob, 1) * local_masks
            
            total_size = tf.reduce_sum(local_masks)
            total_size += 1e-12 # to avoid division by 0 for all-0 weights

            return  local_prob, local_loss, total_size

    return output_fn, sampled_sequence_loss
