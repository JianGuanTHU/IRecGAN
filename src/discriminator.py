import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope
from utils import FLAGS, gen_batched_data

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

class DisModel(object):
    def __init__(self,
            num_items,
            num_embed_units,
            num_units,
            num_layers,
            vocab=None,
            embed=None,
            learning_rate=1e-4,
            learning_rate_decay_factor=0.95,
            beam_size=5,
            max_gradient_norm=5.0,
            num_samples=512,
            max_length=30,
            use_lstm=True):

        self.epoch = tf.Variable(0, trainable=False, name='dis/epoch')
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)

        self.sessions_input = tf.placeholder(tf.int32, shape=(None, None))
        self.sessions_length = tf.placeholder(tf.int32, shape=(None))
        self.rec_lists = tf.placeholder(tf.int32, shape=(None, None, None))
        self.rec_mask = tf.placeholder(tf.float32, shape=(None, None, None))
        self.aims_idx = tf.placeholder(tf.int32, shape=(None, None))
        self.label = tf.placeholder(tf.int32, shape=(None))
        self.purchase = tf.placeholder(tf.int32, shape=(None, None))

        if embed is None:
            self.embed = tf.get_variable('dis/embed', [num_items, num_embed_units], tf.float32)
        else:
            self.embed = tf.get_variable('dis/embed', dtype=tf.float32, initializer=embed)

        encoder_length, rec_length = tf.shape(self.sessions_input)[1], tf.shape(self.rec_lists)[2]

        encoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.sessions_length - 2, 
            encoder_length), reverse=True, axis=1), [-1, encoder_length])

        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.sessions_input) #batch*len*unit

        if use_lstm:
            cell = MultiRNNCell([LSTMCell(num_units) for _ in range(num_layers)])
        else:
            cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])

        # rnn encoder
        encoder_output, _ = dynamic_rnn(cell, self.encoder_input, self.sessions_length, dtype=tf.float32, scope="dis/encoder")

        #[batch_size, length, embed_units]
        self.preference = tf.layers.dense(encoder_output, num_units, name="dis/out2preference")
        #[batch_size, length, rec_len, num_units]
        self.candidate = tf.layers.dense(tf.nn.embedding_lookup(self.embed, self.rec_lists), num_units, name="dis/rec2candidate")
        #[batch_size, length, rec_len]
        self.pre_mul_can = tf.reduce_sum(tf.expand_dims(self.preference, 2) * self.candidate, 3)

        self.max_embed = tf.reduce_sum(tf.expand_dims(tf.nn.softmax(self.pre_mul_can / 0.1), 3) * self.candidate, 2)
        self.aim_embed = tf.reduce_sum(tf.expand_dims(tf.one_hot(self.aims_idx, rec_length), 3) * self.candidate, 2)
        if FLAGS.use_simulated_data:
            purchase_weight = tf.constant(1.0, dtype=tf.float32)
        else:
            W_p = tf.get_variable("Wp", shape=(), dtype=tf.float32)
            b_p = tf.get_variable("bp", shape=(), dtype=tf.float32)
            purchase_weight = tf.cast(self.purchase, tf.float32) * W_p + b_p
        self.logits = tf.reduce_sum(tf.reduce_sum(self.max_embed * self.aim_embed, 2) * purchase_weight * encoder_mask, 1) / tf.reduce_sum(encoder_mask, 1)
        self.prob = tf.nn.sigmoid(self.logits)
        self.decoder_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=tf.cast(self.label, tf.float32)))
        self.acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.greater(self.prob, 0.5), tf.int32), self.label), tf.float32))

        self.params = tf.trainable_variables()

        self.learning_rate = tf.Variable(float(learning_rate), trainable=False,
                dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)

        self.global_step = tf.Variable(0, trainable=False)

        gradients = tf.gradients(self.decoder_loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                max_gradient_norm)
        self.update = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, self.params), 
                global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, 
                max_to_keep=10, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def step_decoder(self, session, data):
        input_feed = {
            self.sessions_input : data["sessions"],
            self.sessions_length : data["sessions_length"],
            self.label : data["labels"],
            self.rec_lists: data['rec_lists'],
            self.rec_mask: data['rec_mask'],  
            self.aims_idx: data['aims'],
            self.purchase: data['purchase']
        }
        output_feed = [self.decoder_loss, self.acc, self.prob, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)


    def train(self, data, data_gen, sess, dis_batch_size=32):
        st, ed, loss, acc = 0, 0, [], []
        while ed < len(data):
            st, ed = ed, ed+dis_batch_size if ed+dis_batch_size < len(data) else len(data)
            st_gen, ed_gen = st % len(data_gen), ed % len(data_gen)
            tmp_data_gen = data_gen[st_gen:ed_gen] if st_gen < ed_gen else data_gen[st_gen:] + data_gen[:ed_gen]

            concat_data = list(data[st:ed]) + tmp_data_gen
            batch_data = gen_batched_data(concat_data)
            batch_data["labels"] = np.array(np.array([1]*(ed-st)).tolist() + np.array([0]*len(tmp_data_gen)).tolist())
            outputs = self.step_decoder(sess, batch_data)
            loss.append(outputs[0])
            acc.append(outputs[1])
        sess.run(self.epoch_add_op)
        return np.mean(loss), np.mean(acc)
