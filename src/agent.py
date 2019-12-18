import numpy as np
import tensorflow as tf

from tensorflow.python.ops.nn import dynamic_rnn
from tensorflow.contrib.rnn import GRUCell, LSTMCell, MultiRNNCell
from tensorflow.contrib.lookup.lookup_ops import HashTable, KeyValueTensorInitializer
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.python.ops import variable_scope
from output_projection import output_projection_layer
from utils import gen_batched_data, compute_acc
from utils import FLAGS, PAD_ID, UNK_ID, GO_ID, EOS_ID, _START_VOCAB

class AgentModel(object):
    def __init__(self,
            num_items,
            num_embed_units,
            num_units,
            num_layers,
            embed=None,
            learning_rate=1e-4,
            action_num=10,
            learning_rate_decay_factor=0.95,
            max_gradient_norm=5.0,
            use_lstm=True):

        self.epoch = tf.Variable(0, trainable=False, name='agn/epoch')
        self.epoch_add_op = self.epoch.assign(self.epoch + 1)

        self.sessions_input = tf.placeholder(tf.int32, shape=(None, None))
        self.rec_lists = tf.placeholder(tf.int32, shape=(None, None, None))
        self.rec_mask = tf.placeholder(tf.float32, shape=(None, None, None))
        self.aims_idx = tf.placeholder(tf.int32, shape=(None, None))
        self.sessions_length = tf.placeholder(tf.int32, shape=(None))
        self.reward = tf.placeholder(tf.float32, shape=(None))

        if embed is None:
            self.embed = tf.get_variable('agn/embed', [num_items, num_embed_units], tf.float32, initializer=tf.truncated_normal_initializer(0,1))
        else:
            self.embed = tf.get_variable('agn/embed', dtype=tf.float32, initializer=embed)

        batch_size, encoder_length, rec_length = tf.shape(self.sessions_input)[0], tf.shape(self.sessions_input)[1], tf.shape(self.rec_lists)[2]

        encoder_mask = tf.reshape(tf.cumsum(tf.one_hot(self.sessions_length - 2, 
            encoder_length), reverse=True, axis=1), [-1, encoder_length])
        # [batch_size, length]
        self.sessions_target = tf.concat([self.sessions_input[:, 1:], tf.ones([batch_size, 1], dtype=tf.int32)*PAD_ID], 1)
        # [batch_size, length, embed_units]
        self.encoder_input = tf.nn.embedding_lookup(self.embed, self.sessions_input) 
        # [batch_size, length, rec_length]
        self.aims = tf.one_hot(self.aims_idx, rec_length)

        if use_lstm:
            cell = MultiRNNCell([LSTMCell(num_units) for _ in range(num_layers)])
        else:
            cell = MultiRNNCell([GRUCell(num_units) for _ in range(num_layers)])

        # Training
        with tf.variable_scope("agn"):
            output_fn, sampled_sequence_loss = output_projection_layer(num_units, num_items)
            self.encoder_output, self.encoder_state = dynamic_rnn(cell, self.encoder_input, 
                    self.sessions_length, dtype=tf.float32, scope="encoder")

            tmp_dim_1 = tf.tile(tf.reshape(tf.range(batch_size), [batch_size, 1, 1, 1]), [1, encoder_length, rec_length, 1])
            tmp_dim_2 = tf.tile(tf.reshape(tf.range(encoder_length), [1, encoder_length, 1, 1]), [batch_size, 1, rec_length, 1])
            # [batch_size, length, rec_length, 3]
            gather_idx = tf.concat([tmp_dim_1, tmp_dim_2, tf.expand_dims(self.rec_lists, 3)], 3)

            # [batch_size, length, num_items], [batch_size*length]
            y_prob, local_loss, total_size = sampled_sequence_loss(self.encoder_output, self.sessions_target, encoder_mask)

            # Compute recommendation rank given rec_list
            # [batch_size, length, num_items]
            y_prob = tf.reshape(y_prob, [batch_size, encoder_length, num_items]) * \
                tf.concat([tf.zeros([batch_size, encoder_length, 2], dtype=tf.float32), 
                            tf.ones([batch_size, encoder_length, num_items-2], dtype=tf.float32)], 2)
            # [batch_size, length, rec_len]
            ini_prob = tf.reshape(tf.gather_nd(y_prob, gather_idx), [batch_size, encoder_length, rec_length])
            # [batch_size, length, rec_len]
            mul_prob = ini_prob * self.rec_mask

            # [batch_size, length, action_num]
            _, self.index = tf.nn.top_k(mul_prob, k=action_num)
            # [batch_size, length, metric_num]
            _, self.metric_index = tf.nn.top_k(mul_prob, k=(FLAGS.metric+1))

            self.loss = tf.reduce_sum(tf.reshape(self.reward, [-1]) * local_loss) / total_size

        # Inference
        with tf.variable_scope("agn", reuse=True):
            # tf.get_variable_scope().reuse_variables()
            self.lstm_state = tf.placeholder(tf.float32, shape=(2, 2, None, num_units))
            self.ini_state = (tf.contrib.rnn.LSTMStateTuple(self.lstm_state[0,0,:,:], self.lstm_state[0,1,:,:]), tf.contrib.rnn.LSTMStateTuple(self.lstm_state[1,0,:,:], self.lstm_state[1,1,:,:]))
            # [batch_size, length, num_units]
            self.encoder_output_predict, self.encoder_state_predict = dynamic_rnn(cell, self.encoder_input, 
                    self.sessions_length, initial_state=self.ini_state, dtype=tf.float32, scope="encoder")

            # [batch_size, num_units]
            self.final_output_predict = tf.reshape(self.encoder_output_predict[:,-1,:], [-1, num_units])
            # [batch_size, num_items]
            self.rec_logits = output_fn(self.final_output_predict)
            # [batch_size, action_num]
            _, self.rec_index = tf.nn.top_k(self.rec_logits[:,len(_START_VOCAB):], action_num)
            self.rec_index += len(_START_VOCAB)

            def gumbel_max(inp, alpha, beta):
                # assert len(tf.shape(inp)) == 2
                g = tf.random_uniform(tf.shape(inp),0.0001,0.9999)
                g = -tf.log(-tf.log(g))
                inp_g = tf.nn.softmax((tf.nn.log_softmax(inp/1.0) + g * alpha) * beta)
                return inp_g
            # [batch_size, action_num]
            _, self.random_rec_index = tf.nn.top_k(gumbel_max(self.rec_logits[:,len(_START_VOCAB):], 1, 1), action_num)
            self.random_rec_index += len(_START_VOCAB)

        # initialize the training process
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, 
                dtype=tf.float32)
        self.learning_rate_decay_op = self.learning_rate.assign(
                self.learning_rate * learning_rate_decay_factor)


        self.global_step = tf.Variable(0, trainable=False)
        self.params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, self.params)
        clipped_gradients, self.gradient_norm = tf.clip_by_global_norm(gradients, 
                max_gradient_norm)
        self.update = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(clipped_gradients, self.params), 
                global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), write_version=tf.train.SaverDef.V2, 
                max_to_keep=100, pad_step_number=True, keep_checkpoint_every_n_hours=1.0)

    def print_parameters(self):
        for item in self.params:
            print('%s: %s' % (item.name, item.get_shape()))

    def step_decoder(self, session, data, forward_only=False):
        input_feed = {self.sessions_input: data['sessions'],
                self.reward: data['cum_env_dis_reward'],
                self.aims_idx: data['aims'],
                self.rec_lists: data['rec_lists'],
                self.rec_mask: data['rec_mask'],
                self.sessions_length: data['sessions_length']}

        if forward_only:
            output_feed = [self.loss, self.metric_index]
        else:
            output_feed = [self.loss, self.metric_index, self.gradient_norm, self.update]
        return session.run(output_feed, input_feed)

    def train(self, sess, dataset, generate_session=None, is_train=True, ftest_name=FLAGS.agn_output_file):
        st, ed, loss, acc, acc_1 = 0, 0, [], [], []
        if generate_session:
            dataset = dataset + generate_session
        print("Get %s data:len(dataset) is %d " % ("training" if is_train else "testing", len(dataset)))
        if not is_train:
            fout = open(ftest_name, "w")
            fout.close()
        while ed < len(dataset):
            st, ed = ed, ed + FLAGS.batch_size if ed + \
                FLAGS.batch_size < len(dataset) else len(dataset)
            batch_data = gen_batched_data(dataset[st:ed])
            outputs = self.step_decoder(sess, batch_data, forward_only=False if is_train else True)
            loss.append(outputs[0])
            predict_id = outputs[1]  # [batch_size, length, 10]

            tmp_acc, tmp_acc_1 = compute_acc(
                batch_data["aims"], predict_id, batch_data["rec_lists"], batch_data["rec_mask"], batch_data["purchase"], ftest_name=ftest_name, output=(not is_train))
            acc.append(tmp_acc)
            acc_1.append(tmp_acc_1)                
        if is_train:
            sess.run(self.epoch_add_op)
        return np.mean(loss), np.mean(acc), np.mean(acc_1)