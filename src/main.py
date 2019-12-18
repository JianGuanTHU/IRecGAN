# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import sys
import time
import random
import copy
from agent import AgentModel
from environment import EnvModel
from discriminator import DisModel
from utils import FLAGS, load_data, build_vocab, gen_batched_data, PAD_ID, UNK_ID, GO_ID, EOS_ID, _START_VOCAB
import os
#**********************************************************************************

# Empty the output file
fout = open(FLAGS.agn_output_file, "w")
fout.close()
fout = open(FLAGS.env_output_file, "w")
fout.close()

# Creating training directory if it does not exist
if not os.path.exists(FLAGS.interact_data_dir):
    os.makedirs(FLAGS.interact_data_dir)
if not os.path.exists(FLAGS.agn_train_dir):
    os.makedirs(FLAGS.agn_train_dir)
if not os.path.exists(FLAGS.env_train_dir):
    os.makedirs(FLAGS.env_train_dir)
if not os.path.exists(FLAGS.dis_train_dir):
    os.makedirs(FLAGS.dis_train_dir)

generate_session, gen_session, gen_rec_list, gen_aims_idx, gen_purchase, session_no, next_session = [], [], [], [], [], 0, True
ini_state = [[[[0.]*FLAGS.units]]*2]*FLAGS.layers
gen_state = ini_state

def select_action(click, state):
    # current_action = [aid2index[item] for item in list(np.random.permutation(vocab[len(_START_VOCAB):])[:FLAGS.action_num])]
    with agn_graph.as_default():
        output = agn_sess.run(
            [agn_model.random_rec_index, agn_model.encoder_state_predict], 
            feed_dict={agn_model.sessions_input: np.reshape(click, [1,1]), 
                    agn_model.sessions_length: np.array([1]),
                    agn_model.lstm_state:state})
    return np.concatenate([np.reshape(output[0], [1, 1, FLAGS.action_num]), np.reshape([EOS_ID], [1,1,1])], 2), output[1]

def rollout(state, click, rollout_list, rollout_rec_list, rollout_aim, rollout_purchase, length):
    rollout_list.append(click)

    with agn_graph.as_default():
        output = agn_sess.run([agn_model.encoder_state_predict, agn_model.random_rec_index], feed_dict={
            agn_model.sessions_input:np.reshape(click, [1,1]),
            agn_model.sessions_length:[1],
            agn_model.lstm_state:state})
        next_state = output[0]
        action = np.concatenate([np.reshape(output[1], [1, 1, FLAGS.action_num]), np.reshape([EOS_ID], [1,1,1])], 2)
        rollout_rec_list.append(action[0,0,:])

    with env_graph.as_default():
        #[1, 1, 10]
        rec_list = np.reshape(rollout_rec_list[-1], [1,1,-1])
        output = env_sess.run([env_model.inf_random_index, env_model.inf_purchase_prob], feed_dict={
            env_model.sessions_input:np.reshape(rollout_list, [1, -1]), 
            env_model.rec_lists:rec_list, 
            env_model.rec_mask:np.ones_like(rec_list),
            env_model.sessions_length:[len(rollout_list)]})

        next_click = rec_list[0,0,output[0][0, -1, 0]]
        rollout_purchase.append(1 if output[1][0, 0, 1] > 0.5 else 0)
        rollout_aim.append(output[0][0,-1,0])

    if len(rollout_list) >= length or click == 3:
        return rollout_list, rollout_rec_list, rollout_aim, rollout_purchase
    return rollout(next_state, next_click, list(rollout_list), list(rollout_rec_list), list(rollout_aim), list(rollout_purchase), length)


def generate_next_click(current_click, flog, use_dis=FLAGS.use_dis):
    global gen_session, gen_rec_list, gen_aims_idx, gen_state, gen_purchase, session_no, next_session

    if len(gen_session) >= max_interact_len or current_click == 3:
        gen_session = [np.random.choice(sort_start_click, p=sort_start_click_prob)]
        gen_rec_list, gen_aims_idx, gen_purchase = [], [], []
        gen_state = ini_state
        session_no += 1
        next_session = True
        current_click = gen_session[-1]
        print >> flog, "------------next session:%d------------" % (session_no)
    else:
        gen_session.append(current_click)
        next_session = False
    session_click = np.reshape(np.array(gen_session), [1, len(gen_session)])
    action, state = select_action(session_click[0,-1], gen_state)
    print >> flog, "current_click:", current_click,
    gen_state = state

    with env_graph.as_default():
        #[1, 1, 10]
        output = env_sess.run([env_model.inf_random_index, env_model.inf_purchase_prob], feed_dict={
            env_model.sessions_input:session_click, 
            env_model.rec_lists:action, 
            env_model.rec_mask:np.ones_like(action),
            env_model.sessions_length:[len(session_click[0])]})
        next_click = action[0, 0, output[0][0, -1, 0]]
        purchase_prob = output[1][0, 0, 1]
        print >> flog, "next_click:", next_click, "purchase_prob:", purchase_prob, "reward:", 4 if purchase_prob > 0.5 else 1,
        gen_rec_list.append(list(action[0,0,:]))
        gen_aims_idx.append(output[0][0,-1,0])
        gen_purchase.append(1 if purchase_prob > 0.5 else 0)
    dis_reward = 1.

    if use_dis:
        with dis_graph.as_default():
            score = []
            rollout_num = 5 if (len(gen_session) < max_interact_len) and (next_click != 3) else 1
            for _ in range(rollout_num):
                tmp_total_click, tmp_total_rec_list, tmp_total_aims_idx, tmp_total_purchase = rollout(gen_state,next_click,list(gen_session), list(gen_rec_list), list(gen_aims_idx), list(gen_purchase), max_interact_len+1)
                prob = dis_sess.run(dis_model.prob, {
                    dis_model.sessions_input:np.reshape(tmp_total_click, [1, -1]),
                    dis_model.sessions_length:np.array([len(tmp_total_click)]),
                    dis_model.rec_lists:np.array([tmp_total_rec_list]),
                    dis_model.rec_mask:np.ones([1,len(tmp_total_click),len(tmp_total_rec_list[-1])]),
                    dis_model.aims_idx:np.reshape(tmp_total_aims_idx, [1, len(tmp_total_click)]),
                    dis_model.purchase:np.reshape(tmp_total_purchase, [1, len(tmp_total_purchase)])
                    })
                score.append(prob[0])
            dis_reward = np.mean(score)
        print >> flog, "dis_reward:%.8f" % dis_reward,

    action = list(action[0,0,:])
    print >> flog, "action:", action
    return current_click, next_click, action, purchase_prob, dis_reward

def generate_data(size, flog, use_dis=FLAGS.use_dis):
    global generate_session, current_click, session_no, next_session
    tmp_session_no = session_no
    current_click = np.random.choice(sort_start_click, p=sort_start_click_prob)
    while session_no < tmp_session_no + size:
        current_click, next_click, current_action, purchase_prob, dis_reward = generate_next_click(current_click, flog, use_dis=use_dis)
        if not next_session and len(generate_session) > 0:
            generate_session[-1].append({"session_no":session_no, "click":current_click, "rec_list": current_action, "purchase":(0 if purchase_prob<=0.5 else 1), "dis_reward": dis_reward})
        else:
            if len(generate_session) > 0:
                length = len(generate_session[-1])
                for i in range(1, length):
                    generate_session[-1][length-i]["rec_list"] = generate_session[-1][length-i-1]["rec_list"]
                    generate_session[-1][length-i]["purchase"] = generate_session[-1][length-i-1]["purchase"]
                generate_session[-1][0]["rec_list"] = [generate_session[-1][0]["click"]]
                generate_session[-1][0]["purchase"] = 0

            generate_session.append([{"session_no":session_no, "click":current_click, "rec_list": current_action, "purchase":(0 if purchase_prob<=0.5 else 1), "dis_reward": dis_reward}])
        current_click = next_click
    next_session = True
    if len(generate_session) > FLAGS.pool_size:
        generate_session = generate_session[-FLAGS.pool_size:]

#**********************************************************************************
#**********************************************************************************
#**********************************************************************************

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
env_graph = tf.Graph()
agn_graph = tf.Graph()
dis_graph = tf.Graph()
env_sess = tf.Session(config=config, graph=env_graph)
agn_sess = tf.Session(config=config, graph=agn_graph)
dis_sess = tf.Session(config=config, graph=dis_graph)

data = load_data(FLAGS.data_dir, FLAGS.data_name)
data = np.random.permutation(data)

max_interact_len = 2 * int(np.mean([len(s) for s in data]))
print("Average length of the dataset:", np.mean([len(s) for s in data]), "max_interact_len:", max_interact_len)
fold = len(data) / 40
data_train = data[:(fold * 38)]
data_dev = data[(fold * 38):(fold * 39)]
data_test = data[(fold * 39):]

vocab, embed = build_vocab(data_train)
aid2index = {}
index2aid = {}
for i,a in enumerate(vocab):
    aid2index[a] = i
    index2aid[i] = a

if FLAGS.use_simulated_data:
    sort_start_click = [data[0][0]["click"]]
    sort_start_click_prob = [1.]
else:
    start_click = {}
    for d in data:
        if d[0]["click"] in aid2index:
            k = aid2index[d[0]["click"]]
        else:
            continue
        if k in start_click:
            start_click[k] += 1
        else:
            start_click[k] = 1
    sort_start_click = sorted(start_click, key=start_click.get, reverse=True)
    sort_start_click_prob = np.array([start_click[item] for item in sort_start_click]) / float(np.sum([start_click[item] for item in sort_start_click]))

def filter(d):
    new_d = []
    for i, s in enumerate(d):
        tmps = []
        for c in s:
            c["click"] = aid2index[c["click"]] if c["click"] in aid2index else UNK_ID
            c["rec_list"] = list(set([aid2index[rl] if rl in aid2index else UNK_ID for rl in c["rec_list"]])) + [EOS_ID]
            tmps.append(c)
        new_d.append(tmps)
    d = copy.deepcopy(new_d)
    return d
data_train = filter(data_train)
data_dev = filter(data_dev)
data_test = filter(data_test)

print("Get training data: number is %d, average length is %.4f" % (len(data_train), np.mean([len(s) for s in data_train])))
print("Get validation data: number is %d, average length is %.4f" % (len(data_dev), np.mean([len(s) for s in data_dev])))
print("Get testing data: number is %d, average length is %.4f" % (len(data_test), np.mean([len(s) for s in data_test])))

with agn_graph.as_default():
    agn_model = AgentModel(
            num_items=len(embed),
            num_embed_units=FLAGS.embed_units,
            num_units=FLAGS.units,
            num_layers=FLAGS.layers,
            embed=embed,
            action_num=FLAGS.action_num)
    agn_model.print_parameters()
    if tf.train.get_checkpoint_state(FLAGS.agn_train_dir):
        print("Reading agent model parameters from %s" % FLAGS.agn_train_dir)
        agn_model.saver.restore(agn_sess, tf.train.latest_checkpoint(FLAGS.agn_train_dir))
    else:
        print("Creating agent model with fresh parameters.")
        agn_sess.run(tf.global_variables_initializer())

if FLAGS.interact:
    with env_graph.as_default():
        env_model = EnvModel(
                num_items=len(embed),
                num_embed_units=FLAGS.embed_units,
                num_units=FLAGS.units,
                num_layers=FLAGS.layers,
                vocab=vocab,
                embed=embed)
        env_model.print_parameters()
        if tf.train.get_checkpoint_state(FLAGS.env_train_dir):
            print("Reading environment model parameters from %s" % FLAGS.env_train_dir)
            env_model.saver.restore(env_sess, tf.train.latest_checkpoint(FLAGS.env_train_dir))
        else:
            print("Creating environment model with fresh parameters.")
            env_sess.run(tf.global_variables_initializer())

    if FLAGS.use_dis:
        with dis_graph.as_default():
            dis_model = DisModel(
                    num_items=len(embed),
                    num_embed_units=FLAGS.embed_units,
                    num_units=FLAGS.units,
                    num_layers=FLAGS.layers,
                    vocab=vocab,
                    embed=embed)
            dis_model.print_parameters()
            if tf.train.get_checkpoint_state(FLAGS.dis_train_dir):
                print("Reading discriminator model parameters from %s" % FLAGS.dis_train_dir)
                dis_model.saver.restore(dis_sess, tf.train.latest_checkpoint(FLAGS.dis_train_dir))
            else:
                print("Creating discriminator model with fresh parameters.")
                dis_sess.run(tf.global_variables_initializer())

best_env_train_acc, best_env_train_acc_1 = 0., 0.
def env_train(size, pg=False):
    global best_env_train_acc, best_env_train_acc_1
    pre_losses = [1e18] * 3
    for _ in range(size):
        with env_graph.as_default():
            start_time = time.time()
            if pg:
                loss = env_model.pg_train(env_sess, generate_session)
                pr_loss, pu_loss = 0, 0
            else:
                loss, pr_loss, pu_loss, _, _ = env_model.train(env_sess, data_train)
            if loss > max(pre_losses):  # Learning rate decay
                env_sess.run(env_model.learning_rate_decay_op)
            pre_losses = pre_losses[1:] + [loss]
            print("Env epoch %d lr %.4f time %.4f ppl [%.8f] pr_loss [%.8f] pu_loss [%.8f]" \
                  % (env_model.epoch.eval(session=env_sess), env_model.learning_rate.eval(session=env_sess), time.time() - start_time, loss, pr_loss, pu_loss))
            loss, pr_loss, pu_loss, acc, acc_1 = env_model.train(env_sess, data_dev, is_train=False)
            print("        dev_set, ppl [%.8f] pr_loss [%.8f] pu_loss [%.8f] best_p@%d [%.4f]" % (loss, pr_loss, pu_loss, FLAGS.metric, best_env_train_acc))
            if acc > best_env_train_acc or acc_1 > best_env_train_acc_1:
                if acc > best_env_train_acc: best_env_train_acc = acc
                if acc_1 > best_env_train_acc_1: best_env_train_acc_1 = acc_1
                loss, pr_loss, pu_loss, _, _ = env_model.train(env_sess, data_test, is_train=False)
                print("        test_set, ppl [%.8f] pr_loss [%.8f] pu_loss [%.8f] best_p@%d [%.4f]" % (loss, pr_loss, pu_loss, FLAGS.metric, best_env_train_acc))
                env_model.saver.save(env_sess, '%s/checkpoint' % FLAGS.env_train_dir, global_step=env_model.global_step)
                print("Saving env model params in %s" % FLAGS.env_train_dir)
            print("------env %strain finish-------"%("pg " if pg else ""))

best_agn_train_acc, best_agn_train_acc_1 = 0., 0.
def agn_train(size):
    global best_agn_train_acc, best_agn_train_acc_1
    for _ in range(size):
        with agn_graph.as_default():
            start_time = time.time()
            loss, acc, acc_1 = agn_model.train(agn_sess, data_train, generate_session)
            print("Agn epoch %d learning rate %.4f epoch-time %.4f loss [%.8f] p@%d %.4f%% p@1 %.4f%%" \
                    % (agn_model.epoch.eval(session=agn_sess), agn_model.learning_rate.eval(session=agn_sess), time.time() - start_time, loss, FLAGS.metric, acc*100, acc_1*100))
            loss, acc, acc_1 = agn_model.train(agn_sess, data_dev, is_train=False)
            print("        dev_set, loss [%.8f] p@%d %.4f%% p@1 %.4f%%" % (loss, FLAGS.metric, acc*100, acc_1*100))
            if acc > best_agn_train_acc or acc_1 > best_agn_train_acc_1:
                if acc > best_agn_train_acc: best_agn_train_acc = acc
                if acc_1 > best_agn_train_acc_1: best_agn_train_acc_1 = acc_1
                loss, acc, acc_1 = agn_model.train(agn_sess, data_test, is_train=False)
                print("        test_set, loss [%.8f] p@%d %.4f%% p@1 %.4f%%" % (loss, FLAGS.metric, acc*100, acc_1*100))
                agn_model.saver.save(agn_sess, '%s/checkpoint' % FLAGS.agn_train_dir, global_step=agn_model.global_step)
                print("Saving agn model params in %s" % FLAGS.agn_train_dir)
            print("------agn train finish-------")

def dis_train(size):
    random_generate_session = np.random.permutation(generate_session).tolist()
    for _ in range(size):
        with dis_graph.as_default():
            start_time = time.time()
            loss, acc = dis_model.train(data_train, random_generate_session, sess=dis_sess)
            dis_model.saver.save(dis_sess, '%s/checkpoint' % FLAGS.dis_train_dir, global_step=dis_model.global_step)
            print("Dis epoch %d learning rate %.4f epoch-time %.4f perplexity [%.8f] acc %.4f%%" \
                    % (dis_model.epoch.eval(session=dis_sess), dis_model.learning_rate.eval(session=dis_sess), time.time() - start_time, loss, acc*100))
            print("------dis train finish-------")

def interact(size, use_dis=FLAGS.use_dis):
    start_time = time.time()
    with open("%s/train_log_%d.txt"%(FLAGS.interact_data_dir, session_no), "w") as flog:
        generate_data(size, flog, use_dis=use_dis)
    print("%d interactions finished after %.4fs " % (size, time.time()-start_time))

# Pretraining
agn_train(50)
if FLAGS.interact:
    env_train(50)
    interact(1000, use_dis=False)
    if FLAGS.use_dis:
        dis_train(3)

# Adversarial training
generate_session = []
pre_losses = [1e18] * 3
while True:
    for _ in range(3):
        for _ in range(3):
            if FLAGS.interact:
                interact(200)
            agn_train(1)

        if FLAGS.use_dis:
            env_train(1, pg=True)
            env_train(1)
    if FLAGS.use_dis:
        dis_train(3)
    print("*"*25)