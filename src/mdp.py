import random
import numpy as np
import sys
from environment import EnvModel
from agent import AgentModel
from utils import gen_batched_data, FLAGS, _START_VOCAB
import os
import tensorflow as tf


def aid2index(aid):
    if type(aid) == int:
        return aid + len(_START_VOCAB)
    elif type(aid) == list:
        return (np.array(aid)+len(_START_VOCAB)).tolist()
    elif type(aid) == np.ndarray:
        return aid+len(_START_VOCAB)
    else:
        print("aid2index(): Input Type Error")
        exit()

def get_prob(num):
    def softmax(l):
        return np.exp(l)/(np.sum(np.exp(l))+1e-20)    
    return softmax(np.random.rand(num))

def gen_mdp(state_num, action_num, output_score_file, output_prob_file):
    with open(output_score_file, "w") as fout:
        state_action_score = [np.random.rand(action_num) for _ in range(state_num)]
        for i in range(state_num):
            print >> fout, " ".join(map(str, state_action_score[i]))
    with open(output_prob_file, "w") as fout:
        state_action_state_prob = [[[] for _ in range(action_num)] for _ in range(state_num)]
        for i in range(state_num):
            for j in range(action_num):
                state_action_state_prob[i][j] = get_prob(state_num)
                print >> fout, " ".join(map(str, state_action_state_prob[i][j]))
            print >> fout, "=" * 5
    return state_action_score, state_action_state_prob

def read_mdp(output_score_file, output_prob_file):
    state_action_score, state_action_state_prob = [], []
    with open(output_score_file, "r") as fin:
        for line in fin:
            state_action_score.append(map(float, line.strip().split()))
    with open(output_prob_file, "r") as fin:
        state_action_state_prob.append([])
        for line in fin:
            if "=" in line:
                state_action_state_prob.append([])
            else:
                state_action_state_prob[-1].append(map(float, line.strip().split()))
    state_action_state_prob = state_action_state_prob[:-1]
    return state_action_score, state_action_state_prob

def get_policy(p):
    def random_policy(click_history, state):
        return list(np.random.permutation(range(action_num))[:recommend_num])
    def max_policy(click_history, state):
        return np.argsort(-np.array(state_action_score[state])).tolist()[:recommend_num]
    def mix_policy(click_history, state):
        if random.random() < 0.5:
            return np.argsort(-np.array(state_action_score[state])).tolist()[:recommend_num]
        else:
            return np.random.choice(np.argsort(-np.array(state_action_score[state])).tolist()[action_num/5:action_num/2], recommend_num, replace=False).tolist()
    return random_policy if p == "random" else (max_policy if p == "max" else mix_policy)

def get_model_policy(mode, sess, graph, model):
    if mode == "agn":
        def model_policy(click_history, state):
            with graph.as_default():
                #[1*len, 10]
                output = sess.run(
                    [model.rec_index, model.random_rec_index],
                    feed_dict={model.sessions_input: aid2index(np.reshape(click_history, [1,-1])),
                            model.sessions_length: np.array([len(click_history)]),
                            model.lstm_state:[[[[0.]*FLAGS.units]]*2]*FLAGS.layers})
            current_action = np.reshape(output[1], [-1])[:recommend_num]
            return map(int, current_action-len(_START_VOCAB))
    elif mode == "env":
        def model_policy(click_history, state):
            with graph.as_default():
                #[1*len, 10]
                output = sess.run(
                    [model.inf_all_argmax_index, model.inf_all_random_index],
                    feed_dict={
                        model.sessions_input: aid2index(np.reshape(click_history, [1, -1])),
                        model.rec_lists:[[range(len(_START_VOCAB), aid2index(action_num))]],
                        model.rec_mask:np.ones_like([[range(len(_START_VOCAB), aid2index(action_num))]]),
                        model.sessions_length:[len(click_history)]})
            current_action = np.reshape(output[1], [-1])[:recommend_num]

            return map(int, current_action)
    else:
        print("get_model_policy(mode, sess, graph, model): Mode Error!")
        exit()
    return model_policy

def interact(file, policy, data_num):
    l = []
    with open(file, "w") as fout:
        s = 0
        while len(l) < data_num:
            if s % 100 == 0:
                print("%d sessions have been generated."%s)
            tmp_l = 0
            state, click_history = 0, [0] # initial state
            print >> fout, str(s+1)+";0;"+str(click_history[-1])+";"+";0"
            while True:
                rec_list = map(int, policy(click_history, state))
                rec_score = [state_action_score[state][r] for r in rec_list]
                record_score[state].append(rec_list)
                candidate = int(np.random.choice(rec_list, 1, p=(np.array(rec_score)/np.sum(rec_score)).tolist())[0])
                candidate_score = state_action_score[state][candidate]
                if random.random() < candidate_score and tmp_l < (1e10 if test_model else 40):
                    click_history.append(candidate)
                    state = int(np.random.choice(range(state_num), 1, p=state_action_state_prob[state][click_history[-1]])[0])
                    print >> fout, str(s+1)+";"+str(state)+";"+str(click_history[-1])+";"+",".join(map(str, rec_list))+";0"
                    tmp_l += 1
                else:
                    if tmp_l > 0 and tmp_l < (1e10 if test_model else 40):
                        l.append(tmp_l)
                    s += 1
                    break
    print("%d sessions which meet the length requiement are generated. Average length is %f"%(len(l), np.mean(l) if l else -1))

def build_model(mode, train_dir):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    graph = tf.Graph()
    sess = tf.Session(config=config, graph=graph)
    with graph.as_default():
        if mode == "env":
            model = EnvModel(
                    aid2index(action_num),
                    FLAGS.embed_units,
                    FLAGS.units,
                    FLAGS.layers)
            if tf.train.get_checkpoint_state(train_dir):
                print("Reading environment model parameters from %s" % train_dir)
                model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
            else:
                print("Lost model parameters")
                sess.run(tf.global_variables_initializer())
        elif mode == "agn":
            model = AgentModel(
                    num_items=aid2index(action_num),
                    num_embed_units=FLAGS.embed_units,
                    num_units=FLAGS.units,
                    num_layers=FLAGS.layers,
                    action_num=FLAGS.action_num)
            if tf.train.get_checkpoint_state(train_dir):
                print("Reading agent model parameters from %s" % train_dir)
                model.saver.restore(sess, tf.train.latest_checkpoint(train_dir))
            else:
                print("Lost model parameters")
                sess.run(tf.global_variables_initializer())
        else:
            print("build_model(mode, train_dir): Mode Error!")
            exit()
        model.print_parameters()
    return sess, graph, model

#**********************************************************************************
#**********************************************************************************
#**********************************************************************************

test_model = False # set True to test a policy and False to generate interaction data with a given policy

action_num, state_num, restore_mdp = 50, 10, False
output_score_file, output_prob_file, output_file = "score.txt", "prob.txt", "output.txt"
data_num, recommend_num = 1000, 10

recommend_policy = "model" # "random" / "max" / "mix" / "model"
if recommend_policy != "model":
    policy = get_policy(recommend_policy)
elif recommend_policy == "model":
    train_dir = "./train" # please specify the model directory of the model (including "agn" for rl-based agent model or "env" for sl-based environment model) if recommend_policy == "model"
    mode = "agn" if "agn" in train_dir else "env"    
    sess, graph, model = build_model(mode, train_dir)
    policy = get_model_policy(mode, sess, graph, model)
else:
    print("Recommendation Policy Error!")
    exit()

if restore_mdp:
    print("Reading mdp parameters......")
    state_action_score, state_action_state_prob = read_mdp(output_score_file, output_prob_file)
else:
    print("Creating mdp parameters......")
    state_action_score, state_action_state_prob = gen_mdp(state_num, action_num, output_score_file, output_prob_file)

assert action_num == np.shape(state_action_score)[1] and action_num == np.shape(state_action_state_prob)[1]
assert state_num == np.shape(state_action_score)[0] and state_num == np.shape(state_action_state_prob)[0] and state_num == np.shape(state_action_state_prob)[2]

record_score = {}
for i in range(state_num):
    record_score[i] = []

if test_model:
    print("Begin testing model...")
else:
    print("Begin generating interaction data...")

interact(output_file, policy, data_num)

if test_model:
    klist = (np.array(range(10)) + 1).tolist()
    for k in klist:
        cov, all = 0., 0.
        for st in range(state_num):
            for rec_list in record_score[st]:
                for ac in np.argsort(-np.array(state_action_score[st])).tolist()[:k]:
                    all += 1.
                    if ac in rec_list:
                        cov += 1.
        print("Coverage@%d:%.4f"%(k, cov / all))