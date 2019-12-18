import tensorflow as tf
import numpy as np
import os
import copy
import sys
tf.app.flags.DEFINE_integer("units", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("symbols", 40000, "Size of item list.")
tf.app.flags.DEFINE_integer("embed_units", 50, "Embedding units.")
tf.app.flags.DEFINE_integer("layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("action_num", 10, "num of recommendations")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")

tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("data_name", "train.csv", "Data name")
tf.app.flags.DEFINE_boolean("use_simulated_data", False, "Set to True to use simulated data")
tf.app.flags.DEFINE_string("interact_data_dir", "./interact_data", "Directory to store interaction data.")
tf.app.flags.DEFINE_string("agn_train_dir", "./train/agn_train", "Training directory for agent model.")
tf.app.flags.DEFINE_string("env_train_dir", "./train/env_train", "Training directory for environment model.")
tf.app.flags.DEFINE_string("dis_train_dir", "./train/dis_train", "Training directory for discriminator model.")
tf.app.flags.DEFINE_boolean("interact", True, "Set to True to use online model-based training")
tf.app.flags.DEFINE_boolean("use_dis", True, "Set to True to use adversarial training")
tf.app.flags.DEFINE_integer("metric", 10, "For the calculation of p@metric")
tf.app.flags.DEFINE_string("agn_output_file", "./agn_test_output.txt", "Output file for agent model on the test set.")
tf.app.flags.DEFINE_string("env_output_file", "./env_test_output.txt", "Output file for environment model on the test set.")
tf.app.flags.DEFINE_integer("pool_size", 6000, "Maximum number of simulated data.")
tf.app.flags.DEFINE_float("gamma", 0.9, "discount factor")

FLAGS = tf.app.flags.FLAGS

PAD_ID = 0
UNK_ID = 1
GO_ID = 2
EOS_ID = 3
_START_VOCAB = ['_PAD', '_UNK', '_GO', '_EOS']

def load_data(path, fname):
    session = {}
    output_session = []
    print("Reading data from: %s/%s" % (path, fname))
    with open("%s/%s"%(path,fname), "r") as fin:
        for line in fin:
            tmp = line.strip().split(";")
            if int(tmp[0]) in session:
                session[int(tmp[0])].append(
                    {"click": tmp[2],
                    "rec_list": tmp[3].strip().split(","),
                    "purchase": int(tmp[4].strip()),
                    "dis_reward":1.})
            else:
                session[int(tmp[0])] = [
                    {"click": tmp[2],
                    "rec_list": tmp[3].strip().split(","),
                    "purchase": int(tmp[4].strip()),
                    "dis_reward":1.}]
            if session[int(tmp[0])][-1]["click"] not in session[int(tmp[0])][-1]["rec_list"]:
                session[int(tmp[0])][-1]["rec_list"] += [session[int(tmp[0])][-1]["click"]]

        skey = sorted(session.keys())
        for key in skey:
            if len(session[key]) > 1 and len(session[key]) <= 40:
                output_session.append(session[key])
        print("Number of sessions after filtering:", len(output_session))
    return output_session

def build_vocab(data):
    print("Creating vocabulary...")
    if FLAGS.use_simulated_data:
        article_list = _START_VOCAB + map(str, range(50))
    else:
        vocab = {}
        for each_session in data:
            for item in each_session:
                v = [item["click"]]
                for token in v:
                    if token in vocab:
                        vocab[token] += 1
                    else:
                        vocab[token] = 1
        article_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)

    with open("article_list.txt", "w") as fout:
        for a in article_list:
            print >> fout, a
    if len(article_list) > FLAGS.symbols:
        article_list = article_list[:FLAGS.symbols]
    else:
        FLAGS.symbols = len(article_list)
    embed = []
    for i, _ in enumerate(article_list):
        if i < len(_START_VOCAB):
            embed.append(np.zeros(FLAGS.embed_units, dtype=np.float32))
        else:
            embed.append(np.random.normal(size=(FLAGS.embed_units)))
    embed = np.array(embed, dtype=np.float32)
    return article_list, embed


def gen_batched_data(data):
    random_appendix = 100
    max_len = max([len(item) for item in data]) + 1
    max_rec_len = max([len(s["rec_list"]) for item in data for s in item] + [random_appendix+20])
    sessions, aims, rec_lists, sessions_length, purchase, rec_mask, dis_reward, env_dis_reward, cum_env_dis_reward, cum_env_reward = [], [], [], [], [], [], [], [], [], []
    def padding(sent, l):
        return sent + [EOS_ID] + [PAD_ID for _ in range(l-len(sent)-1)]

    def padding_m1(vec, l):
        return vec + [-1. for _ in range(l - len(vec))]

    def padding_cum_reward(vec, l):
        for i in range(len(vec)-1):
            inv_i = len(vec) - i - 2
            vec[inv_i] += FLAGS.gamma * vec[inv_i+1]
        return vec + [-1. for _ in range(l - len(vec))]

    def get_vec(session_al):
        session_tmp = []
        mask_tmp = []
        for al in session_al:
            session_tmp.append(al + [0 for _ in range(max_rec_len - len(al))])
            mask_tmp.append([1. for _ in range(len(al))] + [0. for _ in range(max_rec_len - len(al))])
        session_tmp += [[0 for _ in range(max_rec_len)] for k in range(max_len-len(session_tmp))]
        mask_tmp += [[0. for _ in range(max_rec_len)] for  k in range(max_len-len(mask_tmp))]
        return session_tmp, mask_tmp

    def get_aim(session_aim, rec_list):
        s_aim = []
        for a, rlist in zip(session_aim, rec_list):
            try:
                s_aim.append(rlist.index(a))
            except ValueError:
                s_aim.append(len(rlist))
        return s_aim

    for item in data:
        sessions.append(padding([s["click"] for s in item], max_len))    
        purchase.append([s["purchase"] for s in item][1:] + [0 for _ in range(max_len-len(item)+1)])

        env_reward_list = [s["purchase"] * 3 + 1 for s in item][1:] + [0.]
        cum_env_reward.append(padding_cum_reward(env_reward_list, max_len))

        dis_reward_list = [s["dis_reward"] for s in item]
        env_dis_reward.append(padding_m1((np.array(env_reward_list)*np.array(dis_reward_list)).tolist(), max_len))
        cum_env_dis_reward.append(padding_cum_reward((np.array(env_reward_list)*np.array(dis_reward_list)).tolist(), max_len))
        dis_reward.append(padding_m1(dis_reward_list, max_len))

        rl, rm = get_vec([s["rec_list"] for s in item][1:] + [[EOS_ID]+np.random.permutation(range(4,FLAGS.symbols))[:np.random.randint(15,random_appendix)].tolist()])
        rec_lists.append(rl)
        rec_mask.append(rm)
        aims.append(get_aim(sessions[-1][1:] + [0], rl))
        sessions_length.append(len(item)+1)

    batched_data = {'sessions': np.array(sessions), # the click sequence [batch_size, length]
            'rec_lists': np.array(rec_lists),   # the recommendation sequence [batch_size, length, rec_length]
            'rec_mask' : np.array(rec_mask),    # the mask of recommendation sequence [batch_size, length, rec_length]
            'aims': np.array(aims), # aim index [batch_size, length]
            'purchase': np.array(purchase), # 0 or 1 or -1 to indicate whether purchase the next click [batch_size, length]
            'cum_env_reward': np.array(cum_env_reward),    # cum purchase reward from env
            'dis_reward': np.array(dis_reward), # reward \in [0,1] from dis
            'env_dis_reward': np.array(env_dis_reward),    # purchase reward \in [0, 4] from env reweighted by dis
            'cum_env_dis_reward': np.array(cum_env_dis_reward), # cum reward from env reweighted by dis
            'sessions_length': sessions_length} # session_length
    return batched_data


def compute_acc(ba, pi, rl, mask, purchase, ftest_name, output):
    ftest = open(ftest_name, "a+")
    total_num, total_num_1, correct, correct_1 = 0., 0., 0., 0.
    pur, pur_1, all_purchase, all_purchase_1 = 0., 0., 0., 0.
    for batch_aim, batch_predict_idx, batch_rec_list, batch_rec_mask, batch_purchase in zip(ba, pi, rl, mask, purchase):
        for i, (aim, predict_index, rec_list, rec_mask, purchase) in enumerate(zip(batch_aim, batch_predict_idx, batch_rec_list, batch_rec_mask, batch_purchase)):
            if rec_list[0] == EOS_ID:
                break
            if np.sum(rec_mask) > FLAGS.metric:
                if output: print >> ftest, ">%d"%FLAGS.metric,
                total_num += 1

                new_predict_index = []
                for tmpp in predict_index:
                    if rec_list[tmpp] != EOS_ID:
                        new_predict_index.append(tmpp)
                if len(new_predict_index) > FLAGS.metric:
                    predict_index = new_predict_index[:FLAGS.metric]
                else:
                    predict_index = new_predict_index[:]
                for tmpp in predict_index:
                    if rec_list[tmpp] != 1 and rec_mask[tmpp] == 1 and (aim == tmpp):
                        if output: print >> ftest, "p@%d"%FLAGS.metric,
                        correct += 1
                        break

                if batch_purchase[i+1] == 1:
                    all_purchase += 1
                    for tmpp in predict_index:
                        if rec_list[tmpp] != 1 and rec_mask[tmpp] == 1 and (aim == tmpp):
                            pur += 1
                            break

                total_num_1 += 1
                if rec_list[predict_index[0]] != 1 and rec_mask[predict_index[0]] == 1 and (aim == predict_index[0]):
                    if output: print >> ftest, 1,
                    correct_1 += 1
                else:
                    if output: print >> ftest, 0,
                if output: print >> ftest, batch_purchase[i+1] * 3 + 1,
                try:
                    if output: print >> ftest, [rec_list[tmpp] for tmpp in predict_index], rec_list[aim], list(rec_list)
                except:
                    print(predict_index, aim, rec_list)
                    print(ba, pi, rl, mask, purchase)
                    exit()
                if batch_purchase[i+1] == 1:
                    all_purchase_1 += 1
                    if rec_list[predict_index[0]] != 1 and rec_mask[predict_index[0]] == 1 and (aim == predict_index[0]):
                        pur_1 += 1

        if output: print >> ftest, "-------------------------"
    return correct / (total_num + 1e-18), correct_1 / (total_num_1 + 1e-18) #, pur, all_purchase, pur_1, all_purchase_1
