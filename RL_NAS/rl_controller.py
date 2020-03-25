# %%

import logging
import csv
import numpy as np
import tensorflow as tf
import sys
from rl_input import controller_params, HW_constraints
import termplotlib as tpl
import copy
import random
from datetime import datetime
import time
import torch
import os

logger = logging.getLogger(__name__)


def ema(values):
    """
    Helper function for keeping track of an exponential moving average of a list of values.
    For this module, we use it to maintain an exponential moving average of rewards
    """
    weights = np.exp(np.linspace(-1., 0., len(values)))
    weights /= weights.sum()
    a = np.convolve(values, weights, mode="full")[:len(values)]
    return a[-1]


class Controller(object):

    def __init__(self):
        self.graph = tf.Graph()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config, graph=self.graph)

        self.hidden_units = controller_params['hidden_units']

        self.nn1_search_space = controller_params['sw_space']
        self.hw1_search_space = controller_params['hw_space']

        self.nn1_num_para = len(self.nn1_search_space)
        self.hw1_num_para = len(self.hw1_search_space)


        self.num_para = self.nn1_num_para + self.hw1_num_para

        self.nn1_beg, self.nn1_end = 0, self.nn1_num_para
        self.hw1_beg, self.hw1_end = self.nn1_end, self.nn1_end + self.hw1_num_para

        self.para_2_val = {}
        idx = 0
        for hp in self.nn1_search_space:
            self.para_2_val[idx] = hp
            idx += 1
        for hp in self.hw1_search_space:
            self.para_2_val[idx] = hp
            idx += 1


        self.RNN_classifier = {}
        self.RNN_pred_prob = {}
        with self.graph.as_default():
            self.build_controller()

        self.reward_history = []
        self.architecture_history = []
        self.trained_network = {}

        self.explored_info = {}

        self.target_HW_Eff = HW_constraints["target_HW_Eff"]

    def build_controller(self):
        logger.info('Building RNN Network')
        # Build inputs and placeholders
        with tf.name_scope('controller_inputs'):
            # Input to the NASCell
            self.child_network_paras = tf.placeholder(tf.int64, [None, self.num_para], name='controller_input')
            # Discounted rewards
            self.discounted_rewards = tf.placeholder(tf.float32, (None,), name='discounted_rewards')
            # WW 12-18: input: the batch_size variable will be used to determine the RNN batch
            self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')

        with tf.name_scope('embedding'):
            self.embedding_weights = []
            # share embedding weights for each type of parameters
            embedding_id = 0
            para_2_emb_id = {}
            for i in range(len(self.para_2_val.keys())):
                additional_para_size = len(self.para_2_val[i])
                additional_para_weights = tf.get_variable('state_embeddings_%d' % (embedding_id),
                                                          shape=[additional_para_size, self.hidden_units],
                                                          initializer=tf.initializers.random_uniform(-1., 1.))
                self.embedding_weights.append(additional_para_weights)
                para_2_emb_id[i] = embedding_id
                embedding_id += 1

            self.embedded_input_list = []
            for i in range(self.num_para):
                self.embedded_input_list.append(
                    tf.nn.embedding_lookup(self.embedding_weights[para_2_emb_id[i]], self.child_network_paras[:, i]))
            self.embedded_input = tf.stack(self.embedded_input_list, axis=-1)
            self.embedded_input = tf.transpose(self.embedded_input, perm=[0, 2, 1])

        logger.info('Building Controller')
        with tf.name_scope('controller'):
            with tf.variable_scope('RNN'):
                nas = tf.contrib.rnn.NASCell(self.hidden_units)
                tmp_state = nas.zero_state(batch_size=self.batch_size, dtype=tf.float32)
                init_state = tf.nn.rnn_cell.LSTMStateTuple(tmp_state[0], tmp_state[1])

                output, final_state = tf.nn.dynamic_rnn(nas, self.embedded_input, initial_state=init_state,
                                                        dtype=tf.float32)
                tmp_list = []
                # print("output","="*50,output)
                # print("output slice","="*50,output[:,-1,:])
                for para_idx in range(self.num_para):
                    o = output[:, para_idx, :]
                    para_len = len(self.para_2_val[para_idx])
                    # len(self.para_val[para_idx % self.para_per_layer])
                    classifier = tf.layers.dense(o, units=para_len, name='classifier_%d' % (para_idx), reuse=False)
                    self.RNN_classifier[para_idx] = classifier
                    prob_pred = tf.nn.softmax(classifier)
                    self.RNN_pred_prob[para_idx] = prob_pred
                    child_para = tf.argmax(prob_pred, axis=-1)
                    tmp_list.append(child_para)
                self.pred_val = tf.stack(tmp_list, axis=1)

        logger.info('Building Optimization')
        # with tf.name_scope('Optimization'):
        # Global Optimization composes all RNNs in one, like NAS, where arch_idx = 0

        with tf.name_scope('Optimizer'):
            self.global_step = tf.Variable(0, trainable=False)
            self.learning_rate = tf.train.exponential_decay(0.99, self.global_step, 50, 0.5, staircase=True)
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=0.1)
        with tf.name_scope('Loss'):
            # We seperately compute loss of each predict parameter since the dim of predicting parameters may not be same
            for para_idx in range(self.num_para):
                if para_idx == 0:
                    self.policy_gradient_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=self.RNN_classifier[para_idx], labels=self.child_network_paras[:, para_idx])
                else:
                    self.policy_gradient_loss = tf.add(self.policy_gradient_loss,
                                                       tf.nn.sparse_softmax_cross_entropy_with_logits(
                                                           logits=self.RNN_classifier[para_idx],
                                                           labels=self.child_network_paras[:, para_idx]))
                # get mean of loss
            self.policy_gradient_loss /= self.num_para
            self.total_loss = self.policy_gradient_loss
            self.gradients = self.optimizer.compute_gradients(self.total_loss)

            # Gradients calculated using REINFORCE
            for i, (grad, var) in enumerate(self.gradients):
                if grad is not None:
                    # print("aaa",grad)
                    # print("aaa",self.discounted_rewards)
                    # sys.exit(0)
                    self.gradients[i] = (grad * self.discounted_rewards, var)

        with tf.name_scope('Train_RNN'):
            # The main training operation. This applies REINFORCE on the weights of the Controller
            # self.train_operation[arch_idx][pip_idx] = self.optimizer[arch_idx][pip_idx].apply_gradients(self.gradients[arch_idx][pip_idx], global_step=self.global_step[arch_idx][pip_idx])
            # self.train_operation = self.optimizer.minimize(self.total_loss)
            self.train_operation = self.optimizer.apply_gradients(self.gradients)
            self.update_global_step = tf.assign(self.global_step, self.global_step + 1, name='update_global_step')

        logger.info('Successfully built controller')

    def child_network_translate(self, child_network):
        dnn_out = np.zeros_like(child_network)
        for para_idx in range(self.num_para):
            dnn_out[0][para_idx] = (self.para_2_val[para_idx][child_network[0][para_idx]])
        return dnn_out

    def generate_child_network(self, child_network_architecture):
        with self.graph.as_default():
            feed_dict = {
                self.child_network_paras: child_network_architecture,
                self.batch_size: 1
            }
            rnn_out = self.sess.run(self.RNN_pred_prob, feed_dict=feed_dict)
            predict_child = np.array([[0] * self.num_para])
            # random.seed(datetime.now())
            for para_idx, prob in rnn_out.items():
                predict_child[0][para_idx] = np.random.choice(range(len(self.para_2_val[para_idx])), p=prob[0])
            hyperparameters = self.child_network_translate(predict_child)
            return predict_child, hyperparameters

    def plot_history(self, history, ylim=(-1, 1), title="reward"):
        x = list(range(len(history)))
        y = history
        fig = tpl.figure()
        fig.plot(x, y, ylim=ylim, width=60, height=20, title=title)
        fig.show()

    def get_HW_efficienct(self, Network, HW1, RC):
        # Weiwen 01-24: Using the built Network and HW1 explored results to generate hardware efficiency
        # with the consideration of resource constraint RC
        return random.uniform(0, 1)

    def para2interface_NN(self, Para_NN1):
        # Weiwen 01-24: Build NN using explored hyperparamters, return Network
        Network = -1    # func(Para_NN1)
        return Network

    def para2interface_HW(self, Para_HW1):
        # Weiwen 01-24: Build hardware model using the explored paramters
        HW1 = -1        # func(Para_HW1)
        RC = [HW_constraints["r_Ports_BW"],
              HW_constraints["r_DSP"],
              HW_constraints["r_BRAM"],
              HW_constraints["r_BRAM_Size"],
              HW_constraints["BITWIDTH"]]
        return HW1, RC

    def global_train(self):
        with self.graph.as_default():
            self.sess.run(tf.global_variables_initializer())
        step = 0
        total_rewards = 0
        child_network = np.array([[0] * self.num_para], dtype=np.int64)

        for episode in range(controller_params['max_episodes']):
            logger.info(
                '=-=-==-=-==-=-==-=-==-=-==-=-==-=-=>Episode {}<=-=-==-=-==-=-==-=-==-=-==-=-==-=-='.format(episode))
            step += 1
            episode_reward_buffer = []
            arachitecture_batch = []

            if episode % 50 == 0 and episode != 0:
                print("Process:", str(float(episode) / controller_params['max_episodes'] * 100) + "%", file=sys.stderr)
                self.plot_history(self.reward_history, ylim=(min(self.reward_history)-0.01, max(self.reward_history)+0.01))

            for sub_child in range(controller_params["num_children_per_episode"]):
                # Generate a child network architecture
                child_network, hyperparameters = self.generate_child_network(child_network)

                DNA_NN1 = child_network[0][self.nn1_beg:self.nn1_end]
                DNA_HW1 = child_network[0][self.hw1_beg:self.hw1_end]


                Para_NN1 = hyperparameters[0][self.nn1_beg:self.nn1_end]
                Para_HW1 = hyperparameters[0][self.hw1_beg:self.hw1_end]


                str_NN1 = " ".join(str(x) for x in Para_NN1)
                str_NNs = str_NN1

                str_HW1 = " ".join(str(x) for x in Para_HW1)
                str_HWs = str_HW1

                logger.info('=====>Step {}/{} in episode {}: HyperParameters: {} <====='.format(sub_child, \
                                                                                                controller_params[
                                                                                                    "num_children_per_episode"],
                                                                                                episode,
                                                                                                hyperparameters))

                if str_NNs in self.explored_info.keys():
                    accuracy = self.explored_info[str_NNs][0]
                    reward = self.explored_info[str_NNs][1]
                    HW_Eff = self.explored_info[str_NNs][2]

                else:

                    Network = self.para2interface_NN(Para_NN1)
                    HW1, RC = self.para2interface_HW(Para_HW1)
                    HW_Eff = self.get_HW_efficienct(Network, HW1, RC)

                    # HW Efficiency

                    logger.info('------>Hardware Efficiency Exploreation {}<------'.format(HW_Eff))
                    # Dec. 22: Second loop: search hardware
                    # HW_Eff == -1 indicates that violate the resource constriants
                    if HW_Eff == -1 or HW_Eff > self.target_HW_Eff:
                        for i in range(controller_params["num_hw_per_child"]):
                            child_network, hyperparameters = self.generate_child_network(child_network)
                            l_Para_HW1 = hyperparameters[0][self.hw1_beg:self.hw1_end]

                            str_HW1 = " ".join(str(x) for x in l_Para_HW1)
                            str_HWs = str_HW1
                            DNA_HW1 = child_network[0][self.hw1_beg:self.hw1_end]

                            HW1, RC = self.para2interface_HW(Para_HW1)
                            HW_Eff = self.get_HW_efficienct(Network, HW1, RC)

                            if HW_Eff != -1:
                                logger.info('------>Hardware Exploreation  Success<------')
                                break
                    else:
                        logger.info('------>No Need for Hardware Exploreation<------')

                    if HW_Eff != -1 and HW_Eff <= self.target_HW_Eff:

                        if str_NNs in self.trained_network.keys():
                            accuracy = self.trained_network[str_NNs]
                        else:
                            # comment: train network and obtain accuracy for updating controller
                            accuracy = random.uniform(0, 1)
                            # Keep history trained data

                            # self.trained_network[str_NNs] = accuracy

                        norm_HW_Eff = (self.target_HW_Eff - HW_Eff) / self.target_HW_Eff
                        # Weiwen 01-24: Set weight of HW Eff to 1 for hardware exploration only
                        reward = max(accuracy * 0 + norm_HW_Eff * 1, -1)

                        # Help us to build the history table to avoid optimization for the same network
                        # Weiwen 01-24: We comment this for exploration of hardware
                        # self.explored_info[str_NNs] = {}
                        # self.explored_info[str_NNs][0] = accuracy
                        # self.explored_info[str_NNs][1] = reward
                        # self.explored_info[str_NNs][2] = HW_Eff

                    else:
                        accuracy = 0
                        reward = -1

                logger.info("====================Results=======================")
                logger.info("--------->NN: {}, Accuracy: {}".format(str_NNs, accuracy))
                logger.info("--------->HW: {}, Specs.: {}".format(str_HWs, HW_Eff))
                logger.info("--------->Reward: {}".format(reward))
                logger.info("=" * 50)

                episode_reward_buffer.append(reward)
                identified_arch = np.array(
                    list(DNA_NN1) + list(DNA_HW1))
                arachitecture_batch.append(identified_arch)

            current_reward = np.array(episode_reward_buffer)

            mean_reward = np.mean(current_reward)
            self.reward_history.append(mean_reward)
            self.architecture_history.append(child_network)
            total_rewards += mean_reward

            baseline = ema(self.reward_history)
            last_reward = self.reward_history[-1]
            # rewards = current_reward - baseline
            rewards = [last_reward - baseline]

            feed_dict = {
                self.child_network_paras: arachitecture_batch,
                self.batch_size: len(arachitecture_batch),
                self.discounted_rewards: rewards
            }

            with self.graph.as_default():
                _, _, loss, lr, gs = self.sess.run(
                    [self.train_operation, self.update_global_step, self.total_loss, self.learning_rate,
                     self.global_step], feed_dict=feed_dict)

            logger.info('=-=-=-=-=-=>Episode: {} | Loss: {} | LR: {} | Mean R: {} | Reward: {}<=-=-=-=-='.format(
                episode, loss, (lr, gs), mean_reward, rewards))

        print(self.reward_history)
        self.plot_history(self.reward_history, ylim=(min(self.reward_history)-0.01, max(self.reward_history)-0.01))


# %%


seed = 0
torch.manual_seed(seed)
random.seed(seed)
logging.basicConfig(stream=sys.stdout,
                    level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

print("Begin")
controller = Controller()
controller.global_train()

# %%


