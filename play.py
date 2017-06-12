# for snake(snaky)
import snaky as game
import cv2

# for tensor
import numpy as np
import tensorflow as tf
import random
from collections import deque

# Based on NIPS 2013
class DQN:
    def __init__(self, DISCFT, FLAG, INIT_EPSILON, FIN_EPSILON, REPLAY_MEMORY, BATCH_SIZE, ACTIONS):
        # Initialize Variables
        # epoch - frame
        # episode - one round
        self.epoch = 0
        self.episode = 0
        self.observe = 500000
        # discount factor
        self.discft = DISCFT
        # FLAG
        # 0 - train
        # 1 - play
        self.flag = FLAG
        self.epsilon = INIT_EPSILON
        self.finep = FIN_EPSILON
        self.REPLAYMEM = REPLAY_MEMORY
        self.batchsize = BATCH_SIZE
        self.actions = ACTIONS
        self.repmem = deque()
        # Init weight and bias
        self.w1 = tf.Variable(tf.truncated_normal([8, 8, 4, 32], stddev = 0.01))
        self.b1 = tf.Variable(tf.constant(0.01, shape = [32]))

        self.w2 = tf.Variable(tf.truncated_normal([4, 4, 32, 64], stddev=0.01))
        self.b2 = tf.Variable(tf.constant(0.01, shape = [64]))

        self.w3 = tf.Variable(tf.truncated_normal([3, 3, 64, 64], stddev = 0.01))
        self.b3 = tf.Variable(tf.constant(0.01, shape = [64]))

        self.wfc = tf.Variable(tf.truncated_normal([2304, 512], stddev = 0.01))
        self.bfc = tf.Variable(tf.constant(0.01, shape = [512]))

        self.wto = tf.Variable(tf.truncated_normal([512, self.actions], stddev = 0.01))
        self.bto = tf.Variable(tf.constant(0.01, shape = [self.actions]))

        self.initConvNet()
        self.initNN()
    
    def initConvNet(self):
        # input layer
        self.input = tf.placeholder("float", [None, 84, 84, 4])

        # Convolutional Neural Network
        # zero-padding
        # 84 x 84 x 4
        # 8 x 8 x 4 with 32 Filters
        # Stride 4 -> Output 21 x 21 x 32 -> max_pool 11 x 11 x 32
        tf.nn.conv2d(self.input, self.w1, strides = [1, 4, 4, 1], padding = "SAME")
        conv1 = tf.nn.relu(tf.nn.conv2d(self.input, self.w1, strides = [1, 4, 4, 1], padding = "SAME") + self.b1)
        pool = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

        # 11 x 11 x 32
        # 4 x 4 x 32 with 64 Filters
        # Stride 2 -> Output 6 x 6 x 64
        conv2 = tf.nn.relu(tf.nn.conv2d(pool, self.w2, strides = [1, 2, 2, 1], padding = "SAME") + self.b2)

        # 6 x 6 x 64
        # 3 x 3 x 64 with 64 Filters
        # Stride 1 -> Output 6 x 6 x 64
        conv3 = tf.nn.relu(tf.nn.conv2d(conv2, self.w3, strides = [1, 1, 1, 1], padding = "SAME") + self.b3)

        # 6 x 6 x 64 = 2304
        conv3_to_reshaped = tf.reshape(conv3, [-1, 2304])

        # Matrix (1, 2304) * (2304, 512)
        fullyconnected = tf.nn.relu(tf.matmul(conv3_to_reshaped, self.wfc) + self.bfc)

        # output(Q) layer
        # Matrix (1, 512) * (512, ACTIONS) -> (1, ACTIONS)
        self.output = tf.matmul(fullyconnected, self.wto) + self.bto

    def initNN(self):
        self.a = tf.placeholder("float", [None, self.actions])
        self.y = tf.placeholder("float", [None]) 
        out_action = tf.reduce_sum(tf.multiply(self.output, self.a), reduction_indices = 1)
        self.cost = tf.reduce_mean(tf.square(self.y - out_action))
        self.optimize = tf.train.AdamOptimizer(1e-6).minimize(self.cost)
        
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())
        checkpoint = tf.train.get_checkpoint_state("saved")
        # For fresh start, comment below 2 lines
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
    
    def addReplay(self, s_t1, action, reward, done):
        tmp = np.append(self.s_t[:,:,1:], s_t1, axis = 2)
        self.repmem.append((self.s_t, action, reward, tmp, done))
        if len(self.repmem) > self.REPLAYMEM:
            self.repmem.popleft()

        self.s_t = tmp
        self.epoch += 1
        
        return self.epoch, np.max(self.qv)
        
    def getAction(self):
        Q_val = self.output.eval(feed_dict={self.input : [self.s_t]})[0]
        # for print
        self.qv = Q_val
        # action array
        action = np.zeros(self.actions)
        idx = 0

        # epsilon greedily
        if random.random() <= self.epsilon:
            idx = random.randrange(self.actions)
            action[idx] = 1
        else:
            idx = np.argmax(Q_val)
            action[idx] = 1

        return action

    def initState(self, state):
        self.s_t = np.stack((state, state, state, state), axis=2)

class agent:
    def screen_handle(self, screen):
        procs_screen = cv2.cvtColor(cv2.resize(screen, (84, 84)), cv2.COLOR_BGR2GRAY)
        dummy, bin_screen = cv2.threshold(procs_screen, 1, 255, cv2.THRESH_BINARY)
        bin_screen = np.reshape(bin_screen, (84, 84, 1))
        return bin_screen
        
    def run(self):
        # initialize
        # discount factor 0.99
        ag = DQN(0.99, 0, 0.001, 0.001, 50000, 32, 4)
        g = game.gameState()
        a_0 = np.array([1, 0, 0, 0])
        s_0, r_0, d = g.frameStep(a_0)
        s_0 = cv2.cvtColor(cv2.resize(s_0, (84, 84)), cv2.COLOR_BGR2GRAY)
        _, s_0 = cv2.threshold(s_0, 1, 255, cv2.THRESH_BINARY)
        ag.initState(s_0)
        while True:
            a = ag.getAction()
            s_t1, r, done = g.frameStep(a)
            s_t1 = self.screen_handle(s_t1)
            ts, qv = ag.addReplay(s_t1, a, r, done)
            # for Summary
            if done == True:
                sc, ep = g.retScore()
                print(ts,",",qv,",",ep, ",", sc)
            else:
                print(ts,",",qv,",,")

def main():
    run_agent = agent()
    run_agent.run()

if __name__ == '__main__':
    main()