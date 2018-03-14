import gym
import gym_gomoku
import tensorflow as tf
import numpy as np
import random

VAL_EXCLUDE = -100

env = gym.make('Gomoku9x9-v0')

'''
env.reset()
print(env.state.board.encode())
'''


# example 1: take an action
''' 
env.reset() 
env.render() 
env.step(15) # place a single stone, black color first 
'''

class Layer_fc(object):
    def __init__(self, val_in, shape, func_act = tf.nn.relu, dropout = False, keep_prob = 0.5, name = 'fc', lnum = 0):
        self.w = tf.Variable(tf.constant(tf.random_normal(shape)), name = name + '_w' + str(num))
        self.b = tf.Variable(tf.constant(tf.random_normal(shape[1])), name = name + '_b' + str(num))
    
        if dropout:
            self.out_pre = tf.matmul(tf.nn.dropout(val_in, keep_prob), self.w) + self.b
        else:
            self.out_pre = tf.matmul(val_in, self.w) + self.b
                
        self.out = func_act(self.out_pre)

def func_dummy(x): 
    return x

class ReplayMemory(object):
    def __init__(self, size_memory = 1000000):
        self.size = size_memory
        self.memory = list()
        self.game = list()
        self.arr_mem = np.array([])
        self.idx_mem = 0

        # could this ever work??     
    def reverse_step(self, step, bw = [1., -1.]):     # suppose black is represented as 1, white as -1
        return [-step[0], step[1], -step[2], -step[3]]    
    
    def add_step(self, obs_old, action, obs_new, reward):
        #self.memory[-1].append((obs_old, action, obs_new, reward))
        self.game.append([obs_old, action, obs_new, reward])
        self.game.append(reverse_step([obs_old, action, obs_new, reward]))
            
    def new_game(self, gamma = 0.99):
        #self.memory.append(list())
        if len(self.game) < 2: 
            print("ERROR: How could a gomoku game's length be less than 2??")

        # is the commented part below ever necessary???
        '''    
        for j in range(len(self.game)-1):
            self.game[-(j+2)][-1] = self.game[-(j+1)][-1] * gamma # recalculate rewards, discounting from the end
        '''

        self.memory += self.game
        if len(self.memory) > self.size:
            for i in range(len(self.game)):
                self.memory.pop(0)
        self.game = list()

    def update_mem(self, shuffle = True):
        self.arr_mem = np.array(self.memory)
        if shuffle:
            l_idx = range(len(self.memory))
            random.shuffle(l_idx)
            self.arr_mem = np.array(self.memory)[l_idx]
        else:
            self.arr_mem = np.array(self.memory)
    
    def sample_steps(self, size_sample = 100):
        l_idx = range(len(self.memory))
        random.shuffle(l_idx)
        
        return np.array(self.memory)[l_idx][:size_sample] # I don't really think this is the best way...
        
            

class GomokuAI_shson(object):
    def __init__(self, board, shape_fc = [100], shape_conv = [], learning_rate = 0.5, dropout = False):
        self.shape_fc = list()
        self.shape_conv = shape_conv 
        self.layers = list()
        self.keep_prob = tf.placeholder(tf.float32, shape = (), name = 'keep_prob')
        self.learning_rate = tf.Variable(tf.constant(learning_rate, dtype = tf.float32), dtype = tf.float32, name = 'learning_rate')

        board = np.array(board)
        self.size_io = board.shape[0] * board.shape[1]

        self.val_in = tf.placeholder(tf.float32, shape = (None, board.shape[0], board.shape[1]), name = 'placeholder_input')    # placeholder for gomoku board. does it need to contain 'None' in the beginning of the shape??
        self.reward_real = tf.placeholder(tf.float32, shape = ()) # How should I shape the reward??

        if len(shape_conv) == 0: # network consists of only fully connected layers
            self.shape_fc.append(self.size_io)

        for num_shape in shape_fc:
            self.shape_fc.append(num_shape)
        self.shape_fc.append(self.size_io) # last output of the network

        # assigning layers
        if len(shape_conv) == 0:
            val_in = tf.reshape(self.val_in, [-1, self.size_io])
            for i in range(len(self.shape_fc) - 1):
                if i == len(self.shape_fc)-2:
                    self.layers.append(Layer_fc(self.layers[-1].out, [self.shape_fc[i], self.shape_fc[i+1]], func_act = func_dummy, dropout = False, keep_prob = self.keep_prob, lnum = i))    
                else: 
                    self.layers.append(val_in, [self.shape_fc[i], self.shape_fc[i+1]], func_act = tf.nn.relu, dropout = False, keep_prob = self.keep_prob, lnum = i)    
                    val_in = Layer_fc(self.layers[-1].out)
        else:
            print("TODO: assigning conv layers not implemented yet")
            pass    
        
        # how do I define the loss here??????    
        self.q_pre = self.layers[-1].out    
		self.q_valid = tf.gather_nd(self.q_pre, tf.where(self.val_in == 0))
        self.q = tf.argmax(self.q_valid)
		### VERY IMPORTANT: Seems like I need another Q for the next step?? how do i split it??
        self.loss = tf.square(self.q - self.reward_real)
        
        # optimizer
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(loss)
            
        

    def select_action(obs_old):
        pass


# example 2: playing with beginner policy, with basic strike and defend 'opponent'

env.reset()
#AI_shson = GomokuAI_shson(board = env.state.board.encode(), shape_fc = [100], shape_conv = [], learning_rate = 0.5, dropout = False)
obs_old = env.state.board.encode() # TODO: initialize!

num_games_done = 0
num_games_limit = 200

print (dir(env.action_space))
print (env.action_space.valid_spaces)

'''
for _ in range(20):
    #obs_old = observation
    #action = AI_shson.select_action(obs_old)
    action = env.action_space.sample() # sample without replacement
    obs_new, reward, done, info = env.step(action)


    #print (obs_new)
    
    #env.render('human')
    

    if done:    
        num_games_done += 1
        if num_games_done == num_games_limit:
            print ("Game is Over")
            break
        else:
            env.reset()
            obs_new = env.state.board.encode()

    #AI_shson.RL_learn(obs_old, action, obs_new, reward)
    obs_old = obs_new


'''
