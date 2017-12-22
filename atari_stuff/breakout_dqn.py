
import gym
import time
import numpy as np
import random
from statistics import mean, median
from collections import Counter
from collections import deque
from keras.models import Sequential
from scipy.misc import imresize
from keras.layers import Conv2D
from keras.layers.core import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.applications import imagenet_utils
# Deep Q-learning Agent

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=40000)
        self.min_frames = 5000
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Conv2D(32, (8,8), strides=(4, 4), input_shape=self.state_size))
        model.add(Activation('relu'))
        model.add(Conv2D(64,(4, 4), strides=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(64,(3,3)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(np.dstack(state)[None,:,:,:])
        return np.argmax(act_values[0])  # returns action
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                reward = -2
            target = reward
            if not done:
                target = reward + self.gamma * \
                       np.amax(self.model.predict(np.dstack(next_state)[None,:,:,:])[0])
            #print('target: ',target)
            target_f = self.model.predict(np.dstack(state)[None,:,:,:])
            #print('target_f: ', target_f)
            target_f[0][action] = target
            #print('target_f2: ', target_f)
            self.model.fit(np.dstack(state)[None,:,:,:], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def rgb_conv(screen):
    np.array(screen)
    gryscl = np.dot(screen, [.3, .6, .1])
    gryscl = gryscl.reshape(210,160)
    gryscl = imresize(gryscl,(84,84,1))
    gryscl = np.array(gryscl,dtype='f')
    return gryscl

if __name__ == "__main__":
    # initialize gym environment and the agent
    text_file = open("Output.txt", "w")
    next_state = deque(maxlen=3)
    state = deque(maxlen=3)
    env = gym.make('Breakout-v0')
    agent = DQNAgent((84,84,3), 4)  # (state size, action size)
    # Iterate the game
    episodes = 10000
    frame = 0
    for e in range(episodes):
        # reset state in the beginning of each game
        screen = env.reset()
        screen = rgb_conv(screen)#convert to grayscale
        for i in range(episodes):
            state.append(screen) #each state is made up of 3 previous frames
        done = 0
        score = 0
        while done == 0:
            #env.render()
            # Decide action
            action = agent.act(state)
            next_screen, reward, done, _ = env.step(action)
            next_screen = rgb_conv(next_screen)
            state.append(next_screen)
            next_state = state #updates state with newest screen
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}"
                      .format(e, episodes, score))
                break
            # train the agent with the experience of the episode
            score += reward #get total score per game
            frame += 1 #get total frames per game
            if (frame > agent.min_frames or e >2) and (frame % 50 == 0) :
                agent.replay(64)
        #text_file.write("episode: {}/{}, score: {}\n".format(e, episodes, score))
        text_file.write("{}\n".format(score))
        if score >= 8:
            break
    text_file.close()
    agent.model.save('breakoutweights')