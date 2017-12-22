import gym
import time
import numpy as np
import random
from statistics import mean, median
from collections import Counter
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical
from keras.applications import imagenet_utils
from keras.datasets import mnist

# Deep Q-learning Agent
#trains the agent to play cartpole and saves the weights, scores of each game, and the loss of each action
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(15, input_dim=self.state_size, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',optimizer='adam')
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if done:
                reward = -100
            target = reward
            if not done:
                target = reward + self.gamma * \
                       np.amax(self.model.predict(next_state)[0])
            #print('nstate preict: ',self.model.predict(next_state))
            target_f = self.model.predict(state)
            #print('target_f: ', target_f)
            loss = (target - target_f[0][action])*(target - target_f[0][action])
            #loss_file.write("{}\n".format(loss))
            target_f[0][action] = target
            #print('target_f2: ', target_f)
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

if __name__ == "__main__":
    # initialize gym environment and the agent
    env = gym.make('CartPole-v0')
    agent = DQNAgent(4,2) #(state size, action size)
    # Iterate the game
    episodes = 700
    counter = 0
    score_file = open("cart_score.txt", "w")
    loss_file = open('cart_loss.txt','w')
    for e in range(episodes):
        # reset state in the beginning of each game
        state = env.reset()
        state = np.reshape(state, [1, 4])
        # time_t represents each frame of the game
        # Our goal is to keep the pole upright as long as possible until score of 500
        # the more time_t the more score
        for time_t in range(500):
            #env.render()
            # Decide action
            action = agent.act(state)
            # Advance the game to the next frame based on the action.
            # Reward is 1 for every frame the pole survived
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, 4])
            # Remember the previous state, action, reward, and done
            agent.remember(state, action, reward, next_state, done)
            # make next_state the new current state for the next frame.
            state = next_state
            # done becomes True when the game ends
            # ex) The agent drops the pole
            if done:
                # print the score and break out of the loop
                print("episode: {}/{}, score: {}, counter: {}"
                      .format(e, episodes, time_t, counter))
                break
            # train the agent with the experience of the episode
        #score_file.write("{}\n".format(time_t))
        if time_t >= 199:
            counter += 1
        else:
            counter = 0
        if counter >= 12:
            break

        if time_t >= 64 or e > 3:
            agent.replay(64)
    #agent.model.save('bestdqnv2')
    score_file.close()
    loss_file.close()