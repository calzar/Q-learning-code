import gym
import time
from keras.models import load_model
import numpy as np
import random
import matplotlib.pyplot as plt
from statistics import mean, median
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical


def plot_funct(scores):
    games = list(range(0,len(scores)))
    print(scores)
    print(games)
    plt.figure(1)
    plt.plot(games, scores)
    plt.ylabel('Score')
    plt.xlabel('Games Played')
    plt.show()



#loads a neural network with weights that have already been trained to test the perfomance of particular CartPole agent
model = load_model('bestdqnv2')
env = gym.make('CartPole-v1')
scores = []
choices = []
for each_game in range(50): #number of games
    score = 0
    game_memory = []
    prev_obs = []
    env.reset()
    for _ in range(500):#goal steps
        #env.render()
        #time.sleep(.01)
        if len(prev_obs) == 0:
            action = random.randrange(0, 2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1, 4)))
            #print(model.predict(prev_obs.reshape(-1, 4)))
        #print(action)
        choices.append(action)
        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break
    print(score)
    scores.append(score)
print('Average Score:', sum(scores) / len(scores))
print('choice 1:{}  choice 0:{}'.format(choices.count(1) / len(choices), choices.count(0) / len(choices)))
plot_funct(scores)


#to manually read model:
# save as JSON
#json_string = model.to_json()
# save as YAML
#yaml_string = model.to_yaml()