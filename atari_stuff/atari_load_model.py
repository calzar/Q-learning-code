import gym
import time
import O
from keras.models import load_model
import numpy as np
from collections import deque
from ..atari_stuff import breakout_dqn
from statistics import mean, median
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.utils import to_categorical

model = load_model('breakoutweights')
next_state = deque(maxlen=3)
state = deque(maxlen=3)
env = gym.make('Breakout-v0')
# Iterate the game
episodes = 10
frame = 0
for e in range(episodes):
    # reset state in the beginning of each game
    screen = env.reset()
    screen = breakout_dqn.rgb_conv(screen)#convert to grayscale
    for i in range(3):
        state.append(screen) #each state is made up of 3 previous frames
    done = 0
    score = 0
    while done == 0:
        #env.render()
        # Decide action
        action = np.argmax(model.predict(np.dstack(state)[None,:,:,:])[0])
        print(action)
        # Advance the game to the next frame based on the action.
        # Reward is 1 for every block destroyed
        next_screen, reward, done, _ = env.step(action)
        next_screen = rgb_conv(next_screen)
        state.append(next_screen)
        next_state = state #updates state with newest screen
        # Remember the previous state, action, reward, and done
        #agent.remember(state, action, reward, next_state, done)
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


#to manually read model:
# save as JSON
#json_string = model.to_json()
# save as YAML
#yaml_string = model.to_yaml()