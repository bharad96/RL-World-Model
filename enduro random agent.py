import gym
import random
import time
from env import EnduroWrapper
from PIL import Image
import numpy as np
import os


def random_agent():
    env = EnduroWrapper(gym.make('Enduro-v0').env)
    obs = env.reset()
    print(env.get_action_meanings())
    print(env.action_space)

    count = 0
    episode_reward = 0

    repeat = random.randint(1, 3)

    while True:
        if count % 50 == 0:
            print(count)
        # use for debugging
        time.sleep(0.01)
        if count % repeat == 0:
            # action = env.action_space.sample()
            action = random.randint(1, 3)
            repeat = random.randint(1, 3)
        env.render_processed_frame(obs)
        obs, reward, done, info = env.step(action)
        # env.render_processed_frame(obs)
        episode_reward += reward
        count += 1
        if done:
            print('Reward: %s' % episode_reward)
            print(f'Count: ${count}')
            break

random_agent()