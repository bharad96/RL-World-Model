import gym
import random
import time
from env import EnduroWrapperBW
from PIL import Image
import numpy as np
import os


def random_agent():
    env = gym.make('SpaceInvaders-v0').env
    env.reset()

    count = 0
    episode_reward = 0

    repeat = random.randint(1, 11)

    while True:
        # use for debugging
        # time.sleep(0.01)
        if count % repeat == 0:
            action = random.randint(1, 3)
            repeat = random.randint(1, 11)
        env.render()
        time.sleep(0.05)
        obs, reward, done, info = env.step(action)
        cropped_obs = obs[20:195, :, :]
        # env.render_processed_frame(obs)
        episode_reward += reward

        frame = obs[20:195, :, :]
        frame = Image.fromarray(frame, mode='RGB').resize((64, 64))
        frame = np.array(frame)

        if count > 50 and count % 10 == 0:
            print(frame)


        count += 1
        if done:
            print('Reward: %s' % episode_reward)
            break

random_agent()