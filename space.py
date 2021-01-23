import gym
import random
import time
from env import EnduroWrapper
from PIL import Image
import numpy as np

def main():
    # env = gym.make('SpaceInvaders-v0')
    env = EnduroWrapper(gym.make('Enduro-v0').env)
    env.reset()

    count = 0
    episode_reward = 0

    repeat = random.randint(1, 11)

    while True:
        # use for debugging
        # time.sleep(0.01)
        if count % repeat == 0:
            action = random.randint(1,3)
            repeat = random.randint(1, 11)

        obs, reward, done, info = env.step(action)
        env.render_processed_frame(obs)
        episode_reward += reward
        if done:
            print('Reward: %s' % episode_reward)
            break

main()