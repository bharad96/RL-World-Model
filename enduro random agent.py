import gym
import random
import time
from env import EnduroWrapper
from PIL import Image
import numpy as np
import os


def random_agent():
    env = EnduroWrapper(gym.make('Enduro-v0').env)
    env.reset()

    count = 0
    episode_reward = 0

    repeat = random.randint(1, 11)

    while True:
        if count % 50 == 0:
            print(count)
        # use for debugging
        # time.sleep(0.01)
        if count % repeat == 0:
            action = random.randint(1, 3)
            repeat = random.randint(1, 11)
        env.render()
        obs, reward, done, info = env.step(action)
        # env.render_processed_frame(obs)
        episode_reward += reward
        count += 1
        if done:
            print('Reward: %s' % episode_reward)
            break

random_agent()


# Understand data gen
# exp_name = "WorldModels"
# env_name = "Enduro-v0"
# dirname = 'results/{}/{}/record'.format(exp_name, env_name)
# filenames = os.listdir(dirname)[0:2]  # only use first episode
# n = len(filenames)
# for j, fname in enumerate(filenames):
#     if not fname.endswith('npz'):
#         continue
#     file_path = os.path.join(dirname, fname)
#     # data is a dic with keys obs, action, reward
#     with np.load(file_path) as data:
#         # data['obs'] returns a ndarray: (len, 64, 64, 3), shape[0] returns the num of frames, each frame is 64 * 64 * 3
#         N = data['obs'].shape[0]
#         # for every frame
#         # for i, img in enumerate(data['obs']):
#         #     img_i = img / 255.0
#             # print(img_i)
#
#         action1 = np.reshape(data['action'], newshape=[-1, 1])
#         action2 = data['action']