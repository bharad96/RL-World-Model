import gym
import random
import time
from env import BoxingWrapper
from env import BoxingMDNRNN, make_env
from PIL import Image
import numpy as np
import os
from utils import PARSER

def random_agent():
    # env = gym.make('Boxing-v0').env
    env = BoxingWrapper(gym.make('Boxing-v0').env)
    env.reset()
    random.seed(100)

    count = 0
    episode_reward = 0
    print(env.get_action_meanings())
    print(env.action_space)

    # repeat = random.randint(1, 11)

    while True:
        # use for debugging
        # time.sleep(0.01)
        # if count % repeat == 0:
        # action = random.randint(0, env.action_space)
            # repeat = random.randint(1, 11)
        # env.render()
        action = random.randint(2, 4)
        # action = env.action_space.sample()
        time.sleep(0.01)
        obs, reward, done, info = env.step(action)

        env.render_processed_frame(obs)
        episode_reward += reward

        count += 1
        if done:
            print('Reward: %s' % episode_reward)
            print('Steps: %s' % count)
            break

random_agent()

# Understand data gen
# exp_name = "WorldModels"
# env_name = "Boxing-v0"
# dirname = 'results/{}/{}/record'.format(exp_name, env_name)
# filenames = os.listdir(dirname)[0:]  # only use first episode
# n = len(filenames)
#
# len_lst = []
#
# for j, fname in enumerate(filenames):
#     if not fname.endswith('npz'):
#         continue
#     file_path = os.path.join(dirname, fname)
#     # data is a dic with keys obs, action, reward
#     with np.load(file_path) as data:
#         # data['obs'] returns a ndarray: (len, 64, 64, 3), shape[0] returns the num of frames, each frame is 64 * 64 * 3
#         N = np.zeros(100, np.uint16)
#         action = data['action']
#         reward = data['reward']
#         done = data['done']
#
#         len_lst.append(len(reward))
        # for every frame
        # for i, img in enumerate(data['obs']):
        #     img_i = img / 255.0
            # print(img_i)

#         action1 = np.reshape(data['action'], newshape=[-1, 1])
#
# print("Max length: " , max(len_lst))
# print("Min length: " , min(len_lst))
#
# def BoxingMDNRNNAgent(args):
#     env = make_env(args=args, dream_env=args.dream_env)
#
#
# if __name__ == "__main__":
#   args = PARSER.parse_args()
#   BoxingMDNRNNAgent(args)