'''
saves ~ 200 episodes generated from a random policy
'''

import numpy as np
import random
import os
import gym
import time

from env import EnduroWrapper, EnduroMDNRNN
# from controller import make_controller

from utils import PARSER

args = PARSER.parse_args()
dir_name = 'results/{}/{}/record'.format(args.exp_name, args.env_name)
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# controls whether we concatenate (z, c, h), etc for features used for car.
# controller = make_controller(args=args)

total_frames = 0

# env = make_env(args=args, render_mode=args.render_mode, full_episode=args.full_episode, with_obs=True, load_model=False)
env = EnduroWrapper(gym.make('Enduro-v0').env)  # Used to extract obs and action for training vae and mdnrnn

for trial in range(args.max_trials):
    try:
        random_generated_int = random.randint(0, 2 ** 31 - 1)
        filename = dir_name + "/" + str(random_generated_int) + ".npz"

        recording_frame = []
        recording_action = []
        recording_reward = []

        np.random.seed(random_generated_int)
        env.seed(random_generated_int)

        repeat = np.random.randint(1, 11)

        tot_r = 0
        # [obs, frame] = env.reset() # pixels
        frame = env.reset()  # pixels

        done = False
        i=0
        while not done:
            if args.render_mode:
                # human
                env.render_processed_frame(frame)
                time.sleep(0.02)

            recording_frame.append(frame)

            if i % repeat == 0:
                # up, right, left and down only
                action = np.random.randint(1, 4)
                repeat = np.random.randint(1, 11)

            recording_action.append(action)

            # [obs, frame], reward, done, info = env.step(action)
            frame, reward, done, info = env.step(action)
            tot_r += reward
            recording_reward.append(reward)
            i += 1

        total_frames += (i + 1)
        print('total reward {}'.format(tot_r))
        print("dead at", i + 1, "total recorded frames for this worker", total_frames)
        recording_frame = np.array(recording_frame, dtype=np.uint8)
        recording_action = np.array(recording_action, dtype=np.float16)
        recording_reward = np.array(recording_reward, dtype=np.float16)

        if (len(recording_frame) > args.min_frames):
            np.savez_compressed(filename, obs=recording_frame, action=recording_action, reward=recording_reward)
    except gym.error.Error:
        print("stupid gym error, life goes on")
        env.close()
        env = EnduroWrapper(gym.make('Enduro-v0').env)  # Used to extract obs and action for training vae and mdnrnn
        continue
env.close()
