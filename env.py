import numpy as np
import gym
import json
import os
import tensorflow as tf
import gc
import cv2
cv2.ocl.setUseOpenCL(False)
from PIL import Image
from gym.spaces.box import Box

# from ppaquette_gym_doom.doom_take_cover import DoomTakeCoverEnv
from gym.utils import seeding
# class DoomTakeCoverMDNRNN(DoomTakeCoverEnv):
#     def __init__(self, args, render_mode=False, load_model=True, with_obs=False):
#         super(DoomTakeCoverMDNRNN, self).__init__()
#
#         self.with_obs = with_obs
#
#         self.no_render = True
#         if render_mode:
#             self.no_render = False
#         self.current_obs = None
#
#         self.vae = CVAE(args)
#         self.rnn = MDNRNN(args)
#
#         if load_model:
#             self.vae.set_weights([param_i.numpy() for param_i in tf.saved_model.load(
#                 'results/{}/{}/tf_vae'.format(args.exp_name, args.env_name)).variables])
#             self.rnn.set_weights([param_i.numpy() for param_i in tf.saved_model.load(
#                 'results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name)).variables])
#
#         self.action_space = Box(low=-1.0, high=1.0, shape=())
#         self.obs_size = self.rnn.args.z_size + self.rnn.args.rnn_size * self.rnn.args.state_space
#
#         self.observation_space = Box(low=0, high=255, shape=(64, 64, 3))
#         self.actual_observation_space = Box(low=-50., high=50., shape=(self.obs_size))
#
#         self._seed()
#
#         self.rnn_states = None
#         self.z = None
#         self.restart = None
#         self.frame_count = None
#         self.viewer = None
#         self._reset()
#
#     def close(self):
#         super(DoomTakeCoverMDNRNN, self).close()
#         tf.keras.backend.clear_session()
#         gc.collect()
#
#     def _step(self, action):
#
#         # update states of rnn
#         self.frame_count += 1
#
#         self.rnn_states = rnn_next_state(self.rnn, self.z, action, self.rnn_states)
#
#         # actual action in wrapped env:
#
#         threshold = 0.3333
#         full_action = [0] * 43
#
#         if action < -threshold:
#             full_action[11] = 1
#
#         if action > threshold:
#             full_action[10] = 1
#
#         obs, reward, done, _ = super(DoomTakeCoverMDNRNN, self)._step(full_action)
#         small_obs = self._process_frame(obs)
#         self.current_obs = small_obs
#         self.z = self._encode(small_obs)
#
#         if done:
#             self.restart = 1
#         else:
#             self.restart = 0
#
#         if self.with_obs:
#             return [self._current_state(), self.current_obs], reward, done, {}
#         else:
#             return self._current_state(), reward, done, {}
#
#     def _encode(self, img):
#         simple_obs = np.copy(img).astype(np.float) / 255.0
#         simple_obs = simple_obs.reshape(1, 64, 64, 3)
#         z = self.vae.encode(simple_obs)[0]
#         return z
#
#     def _reset(self):
#         obs = super(DoomTakeCoverMDNRNN, self)._reset()
#         small_obs = self._process_frame(obs)
#         self.current_obs = small_obs
#         self.rnn_states = rnn_init_state(self.rnn)
#         self.z = self._encode(small_obs)
#         self.restart = 1
#         self.frame_count = 0
#
#         if self.with_obs:
#             return [self._current_state(), self.current_obs]
#         else:
#             return self._current_state()
#
#     def _process_frame(self, frame):
#         obs = frame[0:400, :, :]
#         obs = Image.fromarray(obs, mode='RGB').resize((64, 64))
#         obs = np.array(obs)
#         return obs
#
#     def _current_state(self):
#         if self.rnn.args.state_space == 2:
#             return np.concatenate(
#                 [self.z, tf.keras.backend.flatten(self.rnn_states[1]), tf.keras.backend.flatten(self.rnn_states[0])],
#                 axis=0)  # cell then hidden fro some reason
#         return np.concatenate([self.z, tf.keras.backend.flatten(self.rnn_states[0])], axis=0)  # only the hidden state
#
#     def _seed(self, seed=None):
#         if seed:
#             tf.random.set_seed(seed)
#         self.np_random, seed = seeding.np_random(seed)
#         return [seed]

class EnduroWrapperBW(gym.ObservationWrapper):
    def observation(self, observation):
        obs = observation[51:151, 30:130, :]  # this corresponds to the car racing area
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs,  (64, 64), interpolation=cv2.INTER_AREA)
        obs = np.array(obs)
        return obs

    def __init__(self, env, full_episode=False):
        super(EnduroWrapperBW, self).__init__(env)
        self.full_episode = full_episode
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 1))  # , dtype=np.uint8

    def step(self, action):
        obs, reward, done, _ = super(EnduroWrapperBW, self).step(action)
        if self.full_episode:
            return obs, reward, False, {}
        return obs, reward, done, {}

    def render_processed_frame(self, img):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)
        return self.viewer.isopen

class EnduroWrapperColor(gym.ObservationWrapper):
    def observation(self, observation):
        obs = observation[51:151, 30:130, :]  # this corresponds to the car racing area
        # obs = Image.fromarray(obs, mode='RGB').resize((64, 64))
        obs = np.array(obs)
        return obs

    def __init__(self, env, full_episode=False):
        super(EnduroWrapperColor, self).__init__(env)
        self.full_episode = full_episode
        self.observation_space = Box(low=0, high=255, shape=(100, 100, 1))  # , dtype=np.uint8

    def step(self, action):
        obs, reward, done, _ = super(EnduroWrapperColor, self).step(action)
        if self.full_episode:
            return obs, reward, False, {}
        return obs, reward, done, {}

    def render_processed_frame(self, img):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)
        return self.viewer.isopen


class BoxingWrapper(gym.ObservationWrapper):
    def observation(self, observation):
        obs = observation[36:177, 32:128, :]  # this corresponds to the car racing area
        obs = Image.fromarray(obs, mode='RGB').resize((64, 64))
        obs = np.array(obs)
        return obs

    def __init__(self, env, full_episode=False):
        super(BoxingWrapper, self).__init__(env)
        self.full_episode = full_episode
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 3))  # , dtype=np.uint8

    def step(self, action):
        obs, reward, done, _ = super(BoxingWrapper, self).step(action)
        if self.full_episode:
            return obs, reward, False, {}
        return obs, reward, done, {}

    def render_processed_frame(self, img):
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.SimpleImageViewer()
        self.viewer.imshow(img)
        return self.viewer.isopen


from vae.vae import CVAE
from rnn.rnn import MDNRNN, rnn_next_state, rnn_init_state
from gym.utils import seeding


class BoxingMDNRNN(BoxingWrapper):
    def __init__(self, args, load_model=True, full_episode=False, with_obs=False):
        super(BoxingMDNRNN, self).__init__(gym.make('Boxing-v0').env, full_episode=full_episode)
        self.with_obs = with_obs  # whether or not to return the frame with the encodings
        self.vae = CVAE(args)
        self.rnn = MDNRNN(args)

        if load_model:
            self.vae.set_weights([param_i.numpy() for param_i in tf.saved_model.load(
                'results/{}/{}/tf_vae'.format(args.exp_name, args.env_name)).variables])
            self.rnn.set_weights([param_i.numpy() for param_i in tf.saved_model.load(
                'results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name)).variables])

        self.full_episode = False
        self.obs_size = self.rnn.args.z_size + self.rnn.args.rnn_size * self.rnn.args.state_space
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 3))
        self.actual_observation_space = Box(low=np.NINF, high=np.Inf, shape=(self.obs_size,))

        self.seed()

        self.rnn_states = None
        self.z = None
        self.restart = None
        self.reset()

    def encode_obs(self, obs):
        # convert raw obs to z, mu, logvar
        result = np.copy(obs).astype(np.float) / 255.0
        result = result.reshape(1, 64, 64, 3)
        z = self.vae.encode(result)[0]
        return z

    def reset(self):
        self.current_obs = super(BoxingMDNRNN, self).reset()
        self.rnn_states = rnn_init_state(self.rnn)
        self.z = self.encode_obs(self.current_obs)
        self.restart = 1
        self.frame_count = 0
        if self.with_obs:
            return [self.current_state(), self.current_obs]
        else:
            return self.current_state()

    def current_state(self):
        if self.rnn.args.state_space == 2:
            return np.concatenate(
                [self.z, tf.keras.backend.flatten(self.rnn_states[1]), tf.keras.backend.flatten(self.rnn_states[0])],
                axis=0)  # cell then hidden fro some reason
        return np.concatenate([self.z, tf.keras.backend.flatten(self.rnn_states[0])], axis=0)  # only the hidden state


    def step(self, action):
        # update states of rnn
        self.frame_count += 1
        self.rnn_states = rnn_next_state(self.rnn, self.z, action, self.rnn_states)

        threshold = 0.3333
        if action < -threshold:
            env_action = 3 # left
        elif action > threshold:
            env_action = 2  # right
        else:
            env_action = 1  # up

        self.current_obs, reward, done, _ = super(BoxingMDNRNN, self).step(env_action)
        self.z = self.encode_obs(self.current_obs)

        if done:
            self.restart = 1
        else:
            self.restart = 0

        if self.with_obs:
            return [self.current_state(), self.current_obs], reward, done, {}
        else:
            return self.current_state(), reward, done, {}


    def seed(self, seed=None):
        if seed:
            tf.random.set_seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        super(BoxingMDNRNN, self).close()
        tf.keras.backend.clear_session()
        gc.collect()

class EnduroMDNRNN(BoxingWrapper):
    def __init__(self, args, load_model=True, full_episode=False, with_obs=False):
        super(BoxingMDNRNN, self).__init__(gym.make('Enduro-v0').env, full_episode=full_episode)
        self.with_obs = with_obs  # whether or not to return the frame with the encodings
        self.vae = CVAE(args)
        self.rnn = MDNRNN(args)

        if load_model:
            self.vae.set_weights([param_i.numpy() for param_i in tf.saved_model.load(
                'results/{}/{}/tf_vae'.format(args.exp_name, args.env_name)).variables])
            self.rnn.set_weights([param_i.numpy() for param_i in tf.saved_model.load(
                'results/{}/{}/tf_rnn'.format(args.exp_name, args.env_name)).variables])

        self.full_episode = False
        self.obs_size = self.rnn.args.z_size + self.rnn.args.rnn_size * self.rnn.args.state_space
        self.observation_space = Box(low=0, high=255, shape=(64, 64, 3))
        self.actual_observation_space = Box(low=np.NINF, high=np.Inf, shape=(self.obs_size,))

        self.seed()

        self.rnn_states = None
        self.z = None
        self.restart = None
        self.reset()

    def encode_obs(self, obs):
        # convert raw obs to z, mu, logvar
        result = np.copy(obs).astype(np.float) / 255.0
        result = result.reshape(1, 64, 64, 3)
        z = self.vae.encode(result)[0]
        return z

    def reset(self):
        self.current_obs = super(EnduroMDNRNN, self).reset()
        self.rnn_states = rnn_init_state(self.rnn)
        self.z = self.encode_obs(self.current_obs)
        self.restart = 1
        self.frame_count = 0
        if self.with_obs:
            return [self.current_state(), self.current_obs]
        else:
            return self.current_state()

    def current_state(self):
        if self.rnn.args.state_space == 2:
            return np.concatenate(
                [self.z, tf.keras.backend.flatten(self.rnn_states[1]), tf.keras.backend.flatten(self.rnn_states[0])],
                axis=0)  # cell then hidden fro some reason
        return np.concatenate([self.z, tf.keras.backend.flatten(self.rnn_states[0])], axis=0)  # only the hidden state


    def step(self, action):
        # update states of rnn
        self.frame_count += 1
        self.rnn_states = rnn_next_state(self.rnn, self.z, action, self.rnn_states)

        # if action < -0.667:
        #     env_action = 0  # no operation
        if action < -0.6:
            env_action = 1  # punch
        elif action < -0.2:
            env_action = 2  # up
        elif action < 0.2:
            env_action = 5  # down
        elif action < 0.6:
            env_action = 4  # left
        else:
            env_action = 3  # right

        self.current_obs, reward, done, _ = super(BoxingMDNRNN, self).step(env_action)
        self.z = self.encode_obs(self.current_obs)

        if done:
            self.restart = 1
        else:
            self.restart = 0

        if self.with_obs:
            return [self.current_state(), self.current_obs], reward, done, {}
        else:
            return self.current_state(), reward, done, {}


    def seed(self, seed=None):
        if seed:
            tf.random.set_seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        super(BoxingMDNRNN, self).close()
        tf.keras.backend.clear_session()
        gc.collect()


def make_env(args, dream_env=False, seed=-1, render_mode=False, full_episode=False, with_obs=False, load_model=True):
    if args.env_name == 'Enduro-v0':
        if dream_env:
            raise ValueError('training in dreams for Enduro-v0 is not yet supported')
        else:
            print('making real Enduro-v0 environment')
            # using .env to remove the time wrapper
            env = EnduroMDNRNN(args, load_model, full_episode, with_obs)
    # if args.env_name == 'Boxing-v0':
    #     if dream_env:
    #         raise ValueError('training in dreams for Boxing-v0 is not yet supported')
    #     else:
    #         print('making real Boxing-v0 environment')
    #         # using .env to remove the time wrapper
    #         env = BoxingMDNRNN(args, load_model, full_episode, with_obs)
    if (seed >= 0):
        env.seed(seed)
    return env
