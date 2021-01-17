import gym
import random
import time

def main():
    env = gym.make('SpaceInvaders-v0')
    env.reset()

    episode_reward = 0
    while True:
        env.render()
        time.sleep(0.1)
        action = env.action_space.sample()
        _, reward, done, _ = env.step(action)
        episode_reward += reward
        if done:
            print('Reward: %s' % episode_reward)
            break

main()