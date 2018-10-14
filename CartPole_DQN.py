from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
import gym
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory
import json
import matplotlib.pyplot as plt

"""
 Agent and Environment Description

* observation:  
    (Cart Position ,  Cart Velocity ,  Pole Angle ,  Pole Velocity At Tip)
 
* Actions:  
    0 Push cart to the left; 1 Push cart to the right

* Reward:  
    Reward is for every step taken ,  including the termination step

* Starting State:  
    All observations are assigned a uniform random value between +-0.05

* Episode Termination:  
    Pole Angle is more than ±12°
    Cart Position is more than ± 2.4 
    (center of the cart reaches the edge of the display)
    Episode length is greater than 200

* reference:  
    http:  //neuro-educator.com/rl1/

"""


def model_gen(env):
    model = Sequential()
    # input: (n x 4) ( Cart Position, Cart Vel, Pole Angle, Pole Vel)
    model.add(Flatten(input_shape=(1, env.observation_space.shape[0])))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(16))
    model.add(Activation('relu'))

    # output: (n x 3) (left, no, right)
    model.add(Dense(env.action_space.n))
    model.add(Activation('linear'))

    return model


def train():
    env = gym.make('CartPole-v0')

    model = model_gen(env)

    memory = SequentialMemory(limit=50000, window_length=1)  # memory replay

    # epsilon greedy algorithm
    policy = EpsGreedyQPolicy(eps=0.001)

    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, gamma=0.99, memory=memory,
                   nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    history = dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)

    with open('cartpole_history.json', 'w') as f:
        json.dump(history.history, f)

    dqn.save_weights('cartpole_dqn.hdf5')


def test():
    env = gym.make('CartPole-v0')

    model = model_gen(env)
    memory = SequentialMemory(limit=50000, window_length=1)  # memory replay
    policy = EpsGreedyQPolicy(eps=0.001)

    dqn = DQNAgent(model=model, nb_actions=env.action_space.n, gamma=0.99, memory=memory,
                   nb_steps_warmup=10, target_model_update=1e-2, policy=policy)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])

    dqn.load_weights('cartpole_dqn.hdf5')
    dqn.test(env, nb_episodes=5, visualize=True)
    env.close()

    y = None
    with open('cartpole_history.json', 'r') as f:
        y = json.load(f)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('DQN_rewards')
    ax.set_xlabel('episodes')
    ax.set_ylabel('rewards')
    ax.plot(y['episode_reward'])
    plt.savefig('CartPole_DQN_rewards')
    plt.show()


def main():
    print('1: train, 0 or others: test')
    flag = input()
    if flag == '1':
        train()
    else:
        test()


if __name__ == "__main__":
    main()
