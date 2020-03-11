import gym, random
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v0')

action_space = env.action_space.n
observation_space = env.observation_space.shape[0]
print("Observation space: "+str(observation_space))
print("Action space: "+str(action_space))

num_episodes = 10
max_steps_per_episode = 200

pvariance = 0.1
# variance of initial parameters
ppvariance = 0.02
# variance of perturbations
nhiddens = 5
# number of hidden neurons

ninputs = env.observation_space.shape[0]
if (isinstance(env.action_space, gym.spaces.box.Box)):
	noutputs = env.action_space.shape[0]
else:
	noutputs = env.action_space.n
# initialize the training parameters randomly by using a gaussian distribution with average 0.0 and variance 0.1
# biases (thresholds) are initialized to 0.0


rewards_all_episodes = []
count = 0
for episode in range(num_episodes):
    state = env.reset()
    W1 = np.random.randn(nhiddens,ninputs) * pvariance
    W2 = np.random.randn(noutputs, nhiddens) * pvariance
    b1 = np.zeros(shape=(nhiddens, 1))
    b2 = np.zeros(shape=(noutputs, 1))

    done = False
    reward_current_episode = 0
    
    for step in range (max_steps_per_episode):

        state.resize(ninputs,1)
        Z1 = np.dot(W1, state) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        A2 = np.tanh(Z2)

        if (isinstance(env.action_space, gym.spaces.box.Box)):
            action = A2
        else:
            action = np.argmax(A2)
        new_state, reward, done, info = env.step(action)
        
        state = new_state
        reward_current_episode = reward_current_episode + reward
        
        if done==True:
            break
            
    rewards_all_episodes.append(reward_current_episode)
#    rr = sum(rewards_all_episodes)/len(rewards_all_episodes)
    print("Episode Count: "+str(episode)+", Rewards collected: "+str(reward_current_episode))

plt.plot(np.linspace(1, num_episodes, num_episodes), rewards_all_episodes)
plt.scatter(np.linspace(1, num_episodes, num_episodes), rewards_all_episodes)
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.show()
