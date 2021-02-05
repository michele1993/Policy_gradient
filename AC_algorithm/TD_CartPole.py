import torch
import gym
import torch.optim as opt

from torch.distributions import Categorical

from Policy_gradient.AC_algorithm.TD_Actor_NN import Actor_net
from Policy_gradient.AC_algorithm.Critic_NN import Critic_NN


env = gym.make("CartPole-v1")

n_episodes = 10000
max_t_steps = 200
discount= 0.99
ln_rate_c = 0.001
ln_rate_a = 0.0001


actor = Actor_net(discount = discount).double()
critic = Critic_NN(discount = discount).double()

optimiser_1 = opt.Adam(actor.parameters(), ln_rate_a)
optimiser_2 = opt.Adam(critic.parameters(), ln_rate_c)

#parameters = list(actor.parameters()) + list(critic.parameters())

#optimiser = opt.Adam(parameters,ln_rate)

av_return = []

for ep in range(n_episodes):

    c_state = env.reset()

    t = 0

    for t in range(max_t_steps):

        mean_action = actor(torch.tensor(c_state))

        d = Categorical(mean_action)

        action = d.sample()

        lp_action = d.log_prob(action)

        n_state,rwd,done,_ = env.step(action.numpy())

        TD_error = critic.advantage(c_state,n_state,rwd,done)

        rf_cost = actor.REINFORCE(lp_action,TD_error)

        critic_cost = TD_error**2

        loss = rf_cost + critic_cost

        #optimiser.zero_grad()

        optimiser_1.zero_grad()
        optimiser_2.zero_grad()

        loss.backward()

        #optimiser.step()
        optimiser_1.step()
        optimiser_2.step()

        if done:
            break

        c_state = n_state


    av_return.append(t)

    if ep %100 == 0:

        print("ep: ",ep," av_return: ", sum(av_return)/100)

        av_return = []









