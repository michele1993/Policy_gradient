import torch
import gym
import torch.optim as opt

from torch.distributions import Categorical

from AC_algorithm.Actor_NN import Actor_net
from AC_algorithm.Critic_NN import Critic_NN


env = gym.make("CartPole-v1")

n_episodes = 100000
max_t_steps = 200
discount= 0.99
ln_rate = 1e-3
#ln_rate_1 = 1e-3
#ln_rate_2 = 1e-4

actor = Actor_net(discount = discount).double()
critic_1 = Critic_NN(discount = discount).double()
critic_2 = Critic_NN(discount = discount,n_inputs = 5).double()

parameters = list(actor.parameters()) + list(critic_1.parameters()) + list(critic_2.parameters())

optimiser = opt.Adam(parameters,ln_rate)
#optimiser_1 = opt.Adam(actor.parameters(),ln_rate_1)
#optimiser_2 = opt.Adam(critic.parameters(),ln_rate_2 )

av_return = []

ac_cost=[]
cr_cost = []


for ep in range(n_episodes):

    c_state = env.reset()

    t = 0
    t_a_cost = 0
    t_c_cost = 0

    for t in range(max_t_steps):

        mean_action = actor(torch.tensor(c_state))

        d = Categorical(mean_action)

        action = d.sample()

        lp_action = d.log_prob(action)

        n_state,rwd,done,_ = env.step(action.numpy())




        Q_value = critic_2(torch.cat([torch.from_numpy(c_state).view(-1),action.view(-1).double()]))

        V_value = critic_1(torch.from_numpy(c_state))

        advantage = rwd + Q_value - V_value

        #advantage , critic_cost = critic.advantage(c_state,n_state,rwd,done) #

        #print(advantage)

        rf_cost = actor.REINFORCE(lp_action,advantage,done)

        critic_cost = advantage**2

        loss = rf_cost + critic_cost

        #optimiser_1.zero_grad()
        #optimiser_2.zero_grad()

        optimiser.zero_grad()

        loss.backward()

        #optimiser_1.step()
        #optimiser_2.step()
        optimiser.step()

        with torch.no_grad():
            t_a_cost += rf_cost
            t_c_cost += critic_cost

        if done:
            break

        c_state = n_state


    ac_cost.append(t_a_cost/t)
    cr_cost.append(t_c_cost/t)

    av_return.append(t)

    if ep %300 == 0:

        print("critic cost: ", sum(cr_cost) / 300)
        print("actor cost: ", sum(ac_cost)/300)
        print("av_return: ", sum(av_return)/300)
        ac_cost = []
        cr_cost = []

        av_return = []









