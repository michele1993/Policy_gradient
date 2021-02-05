import gym
import torch
import torch.nn
from torch.distributions import Categorical

from Policy_gradient.PPO_GAE.Actor_NN import Actor_NN
from Policy_gradient.PPO_GAE.GAE_critic import Critic_NN
from Policy_gradient.PPO_GAE.PPO_alg import PPO





n_episodes= 10000
PPO_steps = 200
std = 0.3
gamma = 0.99
ln_rate_c = 0.001
ln_rate_a = 0.0001
t_printing = 100


env = gym.make("CartPole-v1")




model_actor = Actor_NN(input_size =env.observation_space.shape[0], ouput_size= 2)#.double() #env.action_space.shape[0]
model_critc = Critic_NN(discount= gamma,input_size =env.observation_space.shape[0])#.double()

ppo = PPO(model_actor,model_critc)



accuracy = []

actor_cost = 0
critic_cost = 0

for ep in range(n_episodes):

    states = []

    actions = []

    #values = torch.empty(0, dtype=torch.double)
    values = []

    #rewards = torch.empty(0)
    rewards = []

    #log_ps = torch.empty(0,dtype=torch.double)
    log_ps = []

    #masks = torch.empty(0, dtype= bool) # in case game finishes before end of PPO steps
    masks = []


    c_state = env.reset()

    itr = 0

    for itr in range(PPO_steps):

        c_state = torch.FloatTensor(c_state)
        action_mean = model_actor(c_state)
        v_value = model_critc(c_state)


        #d = Normal(action_mean,std)
        d = Categorical(action_mean)
        action = d.sample()
        log_p = d.log_prob(action)


        obs, rwd, done, _ = env.step(action.numpy())
        mask = not done

        states.append(c_state)
        actions.append(action.detach())

        #values = torch.cat([values, v_value])
        values.append(v_value.detach())

        #rewards = torch.cat([rewards,torch.unsqueeze(torch.tensor(rwd), dim=-1)])
        rewards.append(torch.tensor(rwd))

        #log_ps = torch.cat([log_ps, torch.unsqueeze(log_p, dim=-1)])
        log_ps.append(log_p.detach())

        #masks = torch.cat([masks, torch.unsqueeze(torch.tensor(mask), dim=-1)])
        masks.append(torch.tensor(mask))

        c_state = obs

        if done:
            break


    accuracy.append(itr)

    # get prediction for last state, since exited the loop before computing it
    last_v_value = model_critc(torch.FloatTensor(c_state)).detach()
    #values = torch.cat([values, last_v_value])
    #values.append(last_v_value.detach())

    returns = model_critc.GAE(values,last_v_value ,rewards, masks)

    returns = torch.stack(returns,dim=0)
    log_ps = torch.stack(log_ps,dim=0)
    values = torch.stack(values,dim=0)
    rewards = torch.stack(rewards,dim=0)
    states = torch.stack(states,dim=0)
    actions = torch.stack(actions,dim=0)
    masks = torch.stack(masks,dim=0)
    advantages = returns - values
    advantages = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1e-10)

    # calculate advantages from returns

    actor_c, critic_c = ppo.PPO(states, actions, log_ps, returns, advantages)

    actor_cost+= actor_c
    critic_cost+= critic_c



    if ep % t_printing == 0:

        print("EP: ", ep, " accuracy: ", sum(accuracy)/t_printing, " critic: ", critic_cost/t_printing, " actor: ", actor_cost/t_printing)

        actor_cost = 0
        critic_cost = 0
        accuracy = []
