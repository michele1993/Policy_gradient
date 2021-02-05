import gym
import torch
import random


from Policy_gradient.DPG.DPG_Actor_NN import Actor_NN
from Policy_gradient.DPG.DPG_Critic_NN import Critic_NN

n_episodes = 10000
Buffer_size = 10000
start_update = 500
batch_size = 30
max_t_steps = 200
discount= 0.99
ln_rate_c = 0.001
ln_rate_a = 0.0001
ep_print = 50
decay_upd = 0.1 # decay for polyak average

env = gym.make("Pendulum-v0")

#Initialise actor
agent = Actor_NN().double()
target_agent = Actor_NN().double()

target_agent.load_state_dict(agent.state_dict())
target_agent.freeze_params()

# Initialise two critic NN one to be the fixed target and the other to be trained for stability
critic_target = Critic_NN().double()
critic_nn = Critic_NN().double()

# Make sure two critic NN have the same initial parameters
critic_target.load_state_dict(critic_nn.state_dict())

#Freeze the critic target NN parameter
critic_target.freeze_params()


#Initialise the replay buffer
rpl_buffer = []
eps_acc = []


buffer_c_size = 0

for ep in range(n_episodes):

    c_state = env.reset()

    for t in range(max_t_steps):

        # Perform initial random exploration
        if ep > start_update:
            with torch.no_grad():
                det_action = agent(torch.from_numpy(c_state))
                stocasticity = torch.randn(1) * 0.5
                action = (det_action + stocasticity)


        else:

            action = (torch.randn(1,dtype = torch.double) * 2)


        n_state, rwd, done, _ = env.step(action.numpy())


        # Check if the replay buffer is full
        if len(rpl_buffer) <= Buffer_size:

            rpl_buffer.append((c_state,action, rwd, n_state,done))

        # if full, start replacing values from the first element
        else:

            rpl_buffer[buffer_c_size] = (c_state,action, rwd, n_state,done)
            buffer_c_size+=1

            # Need to restart indx when reach end of list
            if buffer_c_size == Buffer_size:

                buffer_c_size = 0


        c_state = n_state


        # Check if it's time to update
        if ep > start_update:

            # Randomly sample batch of transitions from buffer
            spl_transitions = random.sample(rpl_buffer,batch_size)
            b_spl_c_state, b_spl_a, b_spl_rwd, b_spl_n_state, b_spl_done = zip(*spl_transitions)

            # convert everything to tensor
            b_spl_c_state = torch.tensor(b_spl_c_state)
            b_spl_rwd = torch.tensor(b_spl_rwd)
            b_spl_n_state = torch.tensor(b_spl_n_state)
            b_spl_done = torch.tensor(b_spl_done)

            # Create input for target critic, based on next state and the optimal action there
            trg_crit_inpt = torch.cat([b_spl_n_state, target_agent(b_spl_n_state)],dim=1)#
            # Compute Q target value
            Q_target = b_spl_rwd + discount * (~b_spl_done) * critic_target(trg_crit_inpt).squeeze() # squeeze so that all dim in equation match for element-wise operations

            # Compute Q estimate
            b_spl_a = torch.stack(b_spl_a) # need to increase dim to have same size as states
            critic_nn_inpt = torch.cat([b_spl_c_state,b_spl_a],dim=1)
            Q_estimate = critic_nn(critic_nn_inpt).squeeze() # squeeze to have same dim as Q_target

            # Update critic
            critic_nn.update(Q_target, Q_estimate)

            # Update actor
            actor_loss_inpt = torch.cat([b_spl_c_state,agent(b_spl_c_state)],dim=1)
            actor_loss = - critic_nn(actor_loss_inpt)
            agent.update(actor_loss)


            # Update target NN through polyak average
            target_agent.soft_update(agent, decay_upd)
            critic_target.soft_update(critic_nn,decay_upd)


    eps_acc.append(rwd)

    if ep % ep_print == 0:

        print("ep: ", ep)
        print("final aver accuracy: ",sum(eps_acc) / ep_print)
        eps_acc = []



# b = [i.clone() for i in critic_target.parameters()]
# c = [i.clone() for i in critic_target.parameters()]
# d = [torch.eq(e, f) for e, f in zip(c, b)]
# print(d, "\n")














