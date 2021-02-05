import torch
import torch.optim as opt
from torch.distributions import Categorical
import numpy as np


class PPO:


    def __init__(self, actor, critic, n_batches=1, PPO_epochs=1, ln_rate=1e-3, clipping_val=0.2, critic_discount= 0.5):


        self.clipping_val = clipping_val
        self.n_batches = n_batches
        self.PPO_epochs = PPO_epochs
        self.critic_discount = critic_discount

        self.actor = actor
        self.critic = critic

        parameters = list(self.actor.parameters()) + list(self.critic.parameters())

        self.optimiser = opt.Adam(parameters, ln_rate)


    def PPO(self, states, actions, log_probs, rewards, advantages):

        #KEY:  need to detach() old_log_p otherwise backpro through them

        actor_cost = 0
        critic_cost = 0

        for _ in range(self.PPO_epochs):

            for state, action, old_log_p, return_, adv in self.ppo_iter(states, actions, log_probs, rewards, advantages):

                new_mean = self.actor(state)
                d = Categorical(new_mean)
                new_log_p = d.log_prob(action)

                ratio = (new_log_p - old_log_p).exp()

                surrogate_l1 = ratio * adv
                surrogate_l2 = torch.clamp(ratio, 1.0 - self.clipping_val, 1.0 + self.clipping_val) * adv

                actor_loss = - torch.min(surrogate_l1,surrogate_l2).mean()

                v_value = self.critic(state)

                critic_loss =  ((return_ - v_value)**2).mean()

                total_loss = actor_loss + self.critic_discount * critic_loss # add a critic discount and entropy?

                self.optimiser.zero_grad()

                total_loss.backward()

                self.optimiser.step()

                with torch.no_grad():

                    actor_cost += actor_loss
                    critic_cost += critic_loss

        return actor_cost, critic_cost


    def ppo_iter(self, states, actions, log_probs, returns, advantages):

        trajectory_size = states.size(0)

        for _ in range(trajectory_size // self.n_batches):

            rnd_idx = np.random.randint(0,trajectory_size,self.n_batches)
            # yield returns the current values at the given interation and then continue from the next iteration until the end of the loop
            yield states[rnd_idx], actions[rnd_idx], log_probs[rnd_idx], returns[rnd_idx], advantages[rnd_idx] #[rnd_idx, :]