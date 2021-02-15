import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt


class Critic_NN(nn.Module):


    def __init__(self,state_size = 3,action_size =1, hidden_s =64, Hidden_size = 256, Output_size = 1,ln_rate = 1e-3):

        super().__init__()

        self.l0s = nn.Linear(state_size,hidden_s)
        self.l0a = nn.Linear(action_size, hidden_s)

        self.l1 = nn.Linear(hidden_s+hidden_s,Hidden_size)
        self.l2 = nn.Linear(Hidden_size, Output_size)

        self.optimiser = opt.Adam(self.parameters(),ln_rate)

    def forward(self, s,a):

        # send state and actions through one layer separately
        s = F.relu(self.l0s(s))
        a = F.relu(self.l0a(a))

        x = torch.cat([s,a],dim=1)
        x = F.relu(self.l1(x))
        x = self.l2(x)

        return x


    def freeze_params(self):

        for params in self.parameters():

            params.requires_grad = False


    def update(self, target, estimate):

        loss = torch.mean((target - estimate)**2)
        self.optimiser.zero_grad()
        loss.backward() #needed for the actor
        self.optimiser.step()

        return loss

    def copy_weights(self,estimate):

        for t_param, e_param in zip(self.parameters(),estimate.parameters()):
            t_param.data.copy_(e_param.data)

    def soft_update(self, estimate, decay):

        with torch.no_grad():
          # do polyak averaging to update target NN weights
            for t_param, e_param in zip(self.parameters(),estimate.parameters()):
                t_param.data.copy_( e_param.data * decay + (1 - decay) * t_param.data)
