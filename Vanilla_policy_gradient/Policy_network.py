import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np



class Policy_net(nn.Module):

    def __init__(self,Input_size=4, Hidden_size=56, output_size=2):

        super(Policy_net,self).__init__()

        self.l1 = nn.Linear(Input_size, Hidden_size)
        self.l2 = nn.Linear(Hidden_size,output_size)



    def forward(self,x):

        x = F.relu(self.l1(x))
        x = torch.sigmoid(self.l2(x)) #F.softmax

        return x #.view(-1,1)

    def REINFORCE(self, p_action, advantage):


        Policy_cost = - p_action * advantage.detach() # need minus because need to perform gradient ascent

        #print(Policy_cost.grad_fn)


        return Policy_cost


    def compute_returns(self, rwds, discounts):


        discounts = (discounts**(torch.FloatTensor(range(len(rwds)))))



        return torch.flip(torch.cumsum(torch.flip(discounts * rwds, dims=(0,)), dim=0), dims=(0,)) / discounts
