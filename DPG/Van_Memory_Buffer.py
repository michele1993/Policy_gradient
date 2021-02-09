import random


class V_Memory_B:

    def __init__(self,batch_size = 30,size = 50000):

        self.size = size
        self.batch_size = batch_size

        self.buffer = []
        self.c_size = 0


    def store_transition(self,c_state,action,rwd,n_s,dn):


        # Check if the replay buffer is full
        if len(self.buffer) <= self.size:

            self.buffer.append((c_state,action,rwd,n_s,dn))

        # if full, start replacing values from the first element
        else:

            self.buffer[self.c_size] = (c_state,action,rwd,n_s,dn)
            self.c_size+=1

            # Need to restart indx when reach end of list
            if self.c_size == self.size:
                self.c_size = 0



    def sample_transition(self):

        spl_transitions = random.sample(self.buffer, self.batch_size)
        return zip(*spl_transitions)  # work correctly