import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np



class QMixer_CNN(nn.Module):
    def __init__(self, state_dim):
        super(QMixer_CNN, self).__init__()

        self.n_agents = 3
        # self.state_dim = int(np.prod(args.state_shape))

        self.embed_dim = 128
        kernel_size =3
        strides =2
        if state_dim == (6,20,20):
            self.embedding_network = nn.Sequential(nn.Conv2d(6,16,4,stride=2),
                        nn.ReLU(),
                        nn.Conv2d(16,32,3,stride=1),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32,64,2,stride=1),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(256,self.embed_dim),
                        nn.ReLU(),
                        )
        elif state_dim == (6,30,30):
            self.embedding_network = nn.Sequential(nn.Conv2d(6,16,5,stride=2),
                        nn.ReLU(),
                        nn.Conv2d(16,32,5,stride=2,padding=2),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32,64,2,stride=1),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(256,self.embed_dim),
                        nn.ReLU(),
                        )

        if False:
            self.hyper_w_1 = nn.Linear(self.embed_dim, self.embed_dim * self.n_agents)
            self.hyper_w_final = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            hypernet_embed = 128
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.embed_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.embed_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.embed_dim, self.embed_dim)

        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def forward(self, agent_qs, states):
        bs = agent_qs.size(0)
        state_embedding = self.embedding_network(states)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)
        # First layer
        w1 = th.abs(self.hyper_w_1(state_embedding))
        b1 = self.hyper_b_1(state_embedding)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        hidden = F.elu(th.bmm(agent_qs, w1) + b1)
        # Second layer
        w_final = th.abs(self.hyper_w_final(state_embedding))
        w_final = w_final.view(-1, self.embed_dim, 1)
        # State-dependent bias
        v = self.V(state_embedding).view(-1, 1, 1)
        # Compute final output
        y = th.bmm(hidden, w_final) + v
        # Reshape and return
        q_tot = y.view(bs, -1, 1)
        return q_tot

class CRNNAgent(nn.Module):
    def __init__(self, input_shape):
        super(CRNNAgent, self).__init__()
        # self.args = args
        strides = 2
        kernel_size = 6
        self.embedding_dim = 128
        self.rnn_hidden_dim = 128
        self.embed_dim = 128
        self.n_actions = 5
        if input_shape ==(6,39,39):
            self.embedding_network = nn.Sequential(nn.Conv2d(6,16,5,stride=2,padding=1),
                        nn.ReLU(),
                        nn.Conv2d(16,32,5,stride=2,padding=2),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32,32,2,stride=1),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(512,self.embed_dim),
                        nn.ReLU(),
                        )
        elif input_shape == (6,59,59):
            self.embedding_network = nn.Sequential(nn.Conv2d(6,16,5,stride=2,padding=1),
                        nn.ReLU(),
                        nn.Conv2d(16,32,5,stride=2,padding=2),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32,32,2,stride=1),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(1152,self.embed_dim),
                        nn.ReLU(),
                        )

        self.fc1 = nn.Linear(self.embedding_dim, self.rnn_hidden_dim)
        self.rnn = nn.GRUCell(self.rnn_hidden_dim, self.rnn_hidden_dim)
        self.fc2 = nn.Linear(self.rnn_hidden_dim, self.n_actions)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = self.embedding_network(inputs)
        x = F.relu(self.fc1(x))
        h_in = hidden_state.reshape(-1, self.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h


if __name__ == "__main__":
    state_dim = (6,30,30)
    testNetwork = QMixer_CNN(state_dim)
    testInput = th.rand(1,6,30,30)
    agentQs = th.rand(1,1,3)
    testNetwork(agentQs,testInput)

    state_dim = (6,20,20)
    testNetwork = QMixer_CNN(state_dim)
    testInput = th.rand(1,6,20,20)
    agentQs = th.rand(1,1,3)
    testNetwork(agentQs,testInput)

    state_dim = (6,39,39)
    testNetwork = CRNNAgent(state_dim)
    testInput = th.rand(1,6,39,39)
    hidden_state = th.rand(1,128)
    testNetwork(testInput,hidden_state)
    state_dim = (6,59,59)
    testNetwork = CRNNAgent(state_dim)
    testInput = th.rand(1,6,59,59)
    hidden_state = th.rand(1,128)
    testNetwork(testInput,hidden_state)
