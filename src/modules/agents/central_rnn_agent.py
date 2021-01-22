import torch as th
import torch.nn as nn
import torch.nn.functional as F

class CentralRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(CentralRNNAgent, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions * args.central_action_embed)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        q = q.reshape(-1, self.args.n_actions, self.args.central_action_embed)
        return q, h


class ConvCentralRNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(ConvCentralRNNAgent, self).__init__()
        self.args = args
        self.embedding_dim = self.args.embed_dim
        if input_shape == [6,39,39]:
            self.embedding_network = nn.Sequential(nn.Conv2d(6,16,5,stride=2,padding=1),
                        nn.ReLU(),
                        nn.Conv2d(16,32,5,stride=2,padding=2),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32,32,2,stride=1),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(512,self.embedding_dim),
                        nn.ReLU(),
                        )
        elif input_shape == [6,59,59]:
            self.embedding_network = nn.Sequential(nn.Conv2d(6,16,5,stride=2,padding=1),
                        nn.ReLU(),
                        nn.Conv2d(16,32,5,stride=2,padding=2),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32,32,2,stride=1),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(1152,self.embedding_dim),
                        nn.ReLU(),
                        )
        elif input_shape == [6,19,19]:
            self.embedding_network = nn.Sequential(nn.Conv2d(6,16,5,stride=2),
                        nn.ReLU(),
                        nn.Conv2d(16,32,3,stride=1),
                        nn.MaxPool2d(2),
                        nn.Conv2d(32,64,2,stride=1),
                        nn.ReLU(),
                        nn.Flatten(),
                        nn.Linear(256,self.embedding_dim),
                        nn.ReLU(),
                        )
        else:
            raise Exception("Invalid Input Size for Convolutions: {}".format(input_shape))
        self.fc1 = nn.Linear(self.embedding_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim,  args.n_actions * args.central_action_embed)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        state_embedding = self.embedding_network(inputs)
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(state_embedding, h_in)
        q = self.fc2(h)
        q = q.reshape(-1, self.args.n_actions, self.args.central_action_embed)
        return q, h
