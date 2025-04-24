import torch.utils.data
from torch import nn, FloatTensor


"""Angle"""
class LSTM(torch.nn.Module):

    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer, batch_size, bidirectional=False, dropout=0.2):
        super(LSTM, self).__init__()
        # initialize the configs
        self.n_hidden = n_hidden
        self.n_rnn_layer = n_rnn_layer
        self.batch_size = batch_size
        self.num_directions = 2 if bidirectional else 1
        self.state = None
        self.feature_dim = 32
        self.encoder = nn.Sequential(nn.Linear(n_input, self.feature_dim),
            nn.ReLU(),
        )
        self.gru_encoder = nn.GRU(self.feature_dim, self.feature_dim, 1, bidirectional=bidirectional, batch_first=True)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=n_input, num_heads=1, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=n_input, out_channels=self.feature_dim, kernel_size=1)
        self.rnn = torch.nn.LSTM(self.feature_dim, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=True) # 16->n_input


        self.batch_norm = nn.BatchNorm1d(n_hidden * (2 if bidirectional else 1))

        self.layer_norm = nn.LayerNorm(64)

        self.linear_angle_1 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), 64)
        self.linear_angle_3 = torch.nn.Linear(64, 64)
        self.linear_angle_2 = torch.nn.Linear(64, n_output)


    def forward(self, X):


        X = self.encoder(X)
        Y, self.state = self.rnn(X, self.state)

        last_step_Y = Y[:,-1, :]
        last_step_Y = last_step_Y.view(last_step_Y.shape[0],-1)
        angle = self.linear_angle_1(last_step_Y)

        angle = nn.ReLU()(angle)

        angle = self.linear_angle_2(angle)

        return angle

    def begin_state(self, init_method='zero', device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        # Initialize hidden states H and C
        if not isinstance(self.rnn, nn.LSTM):
            self.state = torch.zeros((self.num_directions * self.n_rnn_layer,
                                      self.batch_size, self.n_hidden),
                                     device=device, requires_grad=True)
        else:
            if init_method == 'zero':
                self.state = (torch.zeros((
                    self.num_directions * self.n_rnn_layer,
                    self.batch_size, self.n_hidden), device=device),
                              torch.zeros((
                                  self.num_directions * self.n_rnn_layer,
                                  self.batch_size, self.n_hidden), device=device))
            elif init_method == 'normal':
                self.state = (torch.randn((
                    self.num_directions * self.n_rnn_layer,
                    self.batch_size, self.n_hidden), device=device),
                              torch.randn((
                                  self.num_directions * self.n_rnn_layer,
                                  self.batch_size, self.n_hidden), device=device))
            else:
                raise NotImplementedError




class LSTMF(torch.nn.Module):

    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer, batch_size, bidirectional=False, dropout=0.2):
        super(LSTMF, self).__init__()
        # initialize the configs
        self.n_hidden = n_hidden
        self.n_rnn_layer = n_rnn_layer
        self.batch_size = batch_size
        self.num_directions = 2 if bidirectional else 1
        self.state = None
        self.feature_dim = 32
        self.encoder = nn.Sequential(nn.Linear(n_input, self.feature_dim),
            nn.ReLU(),
        )
        self.gru_encoder = nn.GRU(self.feature_dim, self.feature_dim, 1, bidirectional=bidirectional, batch_first=True)

        self.multihead_attn = nn.MultiheadAttention(embed_dim=n_input, num_heads=1, batch_first=True)
        self.conv1 = nn.Conv1d(in_channels=n_input, out_channels=self.feature_dim, kernel_size=1)
        self.rnn = torch.nn.LSTM(self.feature_dim, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=True) # 16->n_input


        self.batch_norm = nn.BatchNorm1d(n_hidden * (2 if bidirectional else 1))

        self.layer_norm = nn.LayerNorm(64)

        self.linear_angle_1 = torch.nn.Linear(n_hidden * (2 if bidirectional else 1), 64)
        self.linear_angle_3 = torch.nn.Linear(64, 64)
        self.linear_angle_2 = torch.nn.Linear(64, n_output)


    def forward(self, X):


        X = self.encoder(X)


        Y, self.state = self.rnn(X, self.state)
        last_step_Y = Y[:,-1, :].squeeze(dim=1)

        angle = self.linear_angle_1(last_step_Y)
        angle = nn.ReLU()(angle)
        angle = self.linear_angle_2(angle)
        return angle

    def begin_state(self, init_method='zero', device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        # Initialize hidden states H and C
        if not isinstance(self.rnn, nn.LSTM):
            self.state = torch.zeros((self.num_directions * self.n_rnn_layer,
                                      self.batch_size, self.n_hidden),
                                     device=device, requires_grad=True)
        else:
            if init_method == 'zero':
                self.state = (torch.zeros((
                    self.num_directions * self.n_rnn_layer,
                    self.batch_size, self.n_hidden), device=device),
                              torch.zeros((
                                  self.num_directions * self.n_rnn_layer,
                                  self.batch_size, self.n_hidden), device=device))
            elif init_method == 'normal':
                self.state = (torch.randn((
                    self.num_directions * self.n_rnn_layer,
                    self.batch_size, self.n_hidden), device=device),
                              torch.randn((
                                  self.num_directions * self.n_rnn_layer,
                                  self.batch_size, self.n_hidden), device=device))
            else:
                raise NotImplementedError
