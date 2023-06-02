import torch
import torch.nn as nn
from .config import *


class Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.bidirectional_lstm = nn.LSTM(
            input_size = D_BERT,
            hidden_size = D_LSTM // 2,
            num_layers = 1,
            bias = True,
            batch_first = True,
            bidirectional = True,
        )
        X = 10
        self.fc1 = nn.Linear(D_LSTM * 2, X, bias = False)
        self.fc2 = nn.Linear(X, 1, bias = False)
        self.fc3 = nn.Linear(D_LSTM * 2, D_LSTM)

    def attention(self, x, Time_step, mask = None):

        if (mask is None): raise ValueError()

        # query: h_i (BATCH_SIZE * Time_step * Time_step, D_LSTM)
        query = x.reshape((-1, Time_step, 1, D_LSTM))
        query = torch.tile(query, dims=(1, 1, Time_step, 1))
        # key: h_j (BATCH_SIZE * Time_step * Time_step, D_LSTM)
        key = x.reshape((-1, 1, Time_step, D_LSTM))
        key = torch.tile(key, dims=(1, Time_step, 1, 1))
        # value (BATCH_SIZE, Time_step, D_LSTM)
        value = x
        # mask (BATCH_SIZE, Time_step, Time_step)
        mask = torch.where(mask == 0, -torch.inf, 0.0)
        mask = mask.reshape(x.shape[0], 1, Time_step)
        mask = torch.tile(mask, dims=(1, Time_step, 1))

        # additive attention: FC(tanh(FC([Q;K]))
        # query(h_i) & key(h_j)
        a = self.fc1(torch.cat((query, key), dim=-1))
        a = nn.Tanh()(a)
        a = self.fc2(a)
        a = torch.squeeze(a, dim=-1)
        attention_weight = nn.Softmax(dim=-1)(a + mask)
        context = torch.bmm(attention_weight, value)
        y = self.fc3(torch.cat((value, context), dim=-1))
        return y
    
    def forward(self, x, Time_step, mask = None, use_hidden_state = False):
        hidden_state, _ = self.bidirectional_lstm(x)
        y = self.attention(hidden_state, Time_step, mask = mask)
        if (use_hidden_state):
            return y, hidden_state
        else:
            return y


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm_cell = nn.LSTMCell(
            input_size = D_LSTM + N_CLASS,
            hidden_size = D_LSTM,
            bias = True,
        )
        self.linear = nn.Linear(D_LSTM, N_CLASS)
    
    def forward(self, x, Time_step, mask = None, init_state = None):
        (init_h, _) = init_state
        h = init_h
        c = torch.zeros_like(h)
        y = torch.zeros((x.shape[0], N_CLASS))
        ys = []
        for t in range(Time_step):
            (h, c) = self.lstm_cell(torch.cat((x[:, t, :], y), dim=-1), (h, c))
            y = nn.Softmax(dim=-1)(self.linear(h))
            ys.append(y)
        ys = torch.stack(ys, dim=0).transpose(1, 0)
        return ys


class ValueDetector(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def forward(self, x, mask = None):
        # Batch=len(x)
        Time_step=len(x[0])
        y_enc, hidden_state = self.encoder(x, Time_step, mask = mask, use_hidden_state = True)
        init_h = hidden_state[:, -1, :] # (B, Time_step, D)
        init_c = None
        y = self.decoder(y_enc, Time_step , mask = mask, init_state = (init_h, init_c))
        return y
