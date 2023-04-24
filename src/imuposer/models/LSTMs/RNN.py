import torch.nn as nn
from torch.nn.functional import relu

class RNN(nn.Module):
    r"""
    An RNN Module including a linear input layer, an RNN, and a linear output layer.
    """
    def __init__(self, n_input, n_output, n_hidden, n_rnn_layer=2, bidirectional=True, dropout=0.2):
        super(RNN, self).__init__()
        self.rnn = nn.LSTM(n_hidden, n_hidden, n_rnn_layer, bidirectional=bidirectional, batch_first=True)
        self.linear1 = nn.Linear(n_input, n_hidden)
        self.linear2 = nn.Linear(n_hidden * (2 if bidirectional else 1), n_output)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_lens=None, h=None):
        r"""
        Gets a padded batch
        Step 1: pass through linear layer
        Step 2: pack the padded sequences
        Step 3: pass the packed sequences through the lstm layers
        Step 4: unpack the packed sequences
        Step 5: pass the unpacked sequences (padded) to the linear layers
        Step 6: return the output and output lengths
        """
        # Step 1
        x = relu(self.linear1(self.dropout(x)))

        # Step 2 
        x = nn.utils.rnn.pack_padded_sequence(x, x_lens, batch_first=True, enforce_sorted=False)

        # Step 3
        x, h = self.rnn(x, h)

        # Step 4
        x, output_lens = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # Step 5 
        x = self.linear2(x)

        # how to unpad
        # outputs = [output[:output_lens[i]] for i, output in enumerate(outputs)] 

        return x, output_lens, h
