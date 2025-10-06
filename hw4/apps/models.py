import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)

def ConvBN(a, b, k, s, device):
    return nn.Sequential(nn.Conv(a, b, k, s, device=device), nn.BatchNorm2d(b, device=device), nn.ReLU())

class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.device = device
        self.b1 = ConvBN(3,16,7,4, device = device)
        self.b2 = ConvBN(16,32,3,2, device = device)
        self.b3 = ConvBN(32,32,3,1, device = device)
        self.b4 = ConvBN(32,32,3,1, device = device)
        self.b5 = ConvBN(32,64,3,2, device = device)
        self.b6 = ConvBN(64,128,3,2, device = device)
        self.b7 = ConvBN(128,128,3,1, device = device)
        self.b8 = ConvBN(128,128,3,1, device = device)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(128,128, device = device)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128,10, device = device)
        

    def forward(self, x):
        conv2 = self.b2(self.b1(ndl.Tensor(x, device=self.device)))
        conv4 = self.b4(self.b3(conv2)) + conv2
        conv6 = self.b6(self.b5(conv4))
        conv8 = self.b8(self.b7(conv6)) + conv6
        return self.linear2(self.relu(self.linear1(self.flatten(conv8))))


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', seq_len=40, device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.device = device
        self.model_name = seq_model
        model_list = {
            'rnn': nn.RNN,
            'lstm': nn.LSTM,
            'transformer': nn.Transformer
        }
        self.embedding_layer = ndl.nn.Embedding(output_size, embedding_size, device=device)
        self.seq_model = model_list[seq_model](embedding_size, hidden_size, num_layers, device=device)
        self.linear = nn.Linear(embedding_size if seq_model == 'transformer' else hidden_size, output_size, device=device)        

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        seq_len, bs = x.shape
        layers = self.seq_model.layers
        if h is None and self.model_name != 'transformer':
            if self.model_name == 'rnn':
                h = ndl.init.zeros(layers, bs, self.hidden_size, device=self.device)
            else:
                h = (ndl.init.zeros(layers, bs, self.hidden_size, device=self.device), ndl.init.zeros(layers, bs, self.hidden_size, device=self.device))
        seq_h, h_final = self.seq_model(self.embedding_layer(x), h)
        hidden_size = seq_h.shape[2]
        return self.linear(seq_h.reshape((seq_len*bs, hidden_size))).reshape((seq_len*bs, self.output_size)), h_final


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(cifar10_train_dataset[1][0].shape)
