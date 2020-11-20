import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.gru = nn.GRU(self.input_size, self.hidden_size, batch_first=True)
        
    def forward(self, x):
        output, hidden = self.gru(x)
        return output, hidden


# https://github.com/pytorch/pytorch/issues/805 erogol
class Maxout(nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size()) # 1 B 3H+m
        shape[-1] = self.d_out # 1 B maxoutsize
        shape.append(self.pool_size) # 1 B maxoutsize 2
        max_dim = len(shape) - 1 # 3
        out = self.lin(inputs) # 1 B 2maxout
        m, i = out.view(*shape).max(max_dim) # 1 B maxout
        return m    

# https://github.com/keon/seq2seq/blob/master/model.py
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 3, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0) # Hidden: B H
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1) # B T H
        encoder_outputs = encoder_outputs.transpose(0, 1)  # B*T*2H
        attn_energies = self.score(h, encoder_outputs) # B T
        return F.softmax(attn_energies, dim=1).unsqueeze(1) # B 1 T

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embed_size, maxout_size):
        # embed_size => word embedding size => input size & output size 
        super(DecoderRNN, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = embed_size
        self.maxout_size = maxout_size
        
        # self.embed = nn.Embedding(output_size, embed_size)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size * 2 + embed_size , hidden_size)
        self.maxout = Maxout(hidden_size * 3 + embed_size, maxout_size, 2)
        self.out = nn.Linear(maxout_size, self.output_size)
        
    def forward_step(self, input, last_hidden, encoder_outputs):
        # input should be the last translated word => shape [B, embd_size]
        # encoder_outputs should be of shape [B, T, 2H]
        # last_hidden is the last hidden state of encoder GRU
        # Get the embedding of the current input word (last output word)
        encoder_outputs = encoder_outputs.transpose(0, 1) #B T 2H-> T B 2H
        embedded = input # self.embed(input)  # B m
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs) # B 1 T
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,2H)
        context = context.transpose(0, 1)  # (1,B,2H)
        rnn_input = torch.cat([embedded, context], 2) # 1 B 2H+m
        out, hidden = self.gru(rnn_input, last_hidden) # 1 B H
        maxout_input = torch.cat([last_hidden, embedded, context], 2) # 1 B 3H+m
        output = self.maxout(maxout_input).squeeze(0) # B maxout   
        output = self.out(output) # B output
        return output, hidden, attn_weights # B output / 1 B H / B 1 T
    
    def forward(self, decoder_hidden, encoder_outputs, max_length=22):
        # decoder_hidden: the init hidden state for decoder
        batch_size = encoder_outputs.size(0)
        self_param = next(self.parameters())    # get module parameters, in order to find the device
        decoder_input = torch.zeros([batch_size, self.embed_size]).to(self_param) # B * embd_size
        decoder_outputs = []
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = \
                self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            # decoder_output => [B * H]
            decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder, dim=1) # [B, T, H]
        return decoder_outputs

class RNNseq2seq(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(RNNseq2seq, self).__init__()
        self.encoder = EncoderRNN(input_dim, hidden_size)
        self.decoder = DecoderRNN(hidden_size, output_dim, hidden_size)
    
    def forward(self, x):
        encoder_outputs, encoder_hidden = self.encoder(x)
        decoder_outputs = self.decoder(encoder_hidden, encoder_outputs, max_length=encoder_outputs.size(1))
        return encoder_outputs, decoder_outputs


if __name__ == '__main__':
    a = RNNseq2seq(300, 128, 130)
    x = torch.rand(10, 22, 300)
    out, hidden = a(x)
    print(out.shape)
    print(hidden.shape)