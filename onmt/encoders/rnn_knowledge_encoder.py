"""Define RNN-based encoders."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory


class KnowledgeRNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(KnowledgeRNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.embeddings = embeddings

        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.bridge)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        # filter the cue whose length == 0 and remember it to restore the shape with zero Tensor

        # need to resort the knowledge src and new lengths.
        new_lengths, indices = lengths.sort(descending=True)
        new_src = src.index_select(1,indices)

        # remove the zero length-sized
        valid = new_lengths.ne(0).sum().item()

        input_lengths = new_lengths[:valid]
        input_src = new_src[:, :valid, :]

        emb = self.embeddings(input_src)
        # s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if input_lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = input_lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)

        memory_bank, encoder_final = self.rnn(packed_emb)

        if input_lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]

        if self.use_bridge:
            encoder_final = self._bridge(encoder_final)

        _, rev_indices = indices.sort()

        # padding zero tensor and restore
        if isinstance(encoder_final, tuple):
            assert encoder_final[0].dim() == 3, "dim error for the rnn knowledge encoder"
            # encoder_final = (self._bridge_bidirectional_hidden(encoder_final[0]),
            #                       self._bridge_bidirectional_hidden(encoder_final[1]))
            zero_padding = torch.zeros(encoder_final[0].size(0), lengths.size(0)-valid, encoder_final[0].size(-1)).type_as(lengths).float()
            forward_direction = torch.cat((encoder_final[0], zero_padding), dim=1)
            backward_direction = torch.cat((encoder_final[1], zero_padding), dim=1)
            encoder_final = (forward_direction.index_select(dim=1, index=rev_indices),
                                backward_direction.index_select(dim=1, index=rev_indices))
        else:
            assert encoder_final.dim() == 3, "dim error for the rnn knowledge encoder"
            # encoder_final = self._bridge_bidirectional_hidden(encoder_final)    # num_layers * batch_size * hidden_size
            zero_padding = torch.zeros(encoder_final.size(0), lengths.size(0)-valid, encoder_final.size(-1)).type_as(lengths).float()
            encoder_final = torch.cat((encoder_final, zero_padding), dim=1)
            encoder_final = encoder_final.index_select(dim=1, index=rev_indices)

        memory_bank_zero_padding = torch.zeros((memory_bank.size(0), lengths.size(0), memory_bank.size(-1))).type_as(lengths).float()
        memory_bank = torch.cat((memory_bank,memory_bank_zero_padding),dim=1)
        memory_bank = memory_bank.index_select(dim=1, index=rev_indices)

        return encoder_final, memory_bank, lengths

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout

    def _bridge_bidirectional_hidden(self,hidden):
        num_layers = hidden.size(0) //2
        _, batch_size, hidden_size = hidden.size()
        return hidden.view(num_layers, 2, batch_size, hidden_size).transpose(1,2).\
            contiguous().view(num_layers, batch_size, hidden_size*2)
