""" Onmt NMT Model base class definition """
import torch.nn as nn
from onmt.modules.knowledge_fusion import Fusion

class KnowledgeModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, knowledge_encoder, decoder):
        super(KnowledgeModel, self).__init__()
        self.encoder = encoder
        self.knowledge_encoder = knowledge_encoder
        self.decoder = decoder
        # hidden_size = self.encoder.rnn.hidden_size
        # hidden_size = hidden_size * 2 if self.encoder.rnn.bidirectional else 1
        self.fusion = Fusion(self.encoder.rnn.hidden_size,
                             self.knowledge_encoder.rnn.hidden_size,
                             self.encoder.rnn.hidden_size)

    def forward(self, src, cue, tgt, src_lengths, cue_lengths,  bptt=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence of size ``(tgt_len, batch)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """

        tgt = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, src_lengths = self.encoder(src, src_lengths)

        length, n_facts, features, batch_size = cue.size()
        # note here the cue should be tranformed ,but the features is 1,\
        # so we directly view to convert cue into normal setting here.
        cue_input = cue.view(length, -1, features)
        enc_cue_state, cue_memory_bank, cue_lengths = self.knowledge_encoder(cue_input, cue_lengths.view(-1))

        # transform the enc_cue_state from num_layers * (batch_size)
        if isinstance(enc_cue_state, tuple):
            enc_cue_to_fusion = (enc_cue_state[0].view(-1,
                                               n_facts,
                                               batch_size,
                                               enc_cue_state[0].size(-1)).sum(dim=1),
                             enc_cue_state[1].view(-1,
                                                   n_facts,
                                                   batch_size,
                                                   enc_cue_state[1].size(-1)).sum(dim=1),
                             )
        #    print(enc_cue_to_fusion[0].size())
        else:
            enc_cue_to_fusion = enc_cue_state.view(self.knowledge_encoder.rnn.num_layers,
                                               n_facts,
                                               batch_size,
                                               enc_cue_state[0].size(-1)).sum(dim=1)
        #   print(enc_cue_state.size())
        #print(enc_state[0].size(), enc_state[1].size())
        fusion_state, forget_gate = self.fusion(enc_state, enc_cue_to_fusion)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, fusion_state)
        dec_out, attns = self.decoder(tgt, memory_bank,
                                      memory_lengths=src_lengths)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
