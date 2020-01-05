"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.rnn_knowledge_encoder import KnowledgeRNNEncoder
from onmt.encoders.cnn_encoder import CNNEncoder
from onmt.encoders.mean_encoder import MeanEncoder
from onmt.encoders.audio_encoder import AudioEncoder
from onmt.encoders.image_encoder import ImageEncoder


str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder, "cnn": CNNEncoder,
           "transformer": TransformerEncoder, "img": ImageEncoder,
           "audio": AudioEncoder, "mean": MeanEncoder, 'cue_brnn': KnowledgeRNNEncoder,
           'cue_rnn': KnowledgeRNNEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "KnowledgeRNNEncoder", "CNNEncoder",
           "MeanEncoder", "str2enc"]
