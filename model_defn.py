import tensorflow as tf
import opennmt as onmt

#
# defines the [model] type to train, called in the [train*.sh] script
# tip on overfitting: for single-pair/smaller models dropout=0.3, while for multilingual dropout=[0.1-0.3] shows best result
#

class Transformer(onmt.models.Transformer):
  """Defines a Transformer model as decribed in https://arxiv.org/abs/1706.03762."""
  def __init__(self, dtype=tf.float32, share_embeddings=onmt.models.EmbeddingsSharingLevel.NONE):
    super(Transformer, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512,
            dtype=dtype),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512,
            dtype=dtype),
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        relu_dropout=0.1,
        share_embeddings=share_embeddings)

class TransformerFP16(Transformer):
  """Defines a Transformer model that uses half-precision floating points."""
  def __init__(self):
    super(TransformerFP16, self).__init__(dtype=tf.float16)

class TransformerSharedEmbd(Transformer):
    """Defines a Transformer model that uses shared encoder-decoder embeddings."""
    def __init__(self):
        super(TransformerSharedEmbd, self).__init__(
            share_embeddings=onmt.models.EmbeddingsSharingLevel.SOURCE_TARGET_INPUT
        )

class TransformerMedium(onmt.models.Transformer):
  """Defines a 4 SA layer Transformer model comparable with related works."""
  def __init__(self):
    super(TransformerMedium, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="source_words_vocabulary",
            embedding_size=512),
        target_inputter=onmt.inputters.WordEmbedder(
            vocabulary_file_key="target_words_vocabulary",
            embedding_size=512),
        num_layers=4,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.3,
        attention_dropout=0.1,
        relu_dropout=0.1)

'''
class TransformerSA(onmt.models.Transformer):
  """Defines a Transforme:wqr model as decribed in https://arxiv.org/abs/1706.03762."""
  def __init__(self, dtype=tf.float32, share_embeddings=onmt.models.EmbeddingsSharingLevel.NONE, dropout=0.1):
     super(TransformerSA, self).__init__(
        source_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
        target_inputter=onmt.inputters.WordEmbedder(embedding_size=512),
        num_layers=6,
        num_units=512,
        num_heads=8,
        ffn_inner_dim=2048,
        dropout=0.1,
        attention_dropout=0.1,
        ffn_dropout=0.1,
        share_embeddings=share_embeddings)


class TransformerShareEmbs(TransformerSA):
    """Defines a Transformer model that uses shared encoder-decoder embeddings."""
    def __init__(self):
        super(TransformerShareEmbs, self).__init__(
            share_embeddings=onmt.models.EmbeddingsSharingLevel.ALL
        )


class TransformerShareEmbsDropout(TransformerSA):
  """Defines a Transformer model that uses shared encoder-decoder embeddings."""
  def __init__(self):
    super(TransformerShareEmbsDropout, self).__init__(
            dropout=0.3,
            share_embeddings=onmt.models.EmbeddingsSharingLevel.ALL)
'''

# update accordingly 
#model = Transformer
model = TransformerMedium
