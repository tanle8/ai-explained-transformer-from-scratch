from numpy import mean
import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    The Input Embedding block maps raw token indices to dense vectors which represent those tokens in a learned,
    continuous space that the Transformer can process.

    d_model: dimentinality of the embeddings. In the paper, d_{model} = 512. So each embedding (of a token) is a vector of size d_model = 512.
    vocab_size: the total number of your token in your input vocabulary (e.g. if we do machine translation with a vocabulary of a 10000 word tokens, vocab_size = 10000).
    """

    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        In the beginning, x has shape (batch, len_seq),
        batch : the number of sequence in a mini-batch.
        len_seq : total number of tokens per sequence.

        Each element in x is an integer index: a value between 0 and vocab_size - 1.

        self.embedding(x) looks up the vector for each token index in x. The result will have
        the shape of (batch, seq_len, d_model). It means for each integer token in every sequence,
        we get its d_model - dimentional embedding.

        sqrt(self.d_model) : In section 3.4 Embeddings and Softmax, the paper mentions that we should do it to scale the embeddings
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    According to the paper, for the model can utilize the order of the sequence. 
    So, we have to "inject" some information about the relative or absolute of the tokens in the sequence. 
    That's why positional embeddings are added to input embeddings.

    where `pos` is the position and `i` is the dimension.
    - Even dimensions (2i) use sine
    - Odd dimensions (2i + 1) use cosine.
    - The frequcency of the sine/cosine wave is controlled by 10000^{-2i/d_model}

    Ref: section 3.5 Positional Encoding

    """

    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Step 1: Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Step 2: Create a vector of shape (seq_len, 1)
        position = torch.arange(
            0, seq_len, dtype=torch.float
        ).unsqueeze(1)  # add/insert a new dimension at index 1

        # Step 3: Create a vector of shape (d_model/2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        # Step 4: Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)

        # Step 5: Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Step 6: Add a batch dimension: shape (1, seq_len, d_model)
        pe = pe.unsqueeze(0)

        # Step 7: Register the positional encoding as a buffer,
        # we store `pe` in the model so PyTorch knows about it,
        # but doesn't treat it as a learnable parameter.
        self.register_buffer('pe', pe)

    def forward(self,):
        """
        x : Input embeddings (batch, seq_len, d_model)

        To make sure the positional encoding values aren't trainable, we use requires_grad_(False).
        That makes the gradient go back through `x`, not through the `pe` buffer.
        """

        # Step 8: Add positional encodings to x
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)

        # Step 9: Apply dropout
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """
    Args:
        features: int - the size of the last dimension (or `hidden_size` in Transformers)
        epsilon: float - a small constant to avoid division by zero.

    """

    def __init__(self, features: int, epsilon: float = 10**-6) -> None:
        super().__init__()
        self.epsilon = epsilon

        # alpha is a learnable parameter
        self.alpha = nn.Parameter(torch.ones(features))
        # bias is a learnable parameter
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_size)
        """
        # Compute the mean (across the last dimension, dim=-1), which is the hidden_size dimension.
        mean = x.mean(dim=-1, keepdim=True)  # (batch, seq_len, 1)
        # Compute the standard deviation (also across the last dimension)
        std = x.std(dim=-1, keepdim=True)   # (batch, seq_len, 1)

        return self.alpha * (x - mean) / (std + self.epsilon) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Implements the position-wise feed-forward networks.

    Args:
        d_ff : The intermediate/hidden dimension for this feed-forward layer. According to paper, the inner-layer has dimensionality d_ff = 2048

    Ref: 
        Section 3.3 Position-wise Feed-Forward Networks
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        # w1 and b1 (first projection - expansion)
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        # w2 and b2 (second projection - reduction)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor):
        """
        Applies the transformation: Linear -> ReLU -> Dropout -> Linear.
        The shapes of corresponding transforms: (batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)

        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, d_model)

        Returns:
            torch.Tensor: The transformed tensor of shape (batch, seq_len, d_model)
        """
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class ResidualConnection(nn.Module):
    """
    A Residula (skip) Connection also known as "Add & Norm" block in the paper.

    It adds the identity path (`x + ...`), make it easier for gradients to flow back through many
    layers, reducing vanishing or exploding gradients. 
    """

    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention block

    In the Transformer model, attention is the mechanism that allows each position in a sequence
    focus on (attend) other positions (tokens) in the same sequence or in another one.

    Multi-head attension means intead of computing a single attention func, the model run 
    `h` attention "heads", each head focus on a different part of the sequence or different 
    kind of concerns like one focuses on local text, another is on the next token, ...

    The results from each head are combined (concatenated) and transformed back into the originnal
    dimension.
    """

    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model  # Embedding vector size or total model dimension
        self.h = h              # Number of heads
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"
        self.d_k = d_model // h  # Dimension of each head

        # Projection matrices Wq, Wk, Wv, each in R^(d_model x d_model).
        self.w_q = nn.Linear(d_model, d_model, bias=False)  # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False)  # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False)  # Wv

        # Output projection matrix Wo in R^(d_model x d_model)
        self.w_o = nn.Linear(d_model, d_model, bias=False)  # Wo

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # 1) Score = QK^T / sqrt(d_k)
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # If a mask is given (e.g., to avoid attending to future tokens)
        if mask is not None:
            # Where mask == 0, set attention_scores to a very negative value.
            attention_scores.masked_fill_(mask == 0, -1e9)

        # 2) Apply softmax (batch, h, seq_len, seq_len)
        attention_scores = attention_scores.softmax(dim=-1)

        # 3) Dropout on the attention map
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # 4) Multiply by V
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # Step 1: Project q, k, v from d_model to d_model via w_q, w_k, w_v
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # Step 2: Reshape (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)
        # then transpose -> (batch, h, seq_len, d_k)
        query = query.view(
            query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1],
                       self.h, self.d_k).transpose(1, 2)
        value = value.view(
            value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Step 3: Compute scaled dot-product attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(
            query, key, value, mask, self.dropout)

        # Step 4: Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(
            x.shape[0], -1, self.h * self.d_k)

        # Step 5: Apply final linear projection w_o
        # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
        return self.w_o(x)


class EncoderLayer(nn.Module):
    """
    A layer of encoder
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(2)])  # need two residual connection around self-attention and feed-forward

    def forward(self, x, src_mask):
        """
        t
        """
        x = self.residual_connections[0](
            # Pass the multi-head attention into the residual connection.
            x, lambda x: self.self_attention_block(x, x, x, src_mask)
        )
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    Wrap a list of encoder layers to form a multi layer Transformer encoder (Nx)
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        """
        Args:
            self.layers : A list of EncoderLayer objects (e.g., 6-12 layers, in the paper N = 6)
            self.norm : A final layer normalization after all layers are done.
        """
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """
        Loops over each encoder layer, passing `x` through each of them, along with the `mask`
        and after all, perform normalization.
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    """
    A layer of decoder.

    Args:
        cross-attention_block : this often called "encoder-decoder attention" (like in the paper). It uses the `encoder_output`
            as `key` and `value`, while the target tokens are `query`. This attention mechanism help the decoder 
            focus on the encoder's representation of the source sequence.

        residua_connections: Each sub-layer is wrapped in a residual connection and layer normalization

    Ref:
        3.2.3 Applications of Attention in our Model
    """

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(
            [ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](
            x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](
            x, lambda x: self.cross_attention_block(
                x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](
            x, self.feed_forward_block)
        
        return x


class Decoder(nn.Module):
    """
    Wrap a list of decoder layers to form a multi layer Transformer decoder (Nx). In the paper, N=6
    """

    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class ProjectionLayer(nn.Module):
    """
    At the end of the decoder, the transformer needs to generate a probability distribution over the
    vocabulary for each position in the sequence. Since our ultimate goal is to predict tokens from
    a vocabulary of size `vocab_size`, that's why we need the final step that maps each position's
    hidden vector (`d_model` dimension) to a distribution across `vocab_size`.
    
    It does that by taking it hidden representation of shape `(batch, seq_len, d_model)`
    and projects it into a vector `(batch, seq_len, vocab_size)`.

    Ref:
        Section 3.4 Embeddings and Softmax
    """

    def __init__(self, d_model, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    """
    Final wrapper of elements
    """

    def __init__(self, 
                encoder: Encoder,
                decoder: Decoder,
                src_embed: InputEmbeddings,
                tgt_embed: InputEmbeddings,
                src_pos: PositionalEncoding,
                tgt_pos: PositionalEncoding,
                projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        # 1) Token embedding
        src = self.src_embed(src)
        # 2) Positional encoding
        src = self.src_pos(src)
        # 3) Pass to the encoder
        return self.encoder(src, src_mask)  # (batch, seq_len, d_model)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt: torch.Tensor, tgt_mask: torch.Tensor):
        # 1) Token embedding for the target tokens
        tgt = self.tgt_embed(tgt)
        # 2) Positional encoding for the target
        tgt = self.tgt_pos(tgt)
        # 3) Pass to the decoder, along with encoder outputs
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)    # (batch, seq_len, d_model)

    def project(self, x):
        # final linear projection from hidden dimension to vocab
        return self.projection_layer(x) # (batch, seq_len, vocab_size)


def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    """
    A helper function to instantiate all the submodules, chain them together, and do initialization step,

    Return:
        Transformer: a fully assembled Transformer model.
    """
    # 1) Create the embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

    # 2) Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # 3) Create the encoder blocks
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderLayer(
            d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)


    # 4) Create the decoder blocks
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(
            d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderLayer(d_model, decoder_self_attention_block,
                                    decoder_cross_attention_block, 
                                    feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)


    # Create encoder and decoder from corresponding blocks
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # 5) Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # 6) Instantiate a Transformer object
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

    # 7) Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
