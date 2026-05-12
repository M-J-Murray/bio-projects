# Local Encoder
class LocalEncoderBlock(torch.nn.Module):
    """A local encoder block."""

    norm: torch.nn.RMSNorm
    att: WindowedAttentionLayer
    dtem: WindowedDtemLayer
    ffn: FfnSwiGluLayer

    def __init__(self, embedding_dims: int, num_heads: int, window_size: int, top_k: int, temperature: float) -> None:
        """
        Args:
            embedding_dims: int, The size of the embedding dimension of the input sequences.
            num_heads: int, The number of heads to use for windowed attention
            window_size: int, The amount of tokens around each token to consider during attention and merging.
            top_k: int, The number of tokens from set B to be considered during soft grouping.
            temperature: float, Tuning parameter for regulating DTEM relaxation.
        """
        super().__init__()
        self.norm = torch.nn.RMSNorm(embedding_dims)
        self.att = WindowedAttentionLayer(embedding_dims, num_heads, window_size)
        self.dtem = WindowedDtemLayer(embedding_dims, window_size, top_k, temperature)
        self.ffn = FfnSwiGluLayer(embedding_dims)

    def forward(
        self, x: torch.Tensor, p: torch.Tensor, s: torch.Tensor, reduction: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies the mergeDNA local encoder transformation to the sequence embeddings.

        Args:
            x: Tensor[batch_size, reduced_len, embedding_dims], sequence embeddings.
            p: Tensor[batch_size, reduced_len], The binary padding mask for the sequence.
            s: Tensor[batch_size, reduced_len, sequence_len], The binary source matrix.
            reduction: then number of tokens being merged in this block.

        Returns:
            x: Tensor[batch_size, reduced_len, embedding_dims], the updated sequence embeddings.
            p: Tensor[batch_size, reduced_len], The updated binary padding mask for the sequence.
            s: Tensor[batch_size, reduced_len, sequence_len], The updated binary source matrix.
        """
        x_att = x + self.att(self.norm(x), p, s)
        x_att, p, s = self.dtem(x, x_att, p, s, reduction)
        return x_att + self.ffn(self.norm(x_att)), p, s