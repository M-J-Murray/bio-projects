# Local Decoder
class LocalDecoderBlock(torch.nn.Module):
    """A local encoder block."""

    norm: torch.nn.RMSNorm
    att: WindowedAttentionLayer
    ffn: FfnSwiGluLayer

    def __init__(self, embedding_dims: int, num_heads: int, window_size: int) -> None:
        """
        Args:
            embedding_dims: int, The size of the embedding dimension of the input sequences.
            num_heads: int, The number of heads to use for windowed attention
            window_size: int, The amount of tokens around each token to consider during attention and merging.
        """
        super().__init__()
        self.norm = torch.nn.RMSNorm(embedding_dims)
        self.att = WindowedAttentionLayer(embedding_dims, num_heads, window_size)
        self.ffn = FfnSwiGluLayer(embedding_dims)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Applies the mergeDNA local decoder transformation to the sequence embeddings.

        Args:
            x: Tensor[batch_size, sequence_len, embedding_dims], sequence embeddings.
            p: Tensor[batch_size, sequence_len], The binary padding mask for the sequence.

        Returns:
            x: Tensor[batch_size, sequence_len, embedding_dims], the updated sequence embeddings.
        """
        x = x + self.att(self.norm(x), p)
        return x + self.ffn(self.norm(x))