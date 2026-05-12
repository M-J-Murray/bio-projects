# Latent Encoder/Decoder
class LatentBlock(torch.nn.Module):
    """A latent encoder/decoder block."""

    norm: torch.nn.RMSNorm
    att: GlobalAttentionLayer
    ffn: FfnSwiGluLayer

    def __init__(self, embedding_dims: int, num_heads: int) -> None:
        """
        Args:
            embedding_dims: int, The size of the embedding dimension of the input sequences.
            num_heads: int, The number of heads to use for windowed attention
        """
        super().__init__()
        self.num_heads = num_heads
        self.norm = torch.nn.RMSNorm(embedding_dims)
        self.att = GlobalAttentionLayer(embedding_dims, num_heads)
        self.ffn = FfnSwiGluLayer(embedding_dims)

    def forward(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        s: Optional[torch.Tensor] = None,
        reduction: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Applies the mergeDNA latent block transformation to the sequence embeddings.

        Args:
            x: Tensor[batch_size, reduced_len, embedding_dims], sequence embeddings.
            p: Tensor[batch_size, reduced_len], The binary padding mask for the sequence.
            s: Tensor[batch_size, reduced_len, sequence_len], The binary source matrix. Not included if doing decoding.
            reduction: the number of tokens to merge.

        Returns:
            x: Tensor[batch_size, reduced_len, embedding_dims], the updated sequence embeddings.
            p: Tensor[batch_size, reduced_len], The updated binary padding mask for the sequence.
            s: Tensor[batch_size, reduced_len, sequence_len], The updated binary source matrix.
        """
        x = x + self.att(self.norm(x), p, s)
        if s is not None and reduction is not None:
            x, p, s = bipartite_soft_matching(x, x, p, s, reduction)
        return x + self.ffn(self.norm(x)), p, s