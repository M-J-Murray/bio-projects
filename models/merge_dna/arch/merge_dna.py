from typing import Optional

import torch
from composite.nn.token_merging import unmerge

from models.merge_dna.arch.latent_block import LatentBlock
from models.merge_dna.arch.local_decoder import LocalDecoderBlock
from models.merge_dna.arch.local_encoder import LocalEncoderBlock


# MergeDNA model
class MergeDNAModel(torch.nn.Module):
    embedder: torch.nn.Embedding
    local_encoder: torch.nn.ModuleList
    latent_encoder: torch.nn.ModuleList
    latent_decoder: torch.nn.ModuleList
    local_decoder: torch.nn.ModuleList
    head: torch.nn.Linear

    def __init__(
        self,
        local_encoder_blocks: int,
        latent_encoder_blocks: int,
        latent_decoder_blocks: int,
        local_decoder_blocks: int,
        vocab_size: int,
        embedding_dims: int,
        num_heads: int,
        window_size: int,
        top_k: int,
        temperature: float,
    ) -> None:
        """
        Args:
            local_encoder_blocks: int, The number of local encoder blocks to use in the model.
            latent_encoder_blocks: int, The number of latent encoder blocks to use in the model.
            latent_decoder_blocks: int, The number of latent decoder blocks to use in the model.
            local_decoder_blocks: int, The number of local decoder blocks to use in the model.
            vocab_size: The number of tokens in the model vocabulary.
            embedding_dims: int, The size of the embedding dimension of the input sequences.
            num_heads: int, The number of heads to use for windowed attention
            window_size: int, The amount of tokens around each token to consider during attention and merging.
            top_k: int, The number of tokens from set B to be considered during soft grouping.
            temperature: float, Tuning parameter for regulating DTEM relaxation.
        """
        super().__init__()
        self.embedder = torch.nn.Embedding(vocab_size, embedding_dims)
        self.local_encoder = torch.nn.ModuleList(
            [
                LocalEncoderBlock(embedding_dims, num_heads, window_size, top_k, temperature)
                for _ in range(local_encoder_blocks)
            ]
        )
        self.latent_encoder = torch.nn.ModuleList(
            [LatentBlock(embedding_dims, num_heads) for _ in range(latent_encoder_blocks)]
        )
        self.latent_decoder = torch.nn.ModuleList(
            [LatentBlock(embedding_dims, num_heads) for _ in range(latent_decoder_blocks)]
        )
        self.local_decoder = torch.nn.ModuleList(
            [LocalDecoderBlock(embedding_dims, num_heads, window_size) for _ in range(local_decoder_blocks)]
        )
        self.head = torch.nn.Linear(embedding_dims, vocab_size, bias=False)

    def forward(
        self,
        token_ids: torch.Tensor,
        p: torch.Tensor,
        local_reduction: int,
        latent_reduction: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Applies the forward pass of the mergeDNA model.

        Args:
            token_ids: Tensor[batch_size, sequence_len], sequence token ids.
            p: Tensor[batch_size, sequence_len], The binary padding mask for the sequence.
            local_reduction: int, The number of local tokens to merge per local encoder block.
            latent_reduction: Optional[int], if enabled, the number of latent tokens to merge per latent encoder block.

        Returns:
            x: Tensor[batch_size, sequence_len, embedding_dims], the updated sequence embeddings.
            latent_s: Tensor[batch_size, sequence_len, sequence_len], the binary source matrix from the latent encoder.
        """
        B, S = token_ids.size()
        s = p.new_zeros(B, S, S, dtype=torch.float)  # Tensor[batch_size, sequence_len, sequence_len]
        s.diagonal(dim1=-2, dim2=-1).fill_(1.0)

        # Local Encoder Forward
        with torch.set_grad_enabled(latent_reduction is None):
            x = self.embedder(token_ids)
            local_p, local_s = p, s
            for block in self.local_encoder:
                x, local_p, local_s = block(x, local_p, local_s, local_reduction)

        # Latent Encoder Forward
        latent_p, latent_s = local_p, local_s
        for block in self.latent_encoder:
            x, latent_p, latent_s = block(x, latent_p, latent_s, latent_reduction)

        if latent_reduction is not None:
            x = unmerge(x, latent_s)

        # Latent Decoder Forward
        for block in self.latent_decoder:
            x, _, _ = block(x, local_p, local_s)

        x = unmerge(x, local_s)

        # # Local Decoder Forward
        for block in self.local_decoder:
            x = block(x, p)

        return self.head(x), latent_s
