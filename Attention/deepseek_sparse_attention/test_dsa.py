"""
Test example for DSA.
"""

import torch
from dsa import ModelArgs, MLA, precompute_freqs_cis, torch_hadamard_transform


def test_hadamard_torch():
    """Verify that applying the normalized Hadamard twice recovers the input (within bfloat16 tolerance)."""
    print("=" * 60)
    print("Test: Hadamard Transform Inversion")
    print("=" * 60)

    n = 128
    x = torch.randn(2, n, dtype=torch.bfloat16)
    y = torch_hadamard_transform(x, scale=n**-0.5)
    z = torch_hadamard_transform(y, scale=n**-0.5)
    assert torch.allclose(z.float(), x.float(), atol=1e-2, rtol=1e-2)
    print("✓ Hadamard test passed!\n")


def test_mla_prefill():
    """
    Test MLA layer in prefill mode (with full attention mask).
    This is typically used during the first pass when generating initial tokens.
    """
    print("=" * 60)
    print("Test: MLA Prefill Mode (Full Attention)")
    print("=" * 60)

    # Create input tensor
    batch_size = 2
    seq_len = 256
    device = "cpu"

    # Create model arguments
    args = ModelArgs(
        max_batch_size=batch_size,
        max_seq_len=seq_len,
    )

    # Initialize MLA layer
    mla = MLA(args)
    mla.eval()

    x = torch.randn(batch_size, seq_len, args.dim, device=device, dtype=torch.bfloat16)

    # Precompute rotary embeddings for full sequence
    freqs_cis = precompute_freqs_cis(args)

    # Create causal mask (lower triangular matrix for autoregressive attention)
    mask = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device) * float("-inf"),
        diagonal=1,
    )
    mask = mask.unsqueeze(0)  # [1, seq_len, seq_len]

    # Forward pass with prefill (mask is not None)
    start_pos = 0
    with torch.no_grad():
        output = mla.forward(x, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(
        f"KV cache shape: {mla.kv_cache[:batch_size, start_pos:start_pos+seq_len].shape}"
    )
    print(
        f"PE cache shape: {mla.pe_cache[:batch_size, start_pos:start_pos+seq_len].shape}"
    )
    print(f"✓ Prefill test passed!\n")

    return mla, x, freqs_cis, args


def test_mla_decode(mla, x, freqs_cis, args):
    """
    Test MLA layer in decode mode (one token at a time with KV cache).
    This is typically used during generation after the prefill phase.
    """
    print("=" * 60)
    print("Test: MLA Decode Mode (Cached Decoding)")
    print("=" * 60)

    # Precompute rotary embeddings for full sequence
    freqs_cis = precompute_freqs_cis(args, seqlen=1)

    batch_size, seq_len = x.shape[0], x.shape[1]
    device = x.device

    # In decode, we process one token at a time
    next_token = torch.randn(
        batch_size, 1, args.dim, device=device, dtype=torch.bfloat16
    )

    # Start position is after the prefill
    start_pos = seq_len

    # In decode mode, mask is None (we attend to all previous tokens + current)
    with torch.no_grad():
        output = mla.forward(
            next_token, start_pos=start_pos, freqs_cis=freqs_cis, mask=None
        )

    print(f"Input shape (1 token): {next_token.shape}")
    print(f"Output shape: {output.shape}")
    print(
        f"KV cache up to position {start_pos+1}: {mla.kv_cache[:batch_size, :start_pos+1].shape}"
    )
    print(
        f"PE cache up to position {start_pos+1}: {mla.pe_cache[:batch_size, :start_pos+1].shape}"
    )
    print(f"✓ Decode test passed!\n")


if __name__ == "__main__":

    # Test Hadamard transform
    test_hadamard_torch()

    # Test Prefill mode
    mla, x, freqs_cis, args = test_mla_prefill()

    # Test Decode mode
    test_mla_decode(mla, x, freqs_cis, args)
