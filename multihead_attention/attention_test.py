import pytest
import torch
from attention import Attention


def test_output_shape():
    mha = Attention(d_model=256, num_heads=8)
    x = torch.rand(10, 20, 256)
    out = mha(x)
    assert out.shape == (10, 20, 256), f"Unexpected output shape: {out.shape}"


def test_mask():
    mha = Attention(d_model=8, num_heads=2)
    x = torch.rand(1, 5, 8)
    attention_mask = torch.ones(2, 5, 5)
    attention_mask[:, :, 3:] = 0  # masking the second half of the sequence
    out = mha(x, attention_mask=attention_mask)
    print(out)


if __name__ == "__main__":
    pytest.main([__file__])
