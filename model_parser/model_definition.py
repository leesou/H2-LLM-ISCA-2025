from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelShape:
    dim: int
    ffn_dim: int
    n_heads: int
    n_kv_heads: int


class OPTLayer(nn.Module):
    def __init__(
        self,
        model_shape: ModelShape
    ):
        super().__init__()

        self.dim = model_shape.dim
        self.ffn_dim = model_shape.ffn_dim
        self.head_dim = model_shape.dim // model_shape.n_heads
        self.n_heads = model_shape.n_heads
        self.n_kv_heads = model_shape.n_kv_heads
        self.dtype = torch.float16

        # Attention Block
        self.wq = nn.Linear(
            self.dim,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=self.dtype
        )
        self.wk = nn.Linear(
            self.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            dtype=self.dtype
        )
        self.wv = nn.Linear(
            self.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            dtype=self.dtype
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_dim,
            self.dim,
            bias=False,
            dtype=self.dtype
        )

        # FFN Block
        self.w1 = nn.Linear(
            self.dim,
            self.ffn_dim,
            bias=False,
            dtype=self.dtype
        )
        self.w2 = nn.Linear(
            self.ffn_dim,
            self.dim,
            bias=False,
            dtype=self.dtype
        )

    def forward(self, x):
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2) # (bs, n_heads, seqlen, head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.softmax(torch.matmul(xq, xk.transpose(2, 3)), dim=-1)
        output = torch.matmul(scores, xv).transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        xo = self.wo(output)

        x1 = self.w1(xo)

        x2 = self.w2(x1)

        return x2


class LLaMALayer(nn.Module):
    def __init__(
        self,
        model_shape: ModelShape
    ):
        super().__init__()

        self.dim = model_shape.dim
        self.ffn_dim = model_shape.ffn_dim
        self.head_dim = model_shape.dim // model_shape.n_heads
        self.n_heads = model_shape.n_heads
        self.n_kv_heads = model_shape.n_kv_heads
        self.dtype = torch.float16

        # Attention Block
        self.wq = nn.Linear(
            self.dim,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=self.dtype
        )
        self.wk = nn.Linear(
            self.dim,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=self.dtype
        )
        self.wv = nn.Linear(
            self.dim,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=self.dtype
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_dim,
            self.dim,
            bias=False,
            dtype=self.dtype
        )

        # FFN Block
        self.w1 = nn.Linear(
            self.dim,
            self.ffn_dim,
            bias=False,
            dtype=self.dtype
        )
        self.w2 = nn.Linear(
            self.ffn_dim,
            self.dim,
            bias=False,
            dtype=self.dtype
        )
        self.w3 = nn.Linear(
            self.dim,
            self.ffn_dim,
            bias=False,
            dtype=self.dtype
        )

    def forward(self, x):
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2) # (bs, n_heads, seqlen, head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.softmax(torch.matmul(xq, xk.transpose(2, 3)), dim=-1)
        output = torch.matmul(scores, xv).transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        xo = self.wo(output)

        x1 = self.w1(xo)
        x3 = F.silu(self.w3(xo))

        x2 = self.w2(x1*x3)

        return x2


class PaLMLayer(nn.Module):
    def __init__(
        self,
        model_shape: ModelShape
    ):
        super().__init__()

        self.dim = model_shape.dim
        self.ffn_dim = model_shape.ffn_dim
        self.head_dim = model_shape.dim // model_shape.n_heads
        self.n_heads = model_shape.n_heads
        self.n_kv_heads = model_shape.n_kv_heads
        self.dtype = torch.float16

        # Attention Block
        self.wq = nn.Linear(
            self.dim,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=self.dtype
        )
        self.wk = nn.Linear(
            self.dim,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=self.dtype
        )
        self.wv = nn.Linear(
            self.dim,
            self.n_heads * self.head_dim,
            bias=False,
            dtype=self.dtype
        )
        self.wo = nn.Linear(
            self.n_heads * self.head_dim,
            self.dim,
            bias=False,
            dtype=self.dtype
        )

        # FFN Block
        self.w1 = nn.Linear(
            self.dim,
            self.ffn_dim,
            bias=False,
            dtype=self.dtype
        )
        self.w2 = nn.Linear(
            self.ffn_dim,
            self.dim,
            bias=False,
            dtype=self.dtype
        )
        self.w3 = nn.Linear(
            self.dim,
            self.ffn_dim,
            bias=False,
            dtype=self.dtype
        )

    def forward(self, x):
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2) # (bs, n_heads, seqlen, head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)
        xv = self.wv(x).view(bsz, seqlen, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.softmax(torch.matmul(xq, xk.transpose(2, 3)), dim=-1)
        output = torch.matmul(scores, xv).transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        xo = self.wo(output)

        x1 = self.w1(x)
        x3 = F.silu(self.w3(x))

        x2 = self.w2(x1*x3)

        return x + xo + x2


model_dict = {
    "opt": OPTLayer,
    "llama": LLaMALayer,
    "palm": PaLMLayer
}
