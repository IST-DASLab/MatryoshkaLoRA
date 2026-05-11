import torch
import torch.nn as nn
import math

class LoraLayer(nn.Module):
    def __init__(self, base_layer, rank, scaling):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = scaling
        self.inf_rank = None

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        W = base_layer.weight
        device = W.device
        dtype = W.dtype

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank, dtype=dtype, device=device))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, dtype=dtype, device=device))

        self.lora_A.requires_grad = True
        self.lora_B.requires_grad = True

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def set_inference_rank(self, inf_rank):
        assert (inf_rank is None) or (0 < inf_rank <= self.rank)
        self.inf_rank = inf_rank

    def get_scale(self, r):
        if self.scaling == 'none':
            return 1.
        if self.scaling == 'rank':
            return 1. / r
        if self.scaling == 'sqrt':
            return 1. / math.sqrt(r)
        raise RuntimeError(f'Unknown scaling: {self.scaling}')
        # match self.scaling:
        #     case 'no':
        #         return 1.
        #     case 'rank':
        #         return 1. / r
        #     case 'sqrt':
        #         return 1. / math.sqrt(r)
        #     case _:
        #         raise RuntimeError(f'Unknown scaling: {self.scaling}')

    def forward(self, x):
        base_output = self.base_layer(x)

        A = self.lora_A
        B = self.lora_B

        inf_rank = self.inf_rank
        if inf_rank is None:
            lora_output = self.get_scale(self.rank) * ((x @ A) @ B)
        else:
            assert 0 < inf_rank <= self.rank
            lora_output = self.get_scale(inf_rank) * ((x @ A[:, :inf_rank]) @ B[:inf_rank, :])

        return base_output + lora_output