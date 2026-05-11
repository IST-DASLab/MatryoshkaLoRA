import torch

from global_cache import GlobalCache
from .lora import LoraLayer

class MatryoshkaLayer(LoraLayer):
    def __init__(self, base_layer, rank, scaling, train_ranks=None, mask_type='count_larger'):
        super().__init__(base_layer, rank, scaling)
        assert mask_type in ['diag', 'binary']
        self.mask_type = mask_type
        self.train_ranks = train_ranks

        self.maskA = None
        self.maskB = None

        self._create_masks()

    def _create_masks(self):
        A = self.lora_A
        B = self.lora_B
        train_ranks = self.train_ranks
        mask_type = self.mask_type

        if mask_type == 'diag':
            cache_categ = 'diag'
            cache_key = None
            if not GlobalCache.contains(category=cache_categ, key=cache_key):
                """
                    This is Algorithm 1 from the paper
                """
                mapping = {} # key = count, value = sum

                D = [0] * self.rank
                P = [0] * self.rank

                c = 0
                n = len(self.train_ranks)

                for r in range(self.rank, 0, -1):
                    i = r - 1
                    if r in train_ranks:
                        c += 1
                    D[i] = c

                    if c not in mapping:
                        mapping[c] = sum([
                            self.get_scale(train_ranks[j])
                            for j in range(n-c, n)
                        ])
                # end for r
                for i in range(self.rank):
                    P[i] = mapping[D[i]]
                print(f'{n=}')
                print(f'{D=}')
                print(f'{P=}')
                P = torch.tensor(P, dtype=torch.float, device=A.device)

                GlobalCache.add(category=cache_categ, key=cache_key, item=P)
        elif mask_type == 'binary':
            self.maskA = {}
            self.maskB = {}

            for i, r in enumerate(train_ranks):
                maskA = torch.zeros_like(A, dtype=torch.bool, device=A.device, requires_grad=False)
                maskB = torch.zeros_like(B, dtype=torch.bool, device=B.device, requires_grad=False)
                maskA[:, :r] = True
                maskB[:r, :] = True

                self.maskA[r] = maskA
                self.maskB[r] = maskB
        else:
            raise ValueError(f'Unknown mask_type: {self.mask_type}')

    def forward(self, x):
        base_output = self.base_layer(x)

        A = self.lora_A
        B = self.lora_B

        inf_rank = self.inf_rank
        if inf_rank is None:
            if self.mask_type == 'diag':
                cache_categ = 'diag'
                cache_key = None
                P = GlobalCache.get(category=cache_categ, key=cache_key)
                lora_output = x @ (A * P) @ B # P embeds scales
            elif self.mask_type == 'binary':
                maskA = self.maskA
                maskB = self.maskB

                lora_output = 0

                for r in self.train_ranks:
                    lora_output += (self.get_scale(r) / len(self.train_ranks)) * ((x @ (A * maskA[r])) @ (B * maskB[r]))
            else:
                raise ValueError(f'Unknown mask_type: {self.mask_type}')
        else:
            assert 0 < inf_rank <= self.rank
            lora_output = self.get_scale(inf_rank) * ((x @ A[:, :inf_rank]) @ B[:inf_rank, :])

        return base_output + lora_output
