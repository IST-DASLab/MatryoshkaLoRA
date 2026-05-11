from adapters import LoraLayer

class DyLoraLayer(LoraLayer):
    def forward(self, x):
        base_output = self.base_layer(x)

        A = self.lora_A
        B = self.lora_B

        inf_rank = self.inf_rank
        assert 0 < inf_rank <= self.rank

        lora_output = self.get_scale(inf_rank) * ((x @ A[:, :inf_rank]) @ B[:inf_rank, :])

        return base_output + lora_output