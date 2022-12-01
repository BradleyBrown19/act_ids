import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb
from hydra.utils import instantiate

class Regularization(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        pass

    def forward(self, model, inps, outs, step, epoch, group=""):
        raise NotImplementedError

class NoRegularization(Regularization):
    def __call__(self, model, inps, outs, step, epoch):
        return 0

class AEBottleneckReg(Regularization):
    def __init__(
        self,
        ae_bottlenecks,
        model,
        burn_in
    ):
        super().__init__()

        def ae_bottleneck_loss(ae_bottleneck):
            def hook(model, input, output):
                bs = output.shape[0]
                self.losses.append(ae_bottleneck.loss(output.reshape(bs,-1)))

            return hook

        self.ae_bottlenecks = nn.ModuleList([instantiate(ae_bottleneck, _recursive_=False) \
            for ae_bottleneck in ae_bottlenecks])
        self.hooks = []
        self.burn_in = burn_in

        self.losses = []

        for ae_bottleneck in self.ae_bottlenecks:
            layer_name = ae_bottleneck.layer_name
            
            for n, module in model.named_modules():
                if n == layer_name: break
            if n != layer_name: assert False, f"{layer_name} not found in model"

            self.hooks.append(module.register_forward_hook(ae_bottleneck_loss(ae_bottleneck)))
    
    def forward(self, model, batch, outs, step, epoch):
        self.losses = []
        if epoch < self.burn_in: return 0
        loss = sum(self.losses)
        return loss

class ClassAEBottleneckReg(Regularization):
    def __init__(
        self,
        ae_bottlenecks,
        model,
        burn_in
    ):
        super().__init__()

        def ae_bottleneck_loss(ae_bottleneck):
            def hook(model, input, output):
                bs = output.shape[0]
                ae_bottleneck.inputs = output.reshape(bs,-1)

            return hook

        self.ae_bottlenecks = nn.ModuleList([instantiate(ae_bottleneck, _recursive_=False) \
            for ae_bottleneck in ae_bottlenecks])
        self.hooks = []
        self.burn_in = burn_in

        for ae_bottleneck in self.ae_bottlenecks:
            layer_name = ae_bottleneck.layer_name
            
            for n, module in model.named_modules():
                if n == layer_name: break
            if n != layer_name: assert False, f"{layer_name} not found in model"

            self.hooks.append(module.register_forward_hook(ae_bottleneck_loss(ae_bottleneck)))
    
    def forward(self, model, batch, outs, step, epoch):
        if epoch < self.burn_in: return 0
        loss = 0
        for ae_bottleneck in self.ae_bottlenecks:
            loss += ae_bottleneck.loss(batch)

        return loss