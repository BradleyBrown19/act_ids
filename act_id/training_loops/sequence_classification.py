from copy import deepcopy
import pdb
from act_id.training_loops.training_loop import IDEstimationTL
from torch import nn
import torch
from collections import defaultdict
import pickle

class SequenceClassificationTL(IDEstimationTL):

    def __init__(self, config):
        super().__init__(config)
        self.ce_loss = nn.CrossEntropyLoss()

    def training_step(self, batch):
        outs = self.model(**batch)
        loss = outs.loss

        with torch.no_grad():
            metrics = self.metric(self.model, batch, outs, step=self.step_cnt, group="train", device=self.device)

        self.log_dict({"train/loss": loss, **metrics}, sync_dist=True)

        self.log_step_id(is_train=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        
        outs = self.model(**batch)
        loss = outs.loss

        with torch.no_grad():
            metrics = self.metric(self.model, batch, outs, step=self.step_cnt, group="val", device=self.device)

        self.log_dict({"val/loss": loss, **metrics}, sync_dist=True)

        self.log_step_id(is_train=False)

        return loss
    
    def get_intermediates(self, dataloader):
        self.intermediates = defaultdict(list)
        self.hooks = []

        def get_intermediate(name):
            def hook(model, input, output):

                output_copy = deepcopy(output)

                while type(output_copy) == tuple:
                    output_copy = output_copy[0]

                to_save = output_copy.detach().cpu()
                self.intermediates[name].append(to_save)

            return hook

        def register_hooks(model, base_name=""):
            for name,module in model.named_children():
                if base_name + name in self.config.id_estimation.layers:

                    self.hooks.append(module.register_forward_hook(get_intermediate(base_name + name)))
                    register_hooks(module, base_name + name + ".")
                
                else:

                    register_hooks(module, base_name + name + ".")
        
        register_hooks(self.model)
        
        for idx,(batch) in enumerate(dataloader):
            self.model(**{k:v.to(self.device) for k,v in batch.items()})