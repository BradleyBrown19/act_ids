from email.policy import default
from inspect import indentsize
from time import time
from act_id.training_loops.training_loop import IDEstimationTL
from act_id.utils.intrinsic_dimension import estimate_id
import pickle
from torch import nn
import torch.nn.functional as F
import torch
from scipy.spatial.distance import pdist,squareform
import numpy as np
import time
from torch.utils.data import Subset,  DataLoader

from collections import defaultdict
import pdb

class ImageClassificationTL(IDEstimationTL):
    def __init__(self, config):
        super().__init__(config)
       
        class_weights = self.get_class_weights(config)

        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)

        self.config = config
    
    def get_class_weights(self, config):
        if config.class_weights == 'None': 
            return torch.tensor([1/(config.num_classes) for _ in range(config.num_classes)])
        adjusted_class_weights_sum = sum([w**config.loss_weight_alpha for w in config.class_weights])
        adjusted_class_weights = [(w**config.loss_weight_alpha) / adjusted_class_weights_sum for w in config.class_weights]
        return torch.tensor(adjusted_class_weights)

    def training_step(self, batch, optimizer_idx=0, *args, **kwargs):
        imgs, labels = batch
        outs = self.model(imgs)
        
        base_loss = self.ce_loss(outs, labels)

        reg_loss = self.regs(self.model, batch, outs, step=self.step_cnt, epoch=self.current_epoch)

        with torch.no_grad():
            metrics = self.metric(self.model, batch, outs, step=self.step_cnt, group="train")
        
        loss = base_loss+reg_loss

        self.log_dict({"train/loss": loss, "train/base_loss": base_loss, "train/reg_loss": reg_loss, **metrics}, sync_dist=True)

        self.log_step_id(is_train=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        outs = self.model(imgs)
        base_loss = self.ce_loss(outs, labels)
      
        reg_loss = self.regs(self.model, batch, outs, step=self.step_cnt, epoch=self.current_epoch)

        metrics = self.metric(self.model, batch, outs,step=self.step_cnt,group="val")

        loss = base_loss+reg_loss

        self.log_dict({"val/loss": base_loss+reg_loss, "val/base_loss": base_loss, "val/reg_loss": reg_loss, **metrics}, sync_dist=True)

        self.log_step_id(is_train=False)

        return loss
    
    def get_intermediates(self, dataloader):
        self.intermediates = defaultdict(list)
        
        self.hooks = []

        def get_intermediate(name):
            def hook(model, input, output):
                to_save = output.detach().cpu()
                self.intermediates[name].append(to_save)

            return hook
        
        for name,module in self.model.named_children():
            if name in self.config.id_estimation.layers:
                self.hooks.append(module.register_forward_hook(get_intermediate(name)))
        

        for idx,(batch) in enumerate(dataloader):
            imgs, labels = batch
            outs = self.model(imgs.to(self.device))

    