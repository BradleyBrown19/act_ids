import pdb
from torch import nn
import torch

import hydra
from hydra.utils import instantiate

from pytorch_lightning import LightningModule
from act_id.utils.intrinsic_dimension import estimate_id

import pickle
import torch.nn.functional as F
from scipy.spatial.distance import pdist,squareform
import numpy as np
from torch.utils.data import Subset,  DataLoader
from itertools import chain

from collections import defaultdict
import pdb

class CommonTL(LightningModule):
    def __init__(
        self,
        config
    ):
        super().__init__()
        self.model = instantiate(config.model)
        self.metric = instantiate(config.metrics)
        self.epoch_metric = instantiate(config.epoch_metrics)
        self.regs = instantiate(config.regs, model=self.model, _recursive_=False)

        self.step_cnt = 0

        self.config = config
        self.save_hyperparameters(ignore="model")


    def configure_optimizers(self):

        opt = instantiate(self.config.optimizer,
                          params=chain(self.model.parameters(), self.regs.parameters()))
        sch = instantiate(self.config.lr_scheduler, optimizer=opt)

        return [opt], [sch]
    
    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        optimizer_closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ):
        lr = self.config.lr

        if self.trainer.global_step < self.config.warmup_steps:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / self.config.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * lr

            optimizer.step(closure=optimizer_closure)
        else:
            optimizer.step(closure=optimizer_closure)

class IDEstimationTL(CommonTL):
    def __init__(
        self,
        config
    ):
        super().__init__(config)

        if "id_estimation" in config:
            self.id_dict = {}
            self.id_dict["train"] = defaultdict(dict)
            self.id_dict["val"] = defaultdict(dict)
        
        self.train_step = 0
        self.last_train_step_done_for_val = -1
    
    def log_step_id(self, is_train):
        if "id_estimation" in self.config and self.config.id_estimation.do_steps and (self.config.id_estimation.estimate_train_id or not is_train): 
            self.eval()
            self.log_id(self.trainer.datamodule.train_dataloader() if is_train else self.trainer.datamodule.val_dataloader(), "train")
            self.train()
    
    def on_after_backward(self):
        self.train_step += 1

    def restrict_dataloader(self, dataloader, num_per_class=-1, cls=-1):
        if self.config.id_estimation.no_restrict: return dataloader

        used_indices = []
        labels = dataloader.dataset.targets

        cls_to_count = defaultdict(int)

        for idx,label in enumerate(labels):
            if (cls_to_count[label] < num_per_class or num_per_class == -1) and \
                (label == cls or cls == -1):
                    cls_to_count[label] += 1
                    used_indices.append(idx)
        
        dataset = Subset(dataloader.dataset, used_indices)

        return DataLoader(dataset, **self.config.id_estimation.dataloader)
    
    def compute_id(self, intermediates):
        ID = []
        n = int(np.round(intermediates.shape[0]*self.config.id_estimation.fraction))  

        dist = F.pdist(intermediates.to(self.device)).cpu()
        dist = squareform(dist)

        for i in range(self.config.id_estimation.nres):
            dist_s = dist
            perm = np.random.permutation(dist.shape[0])[0:n]
            dist_s = dist_s[perm,:]
            dist_s = dist_s[:,perm]
            ID.append(estimate_id(dist_s,verbose=False)[2])

        mean = np.mean(ID) 
        error = np.std(ID) 
        return mean,error

    def log_id(self, dataloader, group):

        current_step = self.train_step if self.config.id_estimation.do_steps else self.current_epoch

        if len(self.config.id_estimation.layers) == 0 or (current_step != (self.config.id_estimation.max_epochs-1) and \
            current_step % self.config.id_estimation.estimate_id_every != 0): return
        if self.config.id_estimation.do_steps and self.last_train_step_done_for_val == current_step and group == "val": return 
        
        if self.config.id_estimation.estimate_class_id: 
            num_classes = max(dataloader.dataset.targets)+1

        if self.config.id_estimation.estimate_data_id and current_step == 0:

            total_dataloader = {"total": self.restrict_dataloader(dataloader, num_per_class=self.config.id_estimation.combined_total_per_class)}
            class_dataloaders = {f"Class_{cls}": \
                self.restrict_dataloader(dataloader, num_per_class=self.config.id_estimation.num_per_class, cls=cls) for cls in range(num_classes)} \
                if self.config.id_estimation.estimate_class_id else {}
    
            for data_name,current_dataloader in {**total_dataloader, **class_dataloaders}.items():

                dataset = []

                for idx,(batch) in enumerate(current_dataloader):
                    imgs,_ = batch
                    dataset.append(imgs)
                
                dataset = torch.cat(dataset, 0)
                num_datapoints = dataset.shape[0]

                id,id_err = self.compute_id(dataset.reshape(num_datapoints, -1))
                self.log_dict({f"{group}/{data_name}/id_data": id, f"{group}/id_data_error": id_err, \
                    f"{group}/{data_name}/num_datapoints": float(len(current_dataloader.dataset.indices))}, sync_dist=True)
                
                self.id_dict[f"{group}/{data_name}/id_data"] = id
                self.id_dict[f"{group}/{data_name}/id_data_error"] = id_err
        
        if current_step == 0 and not self.config.id_estimation.estimate_initial_id: return

        total_dataloader = {"total": self.restrict_dataloader(dataloader, num_per_class=self.config.id_estimation.combined_total_per_class)}
        
        class_dataloaders = {f"Class_{cls}": \
            self.restrict_dataloader(dataloader, num_per_class=self.config.id_estimation.num_per_class, cls=cls) for cls in range(num_classes)} \
                if self.config.id_estimation.estimate_class_id else {}

        for data_name,dataloader in {**total_dataloader, **class_dataloaders}.items():
            with torch.no_grad():
                
                self.get_intermediates(dataloader)
                
                self.intermediates = {k: torch.cat([intermediate.reshape(intermediate.shape[0],-1) for intermediate in v], 0) for k,v in self.intermediates.items()}
             
                for name, intermediate in self.intermediates.items():
                    id,id_err = self.compute_id(intermediate)
                    self.log_dict({f"{group}/{data_name}/id_{name}": id, f"{group}/id_{name}_error": id_err, \
                        f"{group}/{data_name}/num_datapoints": float(len(dataloader.dataset))}, sync_dist=True)
                    
                    self.id_dict[f"{group}/{data_name}/id_{name}"] = id
                    self.id_dict[f"{group}/{data_name}/id_{name}_error"] = id_err
                
                for hook in self.hooks:
                    hook.remove()
                
        with open(self.config.id_estimation.save_path, 'wb') as f:
            pickle.dump(self.id_dict, f)
        
        self.last_train_step_done_for_val = current_step

    def on_train_epoch_end(self, *args, **kwargs):
        self.eval()
        metrics = self.epoch_metric(self.model, self.trainer.datamodule.train_dataloader(), epoch=self.current_epoch, group="train", device=self.device)

        self.log_dict(metrics, sync_dist=True)

        if "id_estimation" in self.config and not self.config.id_estimation.do_steps and self.config.id_estimation.estimate_train_id: self.log_id(self.trainer.datamodule.train_dataloader(), "train")
        self.train()
    
    def on_validation_epoch_end(self, *args, **kwargs):
        metrics = self.epoch_metric(self.model, self.trainer.datamodule.val_dataloader(), epoch=self.current_epoch, group="val", device=self.device)

        self.log_dict(metrics, sync_dist=True)
        
        if "id_estimation" in self.config and not self.config.id_estimation.do_steps: self.log_id(self.trainer.datamodule.val_dataloader(), "val")