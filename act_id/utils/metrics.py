import torch
import torch.nn.functional as F
import pdb
from collections import defaultdict

class Metric():
    def __init__(self):
        pass

    def __call__(self, model, inps, outs, step, group="", *args, **kwargs):
        raise NotImplementedError

class EpochMetric():
    def __init__(self):
        pass

    def __call__(self, model, dataloader, epoch, group="", *args, **kwargs):
        raise NotImplementedError

class Accuracy(Metric):
    def __init__(self):
        pass

    def __call__(self, model, inps, outs, step, group="", *args, **kwargs):
        metrics = {}

        metrics[group+"/accuracy"] = (outs.argmax(dim=-1) == inps[1]).float().mean()

        return metrics

class SMAccuracy(Metric):
    def __init__(self):
        pass

    def __call__(self, model, inps, outs, step, group="", *args, **kwargs):
        metrics = {}
        
        metrics[group+"/accuracy"] = (outs["logits"].argmax(dim=-1) == inps["labels"]).float().mean()

        return metrics

class NoEpochMetric(EpochMetric):
    def __init__(self):
        pass 

    def __call__(self, model, dataloader, epoch, group="", *args, **kwargs):
        return {}

class NoMetric():
    def __init__(self):
        pass

    def __call__(self, model, inps, outs, step, group="", *args, **kwargs):
        return {}

class EpochClassAccuracy(EpochMetric):
    def __init__(self):
        pass 

    def __call__(self, model, dataloader, epoch, group="", device="cpu"):
        metrics = {}

        cls_to_acc = defaultdict(list)

        total_corrects = []

        for batch in dataloader:
            imgs, labels = batch
            outs = model(imgs.to(device)).cpu()

            corrects = (outs.argmax(dim=-1) == labels).float().tolist()

            for correct, label in zip(corrects, labels):
                cls_to_acc[label.item()].append(correct)

            total_corrects.extend(corrects)
        
        for cls,acc in cls_to_acc.items():
            metrics[f"{group}/Class_{cls}_accuracy"] = sum(acc) / len(acc)
        
        metrics[f"{group}/total_accuracy"] = sum(total_corrects) / len(total_corrects)

        return metrics        

class MetricsList(Metric):
    def __init__(self, *args):
        self.metrics = args

    def __call__(self, model, inps, outs, step, group="", *args, **kwargs):
        metrics = {}

        with torch.no_grad():
            for metric in self.metrics:
                metrics.update(metric(model, inps, outs, step, group, *args, **kwargs))

        return metrics

class EpochMetricsList(Metric):
    def __init__(self, *args):
        self.metrics = args

    def __call__(self, model, dataloader, epoch, group="", *args, **kwargs):
        metrics = {}

        with torch.no_grad():
            for metric in self.metrics:
                metrics.update(metric(model, dataloader, epoch, group, *args, **kwargs))

        return metrics