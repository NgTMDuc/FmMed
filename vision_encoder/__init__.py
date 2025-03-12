from .CtViT.ctvit import CTViT
from downstream_task import ClassificationTask

def load_classify_task(cfg, model):
    return ClassificationTask(model, cfg.num_classes)

def load_task(cfg, model):
    if cfg.task == "Classification":
        return load_classify_task(cfg, model)