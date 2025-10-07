from torch import nn


def get_criterion(criterion_name):

    # loss function
    if hasattr(nn, criterion_name):
        criterion = getattr(nn, criterion_name)()
    elif 'SimCLRLoss' in criterion_name:  # covers both self-supervised and supervised 
        from models.utils import SimCLRLoss
        criterion = SimCLRLoss()
    
    # metrics
    metrics = [criterion_name]
    if criterion_name == 'CrossEntropyLoss':
        metrics.extend(['acc1', 'acc5'])
    
    return criterion, metrics





