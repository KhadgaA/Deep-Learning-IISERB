import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from models import *
import helper as H
import loader as dl
import plots as pt
import utils as U
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import tqdm
import os
import time
start_time = time.time()



def get_lr(optimizer):
    for parma_group in optimizer.param_groups:
        return parma_group['lr']

def fit(epochs:int, max_lr:float, loss_func:object, model, train_loader, val_loader,model_name, grad_clip:float=None,
        optimizer=torch.optim.SGD,**kwargs)->list:
    history = []
    best_acc = 0
    # setting up optimizer
    optim = optimizer(model.parameters(),lr = max_lr,**kwargs)
    for epoch in range(epochs):
        # training
        model.train()
        train_loss = []
        lrs = []
        for batch in tqdm(train_loader):
            # print(batch[0].shape)
            loss = H.training_step(model, loss_func=loss_func, batch=batch)
            train_loss.append(loss)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optim.step()
            optim.zero_grad()

        # recording lr
        lrs.append(get_lr(optim))


        result = H.evaluate(model=model, loss_func=loss_func, loader=val_loader)
        val_loss = result['val_loss']

        result['train_loss'] = torch.stack(train_loss).mean().item()
        result['lrs'] = lrs
        H.epoch_end(epoch, result)
        history.append(result)
        if(x:=result['val_acc'])>best_acc:
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'acc': x
            }
            torch.save(state,model_name+'best.pth')
            best_acc = result['val_acc']
        print(f'Best_Acc: {best_acc:.3f}')
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'acc': x
    }
    torch.save(state, model_name + 'last.pth')
    return history


if __name__ =='__main__':



    epochs = 50
    max_lr = 0.001
    loss = F.cross_entropy
    batch_size = 256
    dataset = 'cifar10'
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    device = torch.device(f'cuda:{0}') if torch.cuda.is_available() else 'cpu'
    train_trnsfrm = [
        tt.ToTensor(),
        tt.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ]
    test_trnsfrm = [
        tt.ToTensor(),
        tt.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))

    ]

    train_loader, val_loader = dl.dataloader(train_trnsforms=train_trnsfrm, test_trnsforms=test_trnsfrm,
                                             dataset=dataset, device=device, batch_size=batch_size,
                                             root='../../../KDGen/data') # Change the data folder


    def numel(m: torch.nn.Module, only_trainable: bool = False):
        """
        Returns the total number of parameters used by `m` (only counting
        shared parameters once); if `only_trainable` is True, then only
        includes parameters with `requires_grad = True`
        """
        parameters = list(m.parameters())
        if only_trainable:
            parameters = [p for p in parameters if p.requires_grad]
        unique = {p.data_ptr(): p for p in parameters}.values()
        return sum(p.numel() for p in unique)


    for ps in [2,3,4]:
        for st in [1,2]:
            for model in [model1(pool_size=ps,stride=st),model2(pool_size=ps,stride=st),model3(pool_size=ps,stride=st)]:

                model.to(device)
                optimizer = optim.SGD
                model_name = f'{model._get_name()}_{dataset}_{ps}_{st}'

                with open('model_param.txt','a') as f:
                    print(f'{model_name}: ', numel(model, only_trainable=True),file=f) # Counting the number of params of the model


                kwargs = {'momentum' : 0.9,'weight_decay' : 5e-4}
                history = [H.evaluate(model, val_loader,loss)]
                history = []
                history += fit(epochs, max_lr, loss, model, train_loader, val_loader, model_name=model_name,**kwargs)

                plot = pt.Plots(history, model_name)
                plot.plot_accuracies()
                plot.plot_losses()
                plot.plot_lrs()

                U.logger(model_name, history,
                         {'Dataset': dataset, 'Model': f'{model._get_name()}', "DA": 'None', 'max_lr': max_lr,
                          'sched': 'None', 'Optim': 'SGD'}, True)