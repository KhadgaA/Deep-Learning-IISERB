import numpy as np
import torch
import matplotlib.pyplot as plt
import utils as U
from torchvision.utils import make_grid


class Plots():
    def __init__(self, history, model_name=None):
        self.history = history
        self.model_name = model_name

    def plot_accuracies(self):
        accuracies = [x['val_acc'] for x in self.history]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(accuracies)
        ax.set_xlabel('epoch')
        ax.set_ylabel('accuracy')
        ax.set_title('Accuracy vs. No. of epochs');
        fig.savefig(f'accuracy_plot{self.model_name}')
        # plt.show()

    def plot_losses(self):
        train_losses = [x.get('train_loss') for x in self.history]
        val_losses = [x['val_loss'] for x in self.history]
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.plot(train_losses, 'b')
        ax.plot(val_losses, 'r')
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend(['Training', 'Validation'])
        ax.set_title('Loss vs. No. of epochs');
        fig.savefig(f'loss_plot{self.model_name}')
        # plt.show()

    def plot_lrs(self):
        fig,ax = plt.subplots(figsize=(5,3))
        lrs = np.concatenate([x.get('lrs', []) for x in self.history])
        ax.plot(lrs)
        ax.set_xlabel('Batch no.')
        ax.set_ylabel('Learning rate')
        ax.set_title('Learning Rate vs. Batch no.');
        fig.savefig(f'lr_plot{self.model_name}')
        # plt.show()


def predict_image(img, model, train_ds):
    # Convert to a ba   tch of 1
    xb = U.to_device(img.unsqueeze(0), U.get_default_device())
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds = torch.max(yb, dim=1)
    # Retrieve the class label
    return train_ds.classes[preds[0].item()]

def denormalize(images, means, stds,device = torch.device('cuda')):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    means = means.to(device)
    stds = stds.to(device)
    out = images * stds + means
    out = out.cpu()
    return out

def show_batch(dl,stats):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([]); ax.set_yticks([])
        denorm_images = denormalize(images, *stats)
        f =  make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0,1)
        ax.imshow(f)
        fig.savefig('batch_image')
        break
