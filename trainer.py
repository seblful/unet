from unet import UNet
from dataset import CancerDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, optim
from torch.utils.data import DataLoader, random_split

from tqdm import tqdm


class UnetTrainer():
    def __init__(self,
                 dataset,
                 val_ratio,
                 batch_size,
                 num_epochs,
                 checkpoints_path,
                 amp=False,
                 lr=1e-5,
                 weight_decay=1e-8,
                 momentum=0.999,
                 gradient_clipping=1.0):

        self.checkpoints_path = checkpoints_path

        # Device and average mixed precision
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.amp = amp

        # Dataset and dataloader
        self.dataset = dataset

        self.batch_size = batch_size

        self.train_size = int(len(self.dataset) * (1 - val_ratio))
        self.val_size = len(self.dataset) - self.train_size

        self.train_dataset, self.val_dataset = random_split(
            dataset, [self.train_size, self.val_size])

        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True,
                                       num_workers=0,
                                       pin_memory=True)

        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=self.batch_size,
                                     shuffle=False,
                                     num_workers=0,
                                     pin_memory=True)

        # Model
        self.image_dim = self.train_dataset[0]['image'].shape[0]
        self.model = UNet(in_channels=self.image_dim,
                          out_channels=3).to(device=self.device,
                                             memory_format=torch.channels_last)

        # Optimizer, scheduler, loss
        self.num_epochs = num_epochs

        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=lr,
                                       weight_decay=weight_decay,
                                       momentum=momentum)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                              mode='max',
                                                              patience=5)  # goal: maximize Dice score
        self.criterion = nn.BCEWithLogitsLoss()

        self.grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)

        self.gradient_clipping = gradient_clipping

    def dice_coeff(self,
                   input,
                   target,
                   reduce_batch_first=False,
                   epsilon=1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        assert input.dim() == 3 or not reduce_batch_first

        sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)

        return dice.mean()

    def dice_loss(self, input, target):
        # Dice loss (objective to minimize) between 0 and 1
        return 1 - self.dice_coeff(input, target, reduce_batch_first=False)

    @torch.inference_mode()
    def val_step(self):
        self.model.eval()
        dice_score = 0

        # iterate over the validation set
        with torch.autocast(self.device, enabled=self.amp):
            for batch in tqdm(self.val_loader,
                              total=len(self.val_loader),
                              desc='Validation round',
                              unit='batch',
                              leave=False):
                images, true_masks = batch['image'], batch['mask']

                # move images and labels to correct device and type
                images = images.to(device=self.device,
                                   dtype=torch.float32,
                                   memory_format=torch.channels_last)
                true_masks = true_masks.to(device=self.device,
                                           dtype=torch.long)

                # predict the mask
                pred_masks = self.model(images)
                pred_masks = (F.sigmoid(pred_masks) > 0.5).float()

                # compute the Dice score
                dice_score += self.dice_coeff(pred_masks,
                                              true_masks,
                                              reduce_batch_first=False)

        val_score = dice_score / max(len(self.val_loader), 1)
        print(f"Validation loss: {val_score}.")

        self.scheduler.step(val_score)

        self.model.train()

        return

    def train_step(self, epoch):
        self.model.train()
        epoch_loss = 0
        with tqdm(total=len(self.train_dataset),
                  desc=f'Epoch {epoch}/{self.num_epochs}',
                  unit='img') as pbar:
            for batch in self.train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=self.device,
                                   dtype=torch.float32,
                                   memory_format=torch.channels_last)

                true_masks = true_masks.to(
                    device=self.device, dtype=torch.long)

                with torch.autocast(self.device, enabled=self.amp):
                    pred_masks = self.model(images)
                    loss = self.criterion(
                        pred_masks.squeeze(1), true_masks.float())
                    loss += self.dice_loss(F.sigmoid(pred_masks.squeeze(1)),
                                           true_masks.float())

                self.optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.gradient_clipping)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                pbar.update(images.shape[0])
                epoch_loss += loss.item()
                pbar.set_postfix(**{'loss (batch)': loss.item()})

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            # Train step
            self.train_step(epoch)
            # Val step
            self.val_step()
            # Save checkpoint
            self.save_checkpoint(epoch)

    def save_checkpoint(self, epoch):
        state_dict = self.model.state_dict()
        torch.save(
            state_dict, f"{self.checkpoints_path}/checkpoint_epoch{epoch}.pth")
        print(f"Model 'checkpoint_epoch {epoch}.pth' was saved.")
