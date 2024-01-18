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
                          out_channels=1).to(device=self.device,
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
                   predicted,
                   target,
                   eps=1e-6):
        predicted = predicted.view(self.batch_size, -1)
        target = target.view(self.batch_size, -1)
        intersection = (predicted * target).sum(dim=1)
        dice_coeff = (2. * intersection + eps) / \
            (predicted.sum(dim=1) + target.sum(dim=1) + eps)

        return dice_coeff.mean()

    def dice_loss(self, input, target):
        return 1 - self.dice_coeff(input, target)

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
                    print("Masks shapes: ", pred_masks.shape, true_masks.shape)
                    loss = self.criterion(pred_masks, true_masks.float())
                    dice_loss = self.dice_loss(
                        F.sigmoid(pred_masks), true_masks.float())
                    print("Losses: ", loss, dice_loss)
                    loss += dice_loss

                self.optimizer.zero_grad(set_to_none=True)
                self.grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.gradient_clipping)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                epoch_loss += loss.item()

                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

    @torch.inference_mode()
    def val_step(self):
        self.model.eval()
        dice_score = 0

        # Iterate over the validation set
        with torch.autocast(self.device, enabled=self.amp):
            for batch in self.val_loader:
                # Retrieve images and masks
                images, true_masks = batch['image'], batch['mask']

                # Move images and labels to correct device and type
                images = images.to(device=self.device,
                                   dtype=torch.float32,
                                   memory_format=torch.channels_last)
                true_masks = true_masks.to(device=self.device,
                                           dtype=torch.long)

                # Predict the mask
                pred_masks = self.model(images)
                pred_masks = (F.sigmoid(pred_masks) > 0.5).float()

                # Compute the Dice score
                dice_score += self.dice_coeff(pred_masks,
                                              true_masks)

        val_score = dice_score / max(len(self.val_loader), 1)
        print(f"Validation score: {val_score}.")

        self.scheduler.step(val_score)

        self.model.train()

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
