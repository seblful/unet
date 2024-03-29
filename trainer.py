from unet import UNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler

from tqdm import tqdm


class UnetTrainer():
    def __init__(self,
                 dataset,
                 val_ratio,
                 batch_size,
                 num_epochs,
                 checkpoints_path,
                 lr=1e-5,
                 weight_decay=1e-8,
                 momentum=0.9):

        self.checkpoints_path = checkpoints_path

        # Device and average mixed precision
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
                                                              patience=5,
                                                              verbose=True)  # goal: maximize Dice score
        self.criterion = nn.BCEWithLogitsLoss()

        self.scaler = GradScaler()

        self.min_loss = 9_999_999.0

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
        total_len = len(self.train_dataset) // self.batch_size
        for batch in tqdm(self.train_loader,
                          total=total_len,
                          desc=f'Epoch {epoch}/{self.num_epochs}',
                          unit='batch'):

            images, true_masks = batch['image'].to(
                self.device), batch['mask'].to(self.device)

            self.optimizer.zero_grad()

            with torch.autocast(device_type=self.device, dtype=torch.float16):

                pred_masks = self.model(images)
                # pred_masks = pred_masks > 0.5
                loss = self.criterion(pred_masks, true_masks)

            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)

            # Updates the scale for next iteration.
            self.scaler.update()

            epoch_loss += loss.item()

        final_loss = epoch_loss / len(self.train_loader)

        return final_loss

    @torch.inference_mode()
    def val_step(self):
        self.model.eval()
        epoch_loss = 0
        epoch_dice = 0
        # Iterate over the validation set
        for batch in self.val_loader:
            # Retrieve images and masks
            images, true_masks = batch['image'].to(
                self.device), batch['mask'].to(self.device)

            # Predict the mask
            pred_masks = self.model(images)
            loss = self.criterion(pred_masks, true_masks)
            dice_coeff = self.dice_coeff(F.sigmoid(pred_masks), true_masks)

            epoch_loss += loss.item()
            epoch_dice += dice_coeff.item()

        final_loss = epoch_loss / len(self.val_loader)
        final_dice = epoch_dice / len(self.val_loader)

        print(f"Validation loss: {final_loss:.3f}.")
        print(f"Validation dice: {final_dice:.3f}.")

        return final_loss

    def train(self):
        for epoch in range(1, self.num_epochs + 1):
            # Train step
            train_loss = self.train_step(epoch)
            # Val step
            val_loss = self.val_step()
            # Add loss to scheduler
            self.scheduler.step(val_loss)
            # Save checkpoint
            self.save_checkpoint(val_loss)

    def save_checkpoint(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss

            state_dict = self.model.state_dict()
            torch.save(state_dict, f"{self.checkpoints_path}/best.pth")
            print(f"Model has been saved.")
