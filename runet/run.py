import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torchvision import utils
from tqdm import tqdm
from datetime import datetime
from pytz import timezone



def get_checkpoint_name(file_type, CHECKPOINTS_FOLDER=os.path.join(os.path.dirname(os.getcwd()), 'checkpoints')):
    now = datetime.now(timezone('EST'))
    if 'best' in file_type:
      checkpoint_file = os.path.join(CHECKPOINTS_FOLDER, f'checkpoints_{file_type}.pth')
    else:
      checkpoint_file = os.path.join(CHECKPOINTS_FOLDER, f'checkpoints_{now.strftime("%m%d%Y")}_{now.strftime("%H%M%S")}_{file_type}.pth')
    return checkpoint_file

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train(model, train_loader, criterion, optimizer, epoch, log_interval=10):
    model.train()
    print(f'Epoch {epoch}\t Learning rate : {get_lr(optimizer)}')
    train_loss = 0
    with tqdm(total=len(train_loader)) as pbar:
      for batch_idx, img in enumerate(train_loader):
          data, label = img["image"], img["label"]
          data = Variable(data).float().cuda()
          label = Variable(label).float().cuda()

          optimizer.zero_grad()
          output = model(data)
          loss = criterion(output, label)
          loss.backward()
          optimizer.step()
          train_loss += loss.item()
          if (batch_idx % log_interval == 0):
            pbar.set_description('training loss {:.4f}\t'.format(loss.item()))
          pbar.update(1)
      train_loss /= len(train_loader)
      #pbar.set_description('training average loss {:.4f}\t'.format(train_loss))
    return train_loss

def test(model, test_loader, criterion, scheduler):
    model.eval()
    with torch.no_grad():
      val_loss = 0
      with tqdm(total=len(test_loader), desc="validation") as pbar:
        for batch_idx, img in enumerate(test_loader):
            data, label = img["image"], img["label"]
            data = Variable(data).float().cuda()
            label = Variable(label).float().cuda()

            output = model(data)
            val_loss += criterion(output, label).item() # sum up batch loss
            pbar.update(1)
        val_loss /= len(test_loader)
        pbar.set_description('Validation average loss {:.4f}\t'.format(val_loss))
      scheduler.step(np.around(val_loss, 4))
    return val_loss


