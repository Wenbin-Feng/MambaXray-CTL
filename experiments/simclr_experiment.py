import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from loss_functions.nt_xent import NTXentLoss
import os
import shutil
import sys
import pickle
import torch.optim as optim
from datasets.two_dim.NumpyDataLoader  import ImageDataset,SimCLRTransform
from torch.utils.data import DataLoader
#from datasets.two_dim.NumpyDataLoader import NumpyDataSet
#from datasets.two_dim.NumpyDataLoader import JPEGDataSet
from networks.unet_con import GlobalConUnet, MLP

apex_support = False

import numpy as np
from arm.Finetuning.models_mamba import arm_base_pz16, arm_large_pz16
from arm.Finetuning.util.pos_embed import interpolate_pos_embed
torch.manual_seed(0)
def _save_config_file(model_checkpoints_folder, config_path):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy(config_path, os.path.join(model_checkpoints_folder, 'config.yaml'))


class SimCLR(object):

    def __init__(self, config):
        self.config = config
        self.device = self._get_device()
        self.writer = SummaryWriter(os.path.join(self.config['save_dir'], 'tensorboard'))
        self.nt_xent_criterion = NTXentLoss(self.device, **config['loss'])

        split_dir = os.path.join(self.config["base_dir"], "splits.pkl")
        data_dir = self.config["base_dir"]
        print("root",data_dir)
        # with open(split_dir, "rb") as f:
        #     splits = pickle.load(f)
        # tr_keys = splits[0]['train'] + splits[0]['val'] + splits[0]['test']
        # val_keys = splits[0]['val']
        # self.train_loader = JPEGDataSet(data_dir, target_size=self.config["img_size"], batch_size=self.config["batch_size"],
        #                         keys=tr_keys, do_reshuffle=True, mode='simclr')
        # self.val_loader = JPEGDataSet(data_dir, target_size=self.config["img_size"], batch_size=self.config["val_batch_size"],
        #                       keys=val_keys, do_reshuffle=True, mode='simclr')
        simclr_transform = SimCLRTransform(size=224)
        data_dir = self.config['data_dir']
        self.train_loader = DataLoader(ImageDataset(base_dir=data_dir, transform=simclr_transform, pattern='*.png'), batch_size=self.config["batch_size"], shuffle=True, num_workers=4, drop_last=True)
        self.val_loader = DataLoader(ImageDataset(base_dir=data_dir, transform=simclr_transform, pattern='*.png'), batch_size=self.config["batch_size"], shuffle=True, num_workers=4, drop_last=True)

        print(len(self.train_loader))
        #self.model = GlobalConUnet()
        #vision_mamba，在里面空参构造,输入（batch,3,224,224）
        self.model = arm_base_pz16()
        #加载参数
        checkpoint = torch.load(self.config['pretrained_checkpoint'], map_location='cpu')
        checkpoint_model = checkpoint['model']
        new_dict = {}
        for k, v in checkpoint_model.items():
            if "conv1d" in k:
                new_dict[k.replace("conv1d", "conv1d_b")] = v
                new_dict[k.replace("conv1d", "conv1d_c")] = v
                new_dict[k.replace("conv1d", "conv1d_c_b")] = v
            if "dt_proj" in k:
                new_dict[k.replace("dt_proj", "dt_proj_b")] = v
                new_dict[k.replace("dt_proj", "dt_proj_c")] = v
                new_dict[k.replace("dt_proj", "dt_proj_c_b")] = v
            if "x_proj" in k:
                new_dict[k.replace("x_proj", "x_proj_b")] = v
                new_dict[k.replace("x_proj", "x_proj_c")] = v
                new_dict[k.replace("x_proj", "x_proj_c_b")] = v
            if "A" in k:
                new_dict[k.replace("A", "A_b")] = v
                new_dict[k.replace("A", "A_c")] = v
                new_dict[k.replace("A", "A_c_b")] = v
            if "D" in k:
                new_dict[k.replace("D", "D_b")] = v
                new_dict[k.replace("D", "D_c")] = v
                new_dict[k.replace("D", "D_c_b")] = v
            if "dec" not in k:
                new_dict[k] = v     
        #new_dict = interpolate_pos_embed(self.model, new_dict) 
        new_dict = {}
        self.model.load_state_dict(new_dict, strict=False)         
        self.head = MLP(num_class=128)

        self.nt_xent_criterion = NTXentLoss(self.device, **config['loss'])

        # dist.init_process_group(backend='nccl')
        if torch.cuda.device_count() > 1:
            print("Let's use %d GPUs" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model,device_ids=[0, 1, 2])
            self.head = nn.DataParallel(self.head,device_ids=[0, 1, 2])

        self.model.to(self.device)
        self.head.to(self.device)

        self.model = self._load_pre_trained_weights(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 3e-4, weight_decay=eval(self.config['weight_decay']))

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Running on:", device)
        return device
    def _step(self, model, head, xis, xjs, n_iter):

        # get the representations and the projections
        #print(xis.shape)
        ris = model(xis)  # [N,C]
        #print(ris.shape)
        zis = head(ris)
        #print(zis.shape)
        # get the representations and the projections
        rjs = model(xjs)  # [N,C]
        zjs = head(rjs)

        # normalize projection feature vectors
        zis = F.normalize(zis, dim=1)
        zjs = F.normalize(zjs, dim=1)

        # loss = self.nt_xent_criterion(zis, zjs)
        loss = self.nt_xent_criterion(zis,zjs)
        return loss
    def train(self):

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder, self.config['config_path'])

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=len(self.train_loader), eta_min=0,
                                                                    last_epoch=-1)

        for epoch_counter in range(self.config['epochs']):
            print("=====Training Epoch: %d =====" % epoch_counter)
            for i, (xi, xj)in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                # 获取增强后的图像对
                xis = xi.float().to(self.device)
                xjs = xj.float().to(self.device)

                loss = self._step(self.model, self.head, xis, xjs, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print("Train:[{0}][{1}][{2}] loss: {loss:.4f}".format(epoch_counter, i, len(self.train_loader),
                                                                          loss=loss.item()))

                loss.backward()
                self.optimizer.step()
                n_iter += 1

            print("===== Validation =====")
            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(self.val_loader)
                print("Val:[{0}] loss: {loss:.4f}".format(epoch_counter, loss=valid_loss))
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(self.config['save_dir'],
                                                                     'b_{}_model.pth'.format(self.config["batch_size"])))
                    print("save_url:", os.path.join(self.config['save_dir'],))
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                scheduler.step()
            self.writer.add_scalar('cosine_lr_decay', scheduler.get_lr()[0], global_step=n_iter)

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./runs', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, valid_loader):

        # validation steps
        with torch.no_grad():
            self.model.eval()

            valid_loss = 0.0
            counter = 0
            for (xi, xj) in valid_loader:
                xis = xi.float().to(self.device)
                xjs = xj.float().to(self.device)

                loss = self._step(self.model, self.head, xis, xjs, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= (counter)
        return valid_loss


