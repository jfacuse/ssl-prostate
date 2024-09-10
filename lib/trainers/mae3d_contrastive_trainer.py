import os
import time
import random
from pathlib import Path

import numpy as np

import torch

import sys
sys.path.append('..')

import lib.models as models
import lib.networks as networks

import wandb

from .base_trainer import BaseTrainer
from lib.data.med_transforms import get_mae_pretrain_transforms, get_vis_transforms
from lib.data.med_datasets import get_train_loader, get_val_loader
from lib.tools.visualization import patches3d_to_grid
from monai.transforms import LoadImage

from itertools import repeat
import collections.abc
# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

from lib.data.data_generator import prepare_datagens
from picai_baseline.unet.training_setup.augmentations.nnUNet_DA import \
    apply_augmentations

class MAE3DContrastiveTrainer(BaseTrainer):
    r"""
    3D Masked Autoencoder Trainer
    """
    def __init__(self, args):
        super().__init__(args)
        self.model_name = 'MAE3DContrastive'
        self.scaler = torch.cuda.amp.GradScaler()
        self.contrast_loss = Contrast(args, args.batch_size).cuda()

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name} of arch {args.arch}")
            self.model = getattr(models, self.model_name)(
                            encoder=getattr(networks, args.enc_arch), 
                            decoder=getattr(networks, args.dec_arch),
                            decoder_c=getattr(networks, args.dec_arch_c),
                            args=args)
            self.wrap_model()
        elif self.model_name == 'Unknown':
            raise ValueError("=> Model name is still unknown")
        else:
            raise ValueError("=> Model has been created. Do not create twice")
        
    def build_optimizer(self):
        assert(self.model is not None and self.wrapped_model is not None), \
                "Model is not created and wrapped yet. Please create model first."
        print("=> creating optimizer")
        args = self.args

        optim_params = self.get_parameter_groups()
        # TODO: create optimizer factory
        self.optimizer = torch.optim.AdamW(optim_params, 
                                            lr=args.lr,
                                            betas=(args.beta1, args.beta2),
                                            weight_decay=args.weight_decay)

    def build_dataloader(self):
        if self.dataloader is None:
            print("=> creating dataloader")
            args = self.args
            if args.dataset == 'prostate' and args.picai_dataloader:
                train_gen, valid_gen, class_weights = prepare_datagens(args=args, fold_id=0)
                self.dataloader = apply_augmentations(
                                dataloader=train_gen,
                                num_threads=args.workers,
                                disable=False
                                )
                self.val_dataloader = valid_gen
                self.iters_per_epoch = 200
            elif args.dataset in ['btcv', 'msd_brats', 'prostate']:
                train_transform = get_mae_pretrain_transforms(args)
                self.dataloader = get_train_loader(args, 
                                                   batch_size=self.batch_size,
                                                   workers=self.workers,
                                                   train_transform=train_transform)
                val_transform = get_vis_transforms(args)
                self.val_dataloader = get_val_loader(args, 
                                                     batch_size=args.vis_batch_size,
                                                     workers=self.workers,
                                                     val_transform=val_transform)
                self.iters_per_epoch = len(self.dataloader)
            elif args.dataset == 'brats20':
                # TODO
                raise NotImplementedError("brats20 transforms and dataloaders on MONAI has not been implemented yet.")
            else:
                raise ValueError("Currently only support brats2020 dataset")

            
            print(f"==> Length of train dataloader is {self.iters_per_epoch}")
        else:
            raise ValueError(f"Dataloader has been created. Do not create twice.")
        print("=> finish creating dataloader")
    
    def run(self):
        args = self.args
        # Compute iterations when resuming
        niters = args.start_epoch * self.iters_per_epoch

        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                self.dataloader.sampler.set_epoch(epoch)
                torch.distributed.barrier()

            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if epoch == args.start_epoch:
                    print("==> First visualization")
                    self.vis_reconstruction(niters)
            # train for one epoch
            niters = self.epoch_train(epoch, niters)
            #niters += 1
            if epoch == 0 or (epoch + 1) % args.vis_freq == 0:
                print(f"=> start visualizing after {epoch + 1} epochs")
                self.vis_reconstruction(niters)
                print("=> finish visualizing")
            
          
            if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
                if epoch == 0 or (epoch + 1) % args.save_freq == 0:
                    print(f"=> start saving checkpoint after epoch {epoch + 1}")
                    self.save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': self.model.state_dict(),
                        'optimizer' : self.optimizer.state_dict(),
                        'scaler': self.scaler.state_dict(), # additional line compared with base imple
                    }, is_best=False, filename=f'{args.ckpt_dir}/checkpoint_{epoch:04d}.pth.tar')
                    print("=> finish saving checkpoint")

    def epoch_train(self, epoch, niters):
        args = self.args
        train_loader = self.dataloader
        model = self.wrapped_model
        optimizer = self.optimizer
        scaler = self.scaler

        # switch to train mode
        model.train()
        step = 0
        load_start_time = time.time()
        for i, batch_data in enumerate(train_loader):
            step += 1
            load_time = time.time() - load_start_time
            # adjust learning at the beginning of each iteration
            self.adjust_learning_rate(epoch + i / self.iters_per_epoch, args)

            # For SSL pretraining, only image data is required for training
            if args.picai_dataloader:
                try:
                    image = batch_data['data'].cuda(args.gpu, non_blocking=True)
                except Exception:
                    image = torch.from_numpy(batch_data['data']).cuda(args.gpu, non_blocking=True)
            else:
                image = batch_data['image']
            
                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)

            # compute output and loss
            forward_start_time = time.time()
            with torch.cuda.amp.autocast(True):
                loss_reconstruct1, embedding1 = model(image, return_image=False, return_embedding=True)
                loss_reconstruct2, embedding2 = model(image, return_image=False, return_embedding=True)
                loss_reconstruct = (loss_reconstruct1+ loss_reconstruct2) /2
                loss_contrast = self.contrast_loss(embedding1, embedding2)
                loss = loss_reconstruct + loss_contrast
            forward_time = time.time() - forward_start_time

            # compute gradient and do SGD step
            bp_start_time = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            bp_time = time.time() - bp_start_time

            # Log to the screen
            if i % args.print_freq == 0:
                print(f"Epoch: {epoch:03d}/{args.epochs} | "
                      f"Iter: {i:05d}/{self.iters_per_epoch} | "
                      f"TotalIter: {niters:06d} | "
                      f"Init Lr: {self.lr:.05f} | "
                      f"Lr: {optimizer.param_groups[0]['lr']:.05f} | "
                      f"Load Time: {load_time:.03f}s | "
                      f"Forward Time: {forward_time:.03f}s | "
                      f"Backward Time: {bp_time:.03f}s | "
                      f"Loss: {loss.item():.03f}")
                if args.rank == 0 and not args.disable_wandb:
                    wandb.log(
                        {
                        "lr": optimizer.param_groups[0]['lr'],
                        "Loss": loss.item(),
                        "Contrast Loss": loss_contrast.item(),
                        "Recon Loss": loss_reconstruct.item(),
                        },
                        step=niters,
                    )

            niters += 1
            load_start_time = time.time()
            if step >= self.iters_per_epoch:
                break
        return niters

    def vis_reconstruction(self, niters=0, return_images=False):
        args = self.args
        loader = self.val_dataloader
        model = self.wrapped_model

        model.eval()

        for batch_data in loader:
            if args.picai_dataloader:
                axis = 'h'
                try:
                    image = batch_data['data'].cuda(args.gpu, non_blocking=True)
                except Exception:
                    image = torch.from_numpy(batch_data['data']).cuda(args.gpu, non_blocking=True)
            else:
                axis = 'd'
                image = batch_data['image']
                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)

            # compute output and loss
            """
            test_img = image[0][1]
            test_img2 = test_img.permute(2,0,1)
            print('initial shape', image.shape)
            test_img_prostate = '/mnt/workspace/jfacuse/prostate/workdir/nnUNet_raw_data/Task2208_picai_prostate158/imagesTr/10000_1000000.nii.gz'
            data2 = LoadImage(image_only=True, ensure_channel_first=False, simple_keys=True)(test_img_prostate)
            #print('data2 shape', data2.shape, data2)
            data2 = data2[1].permute(2,0,1)[12]
            v_min = data2.min()
            v_max = data2.max()
            nmin, nmax = 0, 255
            v_p = (data2 - v_min)/(v_max - v_min)*(nmax - nmin) + nmin
            print('DATA', v_p)
            test_data2_w =wandb.Image(v_p, caption="Data2Image")
            test_img = test_img2[12].cpu()
            v_min2 = test_img.min()
            v_max2 = test_img.max()
            nmin2, nmax2 = 0, 255
            v_p2 = (test_img - v_min2)/(v_max2 - v_min2)*(nmax2 - nmin2) + nmin2
            print('TEST', v_p2)
            test_img_w = wandb.Image(v_p2, caption="TestImage")
            #print(test_img2[12].cpu())
            #print(v_p)"""
            _, x, recon, masked_x = model(image, return_image=True)
            vis_tensor = torch.cat([x, masked_x, recon], dim=0)

            # visualize
            print('ImageShape',image.shape)
            grid_size = []
            for pa_size, in_size in zip(to_3tuple(args.patch_size), [image.shape[2], image.shape[3], image.shape[4]]):
                grid_size.append(in_size // pa_size)
            vis_grid_hw = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis=axis)
            # import pdb
            # pdb.set_trace()
            # vis_grid_hd = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis='w')
            # vis_grid_wd = patches3d_to_grid(vis_tensor, patch_size=args.patch_size, grid_size=grid_size, in_chans=args.in_chans, hidden_axis='h')
            print(vis_grid_hw.shape)
            print("wandb logging")
            vis_grid_hw0 = wandb.Image(vis_grid_hw[0].cpu(), caption=f"hw_iter{niters:06d}")
            
            # vis_grid_hd = wandb.Image(vis_grid_hd, caption=f"hd_iter{niters:06d}")
            # vis_grid_wd = wandb.Image(vis_grid_wd, caption=f"wd_iter{niters:06d}")

            wandb.log(
                {
                "vis_hw0": vis_grid_hw0,
                #"test_image": test_img_w,
               # "data_image": test_data2_w
                # "vis_hd": vis_grid_hd,
                # "vis_wd": vis_grid_wd
                },
                step=niters,
            )
            break
        print("finish wandb logging")
        if return_images:
            return x, recon, masked_x, vis_grid_hw, vis_grid_hw0


    def resume(self):
        args = self.args
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scaler.load_state_dict(checkpoint['scaler']) # additional line compared with base imple
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    def test(self):
        # Esta funcion esta solo para probar cosas sin entrar al loop de entrenamiento
        # Para usarla en la config poner manual_test: true
        print('En funcion TEST')
        x, recon, masked_x, vis_grid_hw, vis_grid_hw0 = self.vis_reconstruction(return_images=True)
        print('X:', x)
        print('Recon:', recon)
        print('MaskedX:', masked_x)
        print('VisGrid:', vis_grid_hw)
        print('Visgrid0:', vis_grid_hw0)
        np.save('/home/jfacuse/SelfMedMAE/test_grid3.npy', vis_grid_hw)

from torch.nn import functional as F
class Contrast(torch.nn.Module):
    def __init__(self, args, batch_size, temperature=0.5):
        super().__init__()
        device = args.gpu
        self.batch_size = batch_size
        self.register_buffer("temp", torch.tensor(temperature).to(args.gpu))
        self.register_buffer("neg_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())

    def forward(self, x_i, x_j):
        x_i = x_i.flatten(start_dim=1, end_dim=2) # NO ESTOY SEGURO DE ESTA PARTE...
        x_j = x_j.flatten(start_dim=1, end_dim=2)
        z_i = F.normalize(x_i, dim=1)
        z_j = F.normalize(x_j, dim=1)
        z = torch.cat([z_i, z_j], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
        sim_ij = torch.diag(sim, self.batch_size)
        sim_ji = torch.diag(sim, -self.batch_size)
        pos = torch.cat([sim_ij, sim_ji], dim=0)
        nom = torch.exp(pos / self.temp)
        denom = self.neg_mask * torch.exp(sim / self.temp)
        return torch.sum(-torch.log(nom / torch.sum(denom, dim=1))) / (2 * self.batch_size)