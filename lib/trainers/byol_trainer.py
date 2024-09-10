import os
import time
import random
import math
from pathlib import Path

import numpy as np
from numpy.random import randint
from copy import deepcopy

import torch

import sys
sys.path.append('..')

import lib.models as models
import lib.networks as networks

import wandb

from .base_trainer import BaseTrainer
from lib.data.med_transforms import get_mae_pretrain_transforms, get_vis_transforms, Resize
from lib.data.med_datasets import get_train_loader, get_val_loader
from lib.tools.visualization import patches3d_to_grid, images3d_to_grid_alt
from lib.models import UNet, UNetContrastive
from monai.transforms import LoadImage, DivisiblePad, RandSpatialCropd
from lib.networks.contrastive_decoder import MLP

from lib.utils import to_3tuple
import torch.nn as nn

from report_guided_annotation import extract_lesion_candidates
from picai_eval import evaluate

from lib.data.data_generator import prepare_datagens
from picai_baseline.unet.training_setup.augmentations.nnUNet_DA import \
    apply_augmentations

class BYOLTrainer(BaseTrainer):
    r"""
    3D Contrastive Trainer
    """
    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.model_name
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss = BYOLLoss(args)
        self.initial_tau = args.initial_tau
        self.current_tau = args.initial_tau

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name} of arch {args.arch}")
            if self.model_name == 'UNet':
                print('Strides', args.model_strides)
                self.model = UNetContrastive(
                                spatial_dims=len(args.image_shape),
                                in_channels=args.num_channels,
                                out_channels=args.out_chans,
                                strides=args.model_strides,
                                channels=args.model_features,
                                c_out_channels=args.out_channels,
                                contrastive_decoder_type=args.contrastive_decoder_type
                                )
            else:
                self.model = getattr(models, self.model_name)(
                            encoder=getattr(networks, args.enc_arch), 
                            decoder_c=getattr(networks, args.dec_arch_c), 
                            decoder_r=getattr(networks, args.dec_arch_r), 
                            args=args)
            self.wrap_model()
            self.predictor = MLP(input_dim=args.out_channels, hidden_dim=args.out_channels *2, output_dim=args.out_channels).to(args.gpu)
            self.model_target = deepcopy(self.model)
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
                self.iters_per_epoch = 800 // args.batch_size
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
            elif args.dataset == 'Brats2021':
                transforms = [Resize((-1,128,128), mode='trilinear'), DivisiblePad((-1, 16,-1,-1))]
                train_gen, val_gen, class_weights = prepare_datagens(args=args, fold_id=0, is_brats=True,
                                                             transforms=transforms, seg_transforms=transforms)
                """
                self.dataloader = apply_augmentations(
                                dataloader=train_gen,
                                num_threads=args.workers,
                                disable=False
                                )
                """
                self.dataloader = train_gen
                self.val_dataloader = val_gen
                self.iters_per_epoch = 800 // args.batch_size
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
        model_target = self.model_target
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
                    #target = batch_data['seg'].cuda(args.gpu, non_blocking=True)
                except Exception:
                    image = torch.from_numpy(batch_data['data']).cuda(args.gpu, non_blocking=True)
                    #target = torch.from_numpy(batch_data['seg']).cuda(args.gpu, non_blocking=True)
            else:
                image = batch_data['image']
            
                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)

            # compute output and loss
            forward_start_time = time.time()
            if args.augmentation == 'DAE':
                x1 = image
                x2 = image
                x1_aug = self.noise(image)
                x2_aug = self.noise(image) 
            if args.augmentation == 'MAE':
                x1 = image
                x1_aug = self.create_mask(image)
                x2_aug = self.create_mask(image)  
            with torch.no_grad():
               out2_c, _ = model_target(x2_aug) 
            with torch.cuda.amp.autocast(True):
                out1_c, out1_r = model(x1_aug)
                pred_contrastive = self.predictor(out1_c)
                loss, losses = self.loss(pred_contrastive, out2_c, out1_r, x1)
            forward_time = time.time() - forward_start_time

            # compute gradient and do SGD step
            bp_start_time = time.time()
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Update Weights of target net after 
            for online_p, target_p in zip(model.parameters(), model_target.parameters()):
                target_p.data = self.current_tau * target_p.data + (1.0 - self.current_tau) * online_p.data

            # Update Tau
            max_steps = self.iters_per_epoch * args.epochs
            self.current_tau = 1 - (1 - self.initial_tau) * (math.cos(math.pi * niters / max_steps) + 1) / 2
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
                        "Contrast Loss": losses[1].item(),
                        "Recon Loss": losses[0].item()
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
            if args.augmentation == 'DAE':
                x1 = image
                x1_aug = self.noise(image)
                x2_aug = self.noise(image)

            if args.augmentation == 'MAE':
                x1 = image
                x1_aug = self.create_mask(image)
                x2_aug = self.create_mask(image)
            out1_c, out1_r = model(x1_aug)
            vis_tensor = torch.cat([image, x1_aug, out1_r, x2_aug], dim=0)

           
            vis_grid_hw = images3d_to_grid_alt(vis_tensor, n_group=4, in_chans=args.in_chans)
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
            return image, out1_c, x1_aug, vis_grid_hw, vis_grid_hw0


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
 
    def noise(self, image, return_noise=False):
        args = self.args
        n = torch.normal(mean=torch.zeros(image.shape[0], image.shape[1], args.noise_shape, args.noise_shape, args.noise_shape), std=args.noise_std).cuda(args.gpu, non_blocking=True)
        n = args.noise_multiplier*nn.functional.interpolate(n, (args.roi_x, args.roi_y, args.roi_z), mode='trilinear')
        """roll_x = random.choice(range(args.roi_x))
        roll_y = random.choice(range(args.roi_y))
        roll_z = random.choice(range(args.roi_z))
        n = torch.roll(n, shifts=[roll_x, roll_y, roll_z], dims=[-3, -2, -1])"""
        if args.dataset == 'Brats2021':
            # Como el cerebro no esta en gran parte de la imagen, agregamos ruido solo a la zona
            # del cerebro mediante una mascara
            mask = image.sum(dim=1, keepdim=True) > 0.01
            n *= mask
        if return_noise:
            return image.clone() + n, n
        return image.clone() + n
    
    def create_mask(self, image, return_mask=False, mask_percentage=0.75):
        # Obtener las dimensiones de la imagen
        batch_size, channels, depth, height, width = image.shape
        args = self.args
        # Definir el tamaño del parche
        patch_size = args.patch_size
        
        # Dividir la imagen en parches 3D
        num_patches_d = depth // patch_size
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        
        patches = image.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size).unfold(4, patch_size, patch_size)
        patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size, patch_size)
        
        # Calcular el número de parches a enmascarar
        total_patches = patches.shape[2]
        num_mask_patches = int(total_patches * mask_percentage)
        
        # Seleccionar índices aleatorios para enmascarar
        mask_indices = torch.randperm(total_patches)[:num_mask_patches]
        
        # Crear la máscara
        mask = torch.ones_like(image)
        
        # Enmascarar los parches seleccionados
        for idx in mask_indices:
            d_idx = (idx // (num_patches_h * num_patches_w)) * patch_size
            h_idx = ((idx % (num_patches_h * num_patches_w)) // num_patches_w) * patch_size
            w_idx = (idx % num_patches_w) * patch_size
            mask[:, :, d_idx:d_idx+patch_size, h_idx:h_idx+patch_size, w_idx:w_idx+patch_size] = 0
        
        # Aplicar la máscara a la imagen
        masked_image = image * mask
        
        if return_mask:
            return masked_image, mask
        else:
            return masked_image
  
    
    def test(self):
        # Esta funcion esta solo para probar cosas sin entrar al loop de entrenamiento
        # Para usarla en la config poner manual_test: true
        print('En funcion TEST')
        train_loader = self.dataloader
        model = self.wrapped_model
        model_target = deepcopy(model)
        args = self.args
        loss = BYOLLoss(args)
        #self.vis_reconstruction()
        for i, batch_data in enumerate(train_loader):
            print('Holaaaaa')
            try:
                image = batch_data['data'].to(args.gpu, non_blocking=True)
                target = batch_data['seg'].to(args.gpu, non_blocking=True)
            except Exception:
                image = torch.from_numpy(batch_data['data']).to(args.gpu, non_blocking=True)
                target = torch.from_numpy(batch_data['seg']).to(args.gpu, non_blocking=True)
            print(image.shape)   
            print(target.shape)
            x1 = image
            x1_aug = self.noise(image)
            x2_aug = self.noise(image)
            out1_c, out1_r = model(x1_aug)
            out2_c, out2_r = model_target(x2_aug)
            pred_contrastive = self.predictor(out1_c)
            l = loss(pred_contrastive, out2_c, out1_r, x1)
            print(l)
            print('Final')
            break

from torch.nn import functional as F   
class BYOLLoss(torch.nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.recon_loss = nn.MSELoss().cuda()
        self.alpha1 = args.recon_weight
        self.alpha2 = args.contrast_weight

    def __call__(self, out_online, out_target, out_recon, target_recon):
        recon_loss = self.alpha1 * self.recon_loss(out_recon, target_recon)
        contrast_loss = 2 -2 * F.cosine_similarity(out_online, out_target).mean() * self.alpha2
        total_loss = recon_loss + contrast_loss
        return total_loss, (recon_loss, contrast_loss)

