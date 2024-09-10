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
from lib.data.med_transforms import get_mae_pretrain_transforms, get_vis_transforms, Resize
from lib.data.med_datasets import get_train_loader, get_val_loader
from lib.tools.visualization import patches3d_to_grid, images3d_to_grid_alt
from lib.models import UNet
from monai.transforms import LoadImage, DivisiblePad

from lib.utils import to_3tuple
import torch.nn as nn

from report_guided_annotation import extract_lesion_candidates
from picai_eval import evaluate

from lib.data.data_generator import prepare_datagens
from picai_baseline.unet.training_setup.augmentations.nnUNet_DA import \
    apply_augmentations

class DenoiseTrainer(BaseTrainer):
    r"""
    3D Denoise Autoencoder Trainer
    """
    def __init__(self, args):
        super().__init__(args)
        self.model_name = args.model_name
        self.scaler = torch.cuda.amp.GradScaler()
        self.loss = nn.MSELoss()

    def build_model(self):
        if self.model_name != 'Unknown' and self.model is None:
            args = self.args
            print(f"=> creating model {self.model_name} of arch {args.arch}")
            if self.model_name == 'UNet':
                print('Strides', args.model_strides)
                self.model = UNet(
                                spatial_dims=len(args.image_shape),
                                in_channels=args.num_channels,
                                out_channels=args.out_chans,
                                strides=args.model_strides,
                                channels=args.model_features
                                )
            else:
                self.model = getattr(models, self.model_name)(
                            encoder=getattr(networks, args.enc_arch), 
                            decoder=getattr(networks, args.dec_arch), 
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
                train_gen, _, class_weights = prepare_datagens(args=args, fold_id=0)
                args.overviews_dir = args.overviews_valid_dir
                _, valid_gen, _ = prepare_datagens(args=args, fold_id=0)
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
                self.iters_per_epoch = 200
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
                    if args.calc_metrics:
                        metrics = self.evaluate(niters=niters)
            # train for one epoch
            niters = self.epoch_train(epoch, niters)
            #niters += 1
            if epoch == 0 or (epoch + 1) % args.vis_freq == 0:
                print(f"=> start visualizing after {epoch + 1} epochs")
                self.vis_reconstruction(niters)
                print("=> finish visualizing")
                if args.calc_metrics:
                    m = self.evaluate(niters=niters, epoch=epoch)
                    if m.score > metrics.score:
                        metrics = m
                        print('New val best metric, saving model')
                        self.save_checkpoint({
                            'epoch': epoch + 1,
                            'arch': args.arch,
                            'state_dict': self.model.state_dict(),
                            'optimizer' : self.optimizer.state_dict(),
                            'scaler': self.scaler.state_dict(), # additional line compared with base imple
                        }, is_best=False, filename=f'{args.ckpt_dir}/best_model.pth.tar')
                  
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
                    target = batch_data['seg'].cuda(args.gpu, non_blocking=True)
                except Exception:
                    image = torch.from_numpy(batch_data['data']).cuda(args.gpu, non_blocking=True)
                    target = torch.from_numpy(batch_data['seg']).cuda(args.gpu, non_blocking=True)
            else:
                image = batch_data['image']
            
                if args.gpu is not None:
                    image = image.cuda(args.gpu, non_blocking=True)

            # compute output and loss
            forward_start_time = time.time()
            noise_img = self.noise(image)
            if args.dataset == 'Brats2021':
                image = self.hide_tumor_slices(image, target)
                noise_img = self.hide_tumor_slices(noise_img, target)
            with torch.cuda.amp.autocast(True):
                output = model(noise_img)
                loss = self.loss(output, image)
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
                        },
                        step=niters,
                    )

            niters += 1
            load_start_time = time.time()
            if step >= self.iters_per_epoch:
                break
        return niters
    
    @torch.no_grad()
    def evaluate(self, niters=0, epoch=0, return_img = False):
        args = self.args
        model = self.wrapped_model
        val_loader = self.val_dataloader

        model.eval()
        all_valid_preds, all_valid_labels = [], []
        for i, batch_data in enumerate(val_loader):
            if args.gpu is not None:
                try:
                    image = batch_data['data'].to(args.gpu, non_blocking=True)
                    target = batch_data['seg']
                except Exception:
                    image = torch.from_numpy(batch_data['data']).to(args.gpu, non_blocking=True)
                    target = torch.from_numpy(batch_data['seg'])
            if target.max() < 0.5:
                continue
            with torch.cuda.amp.autocast(True):
                output = model(image)
            
            err = (image - output).abs().mean(dim=1, keepdim=True).cpu().numpy() # Mean es en los canales, asi se ve el error promedio de T2, ADC y DWI
            all_valid_preds += [err]
            all_valid_labels += [target.cpu().numpy()]
            
            if return_img:
                return image, output, err, target

        final_det = np.concatenate([x for x in np.array(all_valid_preds)], axis=0).squeeze(1)
        final_target = np.concatenate([x for x in np.array(all_valid_labels)], axis=0).squeeze(1)
        valid_metrics = evaluate(y_det=iter(final_det),
                         y_true=iter(final_target),
                         y_det_postprocess_func=lambda pred: extract_lesion_candidates(pred)[0])
        
        print('Metrics normal: ', valid_metrics)
        if not args.disable_wandb:
            wandb_log_dict = {}
            wandb_log_dict['AUROC'] = valid_metrics.auroc
            wandb_log_dict['AP'] = valid_metrics.AP
            wandb.log(wandb_log_dict, step=niters)
        return valid_metrics
            
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
            noise_img, n = self.noise(image, return_noise=True)
            output = model(noise_img)
            vis_tensor = torch.cat([image, n, noise_img, output], dim=0)

           
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
            return image, output, noise_img, vis_grid_hw, vis_grid_hw0


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
    
    def hide_tumor_slices(self, image, label):
        # La idea es que se escondan las slices que posean un tumor. Esto es para solo mostrarle al
        # modelo tejidos sanos
        args = self.args
        zeros = torch.zeros(image.shape).to(args.gpu)
        indices = torch.max(label.reshape(label.shape[0], label.shape[2], -1), axis=2).values > 0
        indices = indices.unsqueeze(1).repeat(1, args.in_chans, 1).unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,128,128)
        mod_img = torch.where(indices, zeros, image)
        return mod_img

    
    def test(self):
        # Esta funcion esta solo para probar cosas sin entrar al loop de entrenamiento
        # Para usarla en la config poner manual_test: true
        print('En funcion TEST')
        train_loader = self.dataloader
        args = self.args
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

            mod_img = self.hide_tumor_slices(image, target)
            for i in range(target.shape[2]):
                print(image[0,0,i,:,:].max(),mod_img[0,0,i,:,:].max())
            break
        