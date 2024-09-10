# SSL-Prostate

### Preparation

1. Create the environment with the requirements.txt file
2. Download and prepare the datasets to use. In our experiments we used [PICAI](https://pi-cai.grand-challenge.org) as the main training dataset, and [Prostate158](https://zenodo.org/records/6481141) and ChiPC (to be published) as target dataset. Then, prepare the datasets following [PICAI Baseline](https://github.com/DIAGNijmegen/picai_baseline) instructions. The splits we used for ChiPC and Prostate158 are in the splits folder, as splits.json files. We preprocessed the images to be have a shape of (32,256,256). We included the overviews files on the splits folder as well, but the paths should be changed to reflect where is your own data.

### Stage 1: SSL Pretraining
We tested MAE, DAE, SimCLR and BYOL as SSL pretraining. On the configs folder there are example configs for each of them with UNet and UNETR. For example, to run UNet DAE pretraining use the following command:
```
python main.py configs/unet_prostate_dae.yaml
```
For UNETR models, the MAE algorithm is optimized by only passing the unmasked patches to the encoder. You can use this version with the "mae3d" configs, or you can use the standard version with the unetr_prostate_mae config. For Unet models, only the standard version is available.

Remember to update the configs with your paths before running the scripts.
Wandb is used to monitor the training procedures.

### Stage 2: Segmentation training
In our expermiments we performed a main training stage on the PICAI dataset and then a finetuning stage on either ChiPC or Prostate158 dataset. To run either of them run the following command with the appropiate config file:
```
python main.py configs/unet_prostate_picai.yaml
```
If you want to use a pretrained checkpoint, you have two options:

1. Use only the encoder weights. This is what we used to transition from Stage 1 (SSL) to Stage 2 (PICAI segmentation training). To do this, put your pretrained checkpoint on the "pretrain" option on the config file.

2. Use the whole network (encoder and decoder). This is what we used for the transition to of Stage 2 -> Stage 3. To do this, put your pretrained checkpoint on the "start_from" option.

You can also resume a training with the "resume" option, and if you use this it is not necessary to use neither of the previosly mentioned options
