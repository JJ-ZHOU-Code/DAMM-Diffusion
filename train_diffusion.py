import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from email.mime import audio
from pathlib import Path
from datetime import datetime

import torch 
import torch.nn as nn
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np 
# import torchio as tio 

from diffusion.data.datamodules import SimpleDataModule
from diffusion.data.datasets import Dataset_Paired
from diffusion.external.stable_diffusion.unet_openai import UNetModel
from diffusion.models.noise_schedulers import GaussianNoiseScheduler
from diffusion.models.embedders import LabelEmbedder, TimeEmbbeding
from diffusion.models.embedders.latent_embedders import VAE, CLIP_VAE, CLIP_Encoder, VQGAN
from diffusion.models.pipelines.diffusion_pipeline import DiffusionPipeline
from diffusion.models.estimators.unet import UNet_DAMM
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse 
parser = argparse.ArgumentParser(description='Configurations for training the diffusion model.')

### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--path_root', type=str, help='Root directory where dataset files are stored.')
parser.add_argument('--batchsize', type=int, default=32, help='Batch size for training.')
parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes for data loading.')

parser.add_argument('--img_size', type=int, default=256, help='Image size.')

### Checkpoint Paths
parser.add_argument('--ckp_dapi', type=str, help='Path to the checkpoint file for the DAPI model.')
parser.add_argument('--ckp_vessels', type=str, help='Path to the checkpoint file for the vessels model.')
parser.add_argument('--ckp_nuclei', type=str, help='Path to the checkpoint file for the nuclei model.')

args = parser.parse_args()

if __name__ == "__main__":
    # ------------ Load Data ----------------
    ds = Dataset_Paired(
        path_root = args.path_root,
        image_resize = args.img_size
    )

    dm = SimpleDataModule(
        ds_train = ds,
        batch_size=args.batchsize, 
        num_workers=args.num_workers,
        pin_memory=True,
        # weights=ds.get_weights()
    )
    
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / 'foldk_{}'.format(args.fold_idx) / 'diffusion_DAMM_{}'.format(str(current_time))
    path_run_dir.mkdir(parents=True, exist_ok=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    # ------------ Initialize Model ------------
    cond_embedder = None 
    # cond_embedder = LabelEmbedder
    cond_embedder_kwargs = {
        'emb_dim': 1024,
        'num_classes': 2
    }
 
    time_embedder = TimeEmbbeding
    time_embedder_kwargs ={
        'emb_dim': 1024 # stable diffusion uses 4*model_channels (model_channels is about 256)
    }


    noise_estimator = UNet_DAMM
    noise_estimator_kwargs = {
        'in_ch':8, 
        'out_ch':8, 
        'spatial_dims':2,
        'hid_chs': [ 64, 128, 256, 512], # ,[ 256, 256, 512, 1024]
        'kernel_sizes':[3, 3, 3, 3],
        'strides':     [1, 2, 2, 2],
        'time_embedder':time_embedder,
        'time_embedder_kwargs': time_embedder_kwargs,
        'cond_embedder':cond_embedder,
        'cond_embedder_kwargs': cond_embedder_kwargs,
        'deep_supervision': False,
        'use_res_block':True,
        'use_attention':'none',
    }


    # ------------ Initialize Noise ------------
    noise_scheduler = GaussianNoiseScheduler
    noise_scheduler_kwargs = {
        'timesteps': 1000,
        'beta_start': 0.002, # 0.0001, 0.0015
        'beta_end': 0.02, # 0.01, 0.0195
        'schedule_strategy': 'scaled_linear'
    }
    
    # ------------ Initialize Latent Space  ------------
    latent_embedder = VQGAN
    latent_embedder_checkpoint = args.ckp_dapi

    latent_embedder_CD31 = VQGAN
    latent_embedder_CD31_checkpoint = args.ckp_vessels

    latent_embedder_DAPI = VQGAN
    latent_embedder_DAPI_checkpoint = args.ckp_nuclei

    # ------------ Initialize Pipeline ------------
    pipeline = DiffusionPipeline(
        noise_estimator=noise_estimator, 
        noise_estimator_kwargs=noise_estimator_kwargs,
        noise_scheduler=noise_scheduler, 
        noise_scheduler_kwargs = noise_scheduler_kwargs,
        latent_embedder=latent_embedder,
        latent_embedder_checkpoint = latent_embedder_checkpoint,
        latent_embedder_CD31 = latent_embedder_CD31,
        latent_embedder_CD31_checkpoint = latent_embedder_CD31_checkpoint,
        latent_embedder_DAPI = latent_embedder_DAPI,
        latent_embedder_DAPI_checkpoint = latent_embedder_DAPI_checkpoint,
        estimator_objective='x_T',
        estimate_variance=False, 
        use_self_conditioning=False, 
        use_ema=False,
        classifier_free_guidance_dropout=0.5, # Disable during training by setting to 0
        do_input_centering=False,
        clip_x0=False,
        sample_every_n_steps=1000,
        use_img_conditioning=True,
        num_samples = 2,
        #img_condition_num=1,
        img_condition_name=['CD31','DAPI']  # 'CD31','DAPI'
    )
    
    # pipeline_old = pipeline.load_from_checkpoint('runs/2022_11_27_085654_chest_diffusion/last.ckpt')
    # pipeline.noise_estimator.load_state_dict(pipeline_old.noise_estimator.state_dict(), strict=True)

    # -------------- Training Initialization ---------------
    to_monitor = "train/loss"  # "pl/val_loss" 
    min_max = "min"
    save_and_sample_every = 1000

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=24, # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        save_top_k=5,
        mode=min_max,
        # every_n_train_steps=save_and_sample_every,
        save_last=True,
        # every_n_epochs=20,
        save_on_train_epoch_end=True
    )
    trainer = Trainer(
        accelerator=accelerator,
        # devices=[0],
        strategy="ddp",
        # precision=16,
        # amp_backend='apex',
        # amp_level='O2',
        # gradient_clip_val=0.5,
        default_root_dir=str(path_run_dir),
        callbacks=[checkpointing],
        # callbacks=[checkpointing, early_stopping],
        enable_checkpointing=True,
        check_val_every_n_epoch=1,
        log_every_n_steps=save_and_sample_every, 
        auto_lr_find=False,
        # limit_train_batches=1000,
        limit_val_batches=0, # 0 = disable validation - Note: Early Stopping no longer available 
        min_epochs=100,
        max_epochs=200,
        num_sanity_val_steps=2,
        #fast_dev_run=True # 允许断点调试
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(pipeline, datamodule=dm)

    # ------------- Save path to best model -------------
    pipeline.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)


