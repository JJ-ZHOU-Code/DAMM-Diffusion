import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from pathlib import Path
from datetime import datetime

import torch 
from torch.utils.data import ConcatDataset
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


from diffusion.data.datamodules import SimpleDataModule
from diffusion.data.datasets import Dataset_Paired
from diffusion.models.embedders.latent_embedders import VQVAE, VQGAN, VAE, VAEGAN, CLIP_VAE

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse 
parser = argparse.ArgumentParser(description='Configurations for training the latent embedding model.')

parser.add_argument('--data_name', type=str, help='Name of the dataset used for training the VAE (e.g., DAPI).')
parser.add_argument('--path_root', type=str, help='Root directory where dataset files are stored.')
parser.add_argument('--img_size', type=int, default=256, help='Image size.')
parser.add_argument('--batchsize', type=int, default=32, help='Batch size for training.')
parser.add_argument('--num_workers', type=int, default=8, help='Number of worker processes for data loading.')
args = parser.parse_args()


if __name__ == "__main__":

    # --------------- Settings --------------------
    current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    path_run_dir = Path.cwd() / 'runs' / 'VQGAN_foldk_{}_{}'.format(args.data_name, str(current_time))
    path_run_dir.mkdir(parents=True, exist_ok=True)
    gpus = [0] if torch.cuda.is_available() else None


    ds_3 = Dataset_Paired(
        path_root = args.path_root,
        image_resize = args.img_size
    )

    dm = SimpleDataModule(
        ds_train = ds_3,
        batch_size=args.batchsize, 
        num_workers=args.num_workers,
        pin_memory=True
    ) 
    

    # ------------ Initialize Model ------------
    model = VQGAN(  
        data_name=args.data_name,
        in_channels=3, 
        out_channels=3, 
        emb_channels=8,
        spatial_dims=2,
        num_embeddings = 16384, # 8192,
        hid_chs =    [ 64, 128, 256,  256, 512], # [ 64, 128, 256,  512]
        kernel_sizes=[ 3,  3,   3,    3, 3], # [ 3,  3,   3,  3]  除第一个外，其余每个下采样为原来二分之一
        strides =    [ 1,  2,   2,    2, 2], # [ 1,  2,   2,  2]
        deep_supervision=1,
        use_attention= 'none',
        # loss = torch.nn.MSELoss,
        # optimizer_kwargs={'lr':1e-6},
        embedding_loss_weight=1e-6,
        # perceiver = None,
        sample_every_n_steps=500
    )

    # -------------- Training Initialization ---------------
    to_monitor = "train/L1"  # "val/loss" 
    min_max = "min"
    save_and_sample_every = 100

    early_stopping = EarlyStopping(
        monitor=to_monitor,
        min_delta=0.0, # minimum change in the monitored quantity to qualify as an improvement
        patience=30, # number of checks with no improvement
        mode=min_max
    )
    checkpointing = ModelCheckpoint(
        dirpath=str(path_run_dir), # dirpath
        monitor=to_monitor,
        every_n_train_steps=save_and_sample_every,
        save_last=True,
        save_top_k=5,
        mode=min_max,
    )
    trainer = Trainer(
        accelerator='gpu',
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
        min_epochs=20,
        max_epochs=50,
        num_sanity_val_steps=2,
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(model, datamodule=dm)

    # ------------- Save path to best model -------------
    model.save_best_checkpoint(trainer.logger.log_dir, checkpointing.best_model_path)


