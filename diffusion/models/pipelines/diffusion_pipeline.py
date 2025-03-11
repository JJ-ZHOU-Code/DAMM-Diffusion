

from pathlib import Path 
from tqdm import tqdm

import torch 
import torch.nn.functional as F 
from torchvision.utils import save_image 
import streamlit as st

from diffusion.models import BasicModel
from diffusion.utils.train_utils import EMAModel
from diffusion.utils.math_utils import kl_gaussians
# from pytorch_lightning.utilities.cloud_io import load as pl_load
from collections import OrderedDict
from torchmetrics.functional import multiscale_structural_similarity_index_measure as mmssim
from torchmetrics.functional import structural_similarity_index_measure as ssim
from torchmetrics.functional import peak_signal_noise_ratio as psnr

class Identity(torch.nn.Module):
    def encode(self, x):
        return x 
    def decode(self, x):
        return x



class DiffusionPipeline(BasicModel):
    def __init__(self, 
        noise_scheduler,
        noise_estimator,
        latent_embedder=None,
        noise_scheduler_kwargs={},
        noise_estimator_kwargs={},
        latent_embedder_checkpoint='',
        latent_embedder_CD31 = None,
        latent_embedder_CD31_checkpoint = '',
        latent_embedder_DAPI = None,
        latent_embedder_DAPI_checkpoint = '',
        estimator_objective = 'x_T', # 'x_T' or 'x_0'
        estimate_variance=False, 
        use_self_conditioning=False, 
        classifier_free_guidance_dropout=0.5, # Probability to drop condition during training, has only an effect for label-conditioned training 
        num_samples = 4,
        do_input_centering = True, # Only for training
        clip_x0=True, # Has only an effect during traing if use_self_conditioning=True, import for inference/sampling  
        use_ema = False,
        ema_kwargs = {},
        optimizer=torch.optim.AdamW, 
        optimizer_kwargs={'lr':1e-4}, # stable-diffusion ~ 1e-4
        lr_scheduler= None, # stable-diffusion - LambdaLR
        lr_scheduler_kwargs={}, 
        loss=torch.nn.L1Loss,
        loss_kwargs={},
        sample_every_n_steps = 2000,
        use_img_conditioning=True,
        img_condition_num=1,
        img_condition_name=['CD31'],
        uncertain_gamma = 0.5,
        pretrain_epoch = 80
        ):
        # self.save_hyperparameters(ignore=['noise_estimator', 'noise_scheduler']) 
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        self.pretrain_epoch = pretrain_epoch
        self.microbatch = 1
        self.alpha = 1e-4
        self.gamma = uncertain_gamma
        self.loss_fct_s = loss(**loss_kwargs)
        self.loss_fct_m = loss(**loss_kwargs)
        self.sample_every_n_steps=sample_every_n_steps

        self.img_condition_name = img_condition_name
        self.img_condition_num = len(img_condition_name)
        self.use_img_conditioning = self.img_condition_num > 0

        noise_estimator_kwargs['estimate_variance'] = estimate_variance
        noise_estimator_kwargs['use_self_conditioning'] = use_self_conditioning
        noise_estimator_kwargs['use_img_conditioning'] = self.use_img_conditioning
        noise_estimator_kwargs['img_condition_num'] = self.img_condition_num

        self.noise_scheduler = noise_scheduler(**noise_scheduler_kwargs)
        self.noise_estimator = noise_estimator(**noise_estimator_kwargs)
                
        self.latent_embedder = self.instantiate_first_stage(latent_embedder, latent_embedder_checkpoint)

        self.cond_latent_embedder = {}
        if 'CD31' in self.img_condition_name:
            self.latent_embedder_CD31 = self.instantiate_first_stage(latent_embedder_CD31, latent_embedder_CD31_checkpoint)
            self.cond_latent_embedder['CD31'] = self.latent_embedder_CD31
        if 'DAPI' in self.img_condition_name:
            self.latent_embedder_DAPI = self.instantiate_first_stage(latent_embedder_DAPI, latent_embedder_DAPI_checkpoint)
            self.cond_latent_embedder['DAPI'] = self.latent_embedder_DAPI

        self.estimator_objective = estimator_objective
        self.use_self_conditioning = use_self_conditioning
        # self.use_img_conditioning = use_img_conditioning
        # self.img_condition_num = img_condition_num
        self.num_samples = num_samples
        self.classifier_free_guidance_dropout = classifier_free_guidance_dropout
        self.do_input_centering = do_input_centering
        self.estimate_variance = estimate_variance
        self.clip_x0 = clip_x0

        self.use_ema = use_ema
        if use_ema:
            self.ema_model = EMAModel(self.noise_estimator, **ema_kwargs)
    
    @torch.no_grad()
    def instantiate_first_stage(self, latent_embedder=None, latent_embedder_checkpoint=None):
        if latent_embedder is not None:
            latent_embedder = latent_embedder.load_from_checkpoint(latent_embedder_checkpoint)
            for param in latent_embedder.parameters():
                param.requires_grad = False
        else:
            latent_embedder = Identity() 
        return latent_embedder

    def load_preweights(self, pretrained_weights, strict=False, **kwargs):
        filter = kwargs.get('filter', lambda key:key in pretrained_weights)
        init_weights = self.noise_estimator.unet_S.state_dict()
        pretrained_weights = {key.replace('noise_estimator.',''): value for key, value in pretrained_weights.items() if key.startswith('noise_estimator')}
        init_weights.update(OrderedDict(pretrained_weights))
        self.noise_estimator.unet_S.load_state_dict(init_weights, strict=strict)
        
        init_weights = self.noise_estimator.time_embedder.state_dict()
        pretrained_weights = {key.replace('time_embedder.',''): value for key, value in pretrained_weights.items() if key.startswith('time_embedder')}
        init_weights.update(OrderedDict(pretrained_weights))
        self.noise_estimator.time_embedder.load_state_dict(init_weights, strict=strict)
        # for param_name, param_value in self.noise_estimator.unet_S.state_dict().items():
        #     print(f"Parameter: {param_name}, Value: {param_value}")

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx:int):
        results = {}
        x_0 = batch['source']
        xc_CD31 = batch['CD31']
        xc_DAPI = batch['DAPI']
        condition = batch.get('target', None) 

        if self.latent_embedder is not None:
            self.latent_embedder.eval() 
            with torch.no_grad():
                x_0 = self.latent_embedder.encode(x_0)

        # ------------ Image condition -------------------
        img_cond=[]
        for cond_name in self.img_condition_name:
            cond_batch = batch[cond_name]
            embedder = self.cond_latent_embedder[cond_name]
            embedder.eval()
            with torch.no_grad():
                c = embedder.encode(cond_batch)
            img_cond.append(c)

        if self.do_input_centering:
            x_0 = 2*x_0-1 # [0, 1] -> [-1, 1]

        # if self.clip_x0:
        #     x_0 = torch.clamp(x_0, -1, 1)
        
        # Sample Noise
        with torch.no_grad():
            # Randomly selecting t [0,T-1] and compute x_t (noisy version of x_0 at t)
            x_t, x_T, t = self.noise_scheduler.sample(x_0) 
                
        # Use EMA Model
        if self.use_ema and (state != 'train'):
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Re-estimate x_T or x_0, self-conditioned on previous estimate 
        self_cond = None 
        if self.use_self_conditioning:
            with torch.no_grad():
                pred, pred_vertical = noise_estimator(x_t, t, condition, None) 
                if self.estimate_variance:
                    pred, _ =  pred.chunk(2, dim = 1)  # Seperate actual prediction and variance estimation 
                if self.estimator_objective == "x_T": # self condition on x_0 
                    self_cond = self.noise_scheduler.estimate_x_0(x_t, pred, t=t, clip_x0=self.clip_x0)
                elif self.estimator_objective == "x_0": # self condition on x_T 
                    self_cond = self.noise_scheduler.estimate_x_T(x_t, pred, t=t, clip_x0=self.clip_x0)
                else:
                    raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")

        # Classifier free guidance 
        if torch.rand(1)<self.classifier_free_guidance_dropout:
            condition = None 

        # Separate variance (scale) if it was learned 
        if self.estimate_variance:
            pred, pred_var =  pred.chunk(2, dim = 1)  # Separate actual prediction and variance estimation 

        # Specify target 
        if self.estimator_objective == "x_T":
            target = x_T 
        elif self.estimator_objective == "x_0":
            target = x_0 
        else:
            raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")

        # ------------------------- Compute Loss ---------------------------
        loss = 0
        loss_u = 0
        c_value_epoch = 0
        corr_epoch = 0
        loss_c_epoch, loss_m_epoch = 0, 0

        if self.current_epoch < self.pretrain_epoch: # 
            pretrain = True
            o_CD31, o_MM, w_i, confi_map = noise_estimator(x_t, t, img_cond, use_uncertain=False) 
            loss_c = self.loss_fct_s(o_CD31, target)
            loss_m = self.loss_fct_m(o_MM, target)
            loss = (loss_c + loss_m) / 2
            pred = o_MM

            loss_c_epoch = loss_c 
            loss_m_epoch = loss_m 
        else:
            pretrain = False
            self.microbatch = 1
            pred = torch.zeros_like(x_0)
            c_pred = torch.zeros(x_0.shape[0], device=self.device)
            c_label = torch.zeros(x_0.shape[0], device=self.device)
            for i in range(0, x_0.shape[0], self.microbatch):
                target_micro = target[i : i + self.microbatch]
                img_cond_micro = [img[i:i+self.microbatch] for img in img_cond]
                x_t_micro = x_t[i : i+self.microbatch]
                t_micro = t[i: i+self.microbatch]
                o_CD31, o_MM, w_i, confi_map = noise_estimator(x_t_micro, t_micro, img_cond_micro, use_uncertain=True) # u_map值越低，置信度越低

                c_value = torch.mean(confi_map.to(torch.float32)) # confidence score: (b,)
                c_pred[i] = c_value
                # o_MM = u_penalty*o_MM + (1-u_penalty)*target_micro
                # corr = torch.sum(w_i)

                loss_c = self.loss_fct_s(o_CD31, target_micro)
                loss_m = self.loss_fct_m(o_MM, target_micro)

                # c_label[i] = 1 - loss_m/(loss_m+loss_c)
                if loss_m <= loss_c: 
                    c_label[i] = 1 - c_label[i]

                if c_value < self.gamma: 
                    loss = loss + loss_c
                    pred[i:i+self.microbatch] = o_CD31
                else: 
                    loss = loss + (loss_c + loss_m)/2
                    pred[i:i+self.microbatch] = o_MM
                
                c_value_epoch = c_value_epoch + c_value # -torch.log(u_penalty)
                loss_c_epoch += loss_c 
                loss_m_epoch += loss_m 

                # corr_epoch = corr_epoch + torch.mean(corr)
            loss_u = F.binary_cross_entropy(c_pred, c_label)*0.5 + torch.mean((c_pred-self.gamma)**2)
            # loss_u = - (c_label * torch.log(c_pred))

            # loss = (loss + u_penalty_loss*1e-3) / x_0.shape[0]
            c_value_epoch = c_value_epoch / x_0.shape[0]
            loss_c_epoch, loss_m_epoch = loss_c_epoch/x_0.shape[0], loss_m_epoch/x_0.shape[0]
            # corr_epoch = corr_epoch / x_0.shape[0]
            loss = loss / x_0.shape[0] + self.alpha*loss_u

        # ----------------- MSE/L1, ... ----------------------
        results['loss']  = loss
        # self.log('loss_u', self.alpha*loss_u, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log('c', c_value_epoch, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log('loss_c', loss_c_epoch, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        self.log('loss_m', loss_m_epoch, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        # self.log('corr', corr_epoch, prog_bar=True, logger=True, on_step=True, on_epoch=False)
       
        # ------------------------ Compute Metrics  --------------------------
        with torch.no_grad():
            results['L2'] = F.mse_loss(pred, target)
            results['L1'] = F.l1_loss(pred, target)
            # results['SSIM'] = SSIMMetric(data_range=pred.max()-pred.min(), spatial_dims=source.ndim-2)(pred, target)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in results.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x_0.shape[0], on_step=True, on_epoch=True)           
        
        #------------------ Log Image -----------------------
        # if self.global_step != 0 and self.global_step % self.sample_every_n_steps == 0:
        if self.global_rank==0 and self.global_step % self.sample_every_n_steps == 0: # and self.global_step != 0 : # self.global_step != 0 and 
            dataformats =  'NHWC' if x_0.ndim == 5 else 'HWC'
            def norm(x):
                return (x-x.min())/(x.max()-x.min())

            sample_cond = condition[0:self.num_samples] if condition is not None else None
            # img_cond = img_cond[0:self.num_samples] if img_cond is not None else None
            img_cond = [img[:self.num_samples] for img in img_cond]
            img_src = batch['source'][0:self.num_samples]
            sample_img = {}
            
            sample_img['CD31'] = self.sample(num_samples=self.num_samples, img_size=x_0.shape[1:],\
                 condition=sample_cond, img_cond=img_cond, pretrain=pretrain, target_pred='CD31').detach()
            sample_img['MM'] = self.sample(num_samples=self.num_samples, img_size=x_0.shape[1:],\
                 condition=sample_cond, img_cond=img_cond, pretrain=pretrain, target_pred='MM').detach()
            if pretrain == False:
                sample_img['UMM'] = self.sample(num_samples=self.num_samples, img_size=x_0.shape[1:],\
                    condition=sample_cond, img_cond=img_cond, pretrain=pretrain, target_pred='UMM').detach()

            log_step = self.global_step // self.sample_every_n_steps
            # self.logger.experiment.add_images("predict_img", norm(torch.moveaxis(pred[0,-1:], 0,-1)), global_step=self.current_epoch, dataformats=dataformats) 
            # self.logger.experiment.add_images("target_img", norm(torch.moveaxis(target[0,-1:], 0,-1)), global_step=self.current_epoch, dataformats=dataformats) 
            
            # self.logger.experiment.add_images("source_img", norm(torch.moveaxis(x_0[0,-1:], 0,-1)), global_step=log_step, dataformats=dataformats) 
            # self.logger.experiment.add_images("sample_img", norm(torch.moveaxis(sample_img[0,-1:], 0,-1)), global_step=log_step, dataformats=dataformats) 
            
            path_out = Path(self.logger.log_dir)/'images'
            path_out.mkdir(parents=True, exist_ok=True)

            if pretrain:
                images = torch.cat([img_src, sample_img['CD31'], sample_img['MM']])
            else: 
                images = torch.cat([img_src, sample_img['CD31'], sample_img['MM'], sample_img['UMM']])
            
            save_image(images, path_out/f'sample_{log_step}_N.png', nrow=self.num_samples ,normalize=True)
            save_image(images, path_out/f'sample_{log_step}.png', nrow=self.num_samples)        
        
        return loss

    
    def forward(self, x_t, t, condition=None, self_cond=None, img_cond=None, guidance_scale=1.0, cold_diffusion=False, un_cond=None, pretrain=False, target_pred='MM'):
        # Note: x_t expected to be in range ~ [-1, 1]
        if self.use_ema:
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Concatenate inputs for guided and unguided diffusion as proposed by classifier-free-guidance
        pred = {}
        if pretrain:
            o_CD31, o_MM, w_i, confi_map =  noise_estimator(x_t, t, img_cond=img_cond, use_uncertain=False)
            pred['CD31'], pred['MM'] = o_CD31, o_MM 
        else:
            pred['CD31'] = torch.zeros_like(x_t)
            pred['MM'] = torch.zeros_like(x_t)
            pred['UMM'] = torch.zeros_like(x_t)
            
            for i in range(0, x_t.shape[0], self.microbatch):
                img_cond_micro = [img[i:i+self.microbatch] for img in img_cond]
                x_t_micro = x_t[i : i+self.microbatch]
                t_micro = t[i: i+self.microbatch]
                o_CD31, o_MM, w_i, confi_map = noise_estimator(x_t_micro, t_micro, img_cond_micro,use_uncertain=True) # u_map值越低，置信度越低

                c_value = torch.mean(confi_map.to(torch.float32)) # uncertainty score: (b,)
                # corr = torch.sum(w_i)
                # print(corr, end='\r')

                if c_value < self.gamma: # u_penalty < self.gamma: # 置信度confidence很低
                    pred['UMM'][i:i+self.microbatch] = o_CD31
                else: 
                    pred['UMM'][i:i+self.microbatch] = o_MM
                    
                pred['CD31'][i:i+self.microbatch] = o_CD31
                pred['MM'][i:i+self.microbatch] = o_MM

        pred = pred[target_pred]

        if self.estimate_variance:
            pred, pred_var =  pred.chunk(2, dim = 1)  

        if self.estimate_variance:
            pred_var_scale = pred_var/2+0.5 # [-1, 1] -> [0, 1]
            pred_var_value = pred_var  
        else:
            pred_var_scale = 0
            pred_var_value = None 

        # pred_var_scale = pred_var_scale.clamp(0, 1)

        if  self.estimator_objective == 'x_0':
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_0(x_t, t, pred, clip_x0=self.clip_x0, var_scale=pred_var_scale, cold_diffusion=cold_diffusion)
            x_T = self.noise_scheduler.estimate_x_T(x_t, x_0=pred, t=t, clip_x0=self.clip_x0)
            self_cond = x_T 
        elif self.estimator_objective == 'x_T':
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_T(x_t, t, pred, clip_x0=self.clip_x0, var_scale=pred_var_scale, cold_diffusion=cold_diffusion)
            x_T = pred 
            self_cond = x_0 
        else:
            raise ValueError("Unknown Objective")
        
        return x_t_prior, x_0, x_T, self_cond 


    @torch.no_grad()
    def denoise(self, x_t, steps=None, condition=None, img_cond=None, use_ddim=True, pretrain=False, target_pred='MM', **kwargs):
        self_cond = None 

        # ---------- run denoise loop ---------------
        if use_ddim:
            steps = self.noise_scheduler.timesteps if steps is None else steps
            timesteps_array = torch.linspace(0, self.noise_scheduler.T-1, steps, dtype=torch.long, device=x_t.device) # [0, 1, 2, ..., T-1] if steps = T 
        else:
            timesteps_array = self.noise_scheduler.timesteps_array[slice(0, steps)] # [0, ...,T-1] (target time not time of x_t)
            
        #st_prog_bar = st.progress(0)
        for i, t in enumerate(tqdm(reversed(timesteps_array))):
            #st_prog_bar.progress((i+1)/len(timesteps_array))

            # UNet prediction 
            x_t, x_0, x_T, self_cond = self(x_t, t.expand(x_t.shape[0]), condition, self_cond=self_cond, \
                            img_cond=img_cond, pretrain=pretrain,target_pred=target_pred, **kwargs)
            self_cond = self_cond if self.use_self_conditioning else None  
        
            if use_ddim and (steps-i-1>0):
                t_next = timesteps_array[steps-i-2]
                alpha = self.noise_scheduler.alphas_cumprod[t]
                alpha_next = self.noise_scheduler.alphas_cumprod[t_next]
                sigma = kwargs.get('eta', 1) * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(x_t)
                x_t = x_0 * alpha_next.sqrt() + c * x_T + sigma * noise

        # ------ Eventually decode from latent space into image space--------
        if self.latent_embedder is not None:
            x_t = self.latent_embedder.decode(x_t)
        
        return x_t # Should be x_0 in final step (t=0)

    @torch.no_grad()
    def sample(self, num_samples, img_size, condition=None, img_cond=None, pretrain=False,target_pred='MM', **kwargs):
        template = torch.zeros((num_samples, *img_size), device=self.device)
        x_T = self.noise_scheduler.x_final(template)
        x_0 = self.denoise(x_T, condition=condition, img_cond=img_cond, pretrain=pretrain,target_pred=target_pred, **kwargs)
        return x_0 


    @torch.no_grad()
    def interpolate(self, img1, img2, i = None, condition=None, lam = 0.5, **kwargs):
        assert img1.shape == img2.shape, "Image 1 and 2 must have equal shape"

        t = self.noise_scheduler.T-1 if i is None else i
        t = torch.full(img1.shape[:1], i, device=img1.device)

        img1_t = self.noise_scheduler.estimate_x_t(img1, t=t, clip_x0=self.clip_x0)
        img2_t = self.noise_scheduler.estimate_x_t(img2, t=t, clip_x0=self.clip_x0)

        img = (1 - lam) * img1_t + lam * img2_t
        img = self.denoise(img, i, condition, **kwargs)
        return img

    def on_train_batch_end(self, *args, **kwargs):
        lr = self.optimizers().param_groups[0]['lr']
        # self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        if self.use_ema:
            self.ema_model.step(self.noise_estimator)
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.noise_estimator.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = {
                'scheduler': self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs),
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]