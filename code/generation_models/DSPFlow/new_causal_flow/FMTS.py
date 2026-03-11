import torch
import torch.nn.functional as F
from torch import nn
from einops import reduce
from .transformer import Transformer
import os
from .vqvae import VQVAE
from tqdm import tqdm


class CausalFlow(nn.Module):
    def __init__(
            self,
            seq_length,
            feature_size,
            proto_dim=2048,
            n_layer_enc=3,
            n_layer_dec=6,
            d_model=None,
            n_heads=4,
            mlp_hidden_times=4,
            attn_pd=0.,
            resid_pd=0.,
            kernel_size=None,
            padding_size=None,
    ):
        super(CausalFlow, self).__init__()

        self.seq_length = seq_length
        self.feature_size = feature_size

        self.model = Transformer(
            n_feat=feature_size, n_channel=seq_length, n_layer_enc=n_layer_enc, n_layer_dec=n_layer_dec,
            n_heads=n_heads, attn_pdrop=attn_pd, resid_pdrop=resid_pd, mlp_hidden_times=mlp_hidden_times,
            max_len=seq_length, n_embd=d_model, conv_params=[kernel_size, padding_size],
            proto_dim=proto_dim,
        )

        self.alpha = 3  ## t shifting, change to 1 is the uniform sampling during inference
        self.time_scalar = 1000 ## scale 0-1 to 0-1000 for time embedding

        self.num_timesteps = int(os.environ.get('hucfg_num_steps', '100'))


    def freeze_proto_mlp(self):
        for name, param in self.model.named_parameters():
            if 'proto_mlp' in name:
                param.requires_grad = False
        print('Frozen proto_mlp')


    def output(self, x, t, history, prototypes, self_attn_masks, cross_attn_masks):
        breakpoint()
        if self_attn_masks is not None:
            history = history * self_attn_masks.unsqueeze(-1)
        output = self.model(
            x, t,
            history,
            prototypes,
            self_attn_masks,
            cross_attn_masks)

        return output

    # @torch.no_grad()
    # def sample(self, shape, prototype_id, padding_masks):
    #     model_device = next(self.parameters()).device
    #     if prototype_id != -100:
    #         prototypes = prototype_id * torch.ones(shape).mean((1, 2)).to(dtype=torch.long, device=model_device)
    #         prototype_embed = self.prototype_embedding(prototypes)
    #     else:
    #         prototype_embed = None
    #     self.eval()
    #     zt = torch.randn(shape).to(model_device)  ## init the noise
    #     ## t shifting from stable diffusion 3
    #     timesteps = torch.linspace(0, 1, self.num_timesteps+1)
    #     t_shifted = 1-(self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
    #     t_shifted = t_shifted.flip(0)
    #     for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
    #         step = t_prev - t_curr
    #         t_input = torch.tensor([t_curr*self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).to(model_device).view(-1)
    #         v = self.output(zt.clone(), t_input, prototype_embed, padding_masks)
    #         zt = zt.clone() + step * v
    #     return zt


    # @torch.no_grad()
    # def impute(self, signals, missing_signals, attn_mask, noise_mask):
    #     """
    #     x_start: (B, T, C)
    #     anomaly_label: (B, T, C)   1 = missing, 0 = observed
    #     """
    #     self.eval()
    #
    #     batch_size = signals.shape[0]
    #     prototype_embeds = self.vqvae.encode(missing_signals)
    #     prototype_embeds = prototype_embeds.reshape(batch_size, -1)
    #
    #
    #     zt = torch.randn_like(signals) * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))
    #
    #     # --- identical timestep shifting as unconditional sample ---
    #     timesteps = torch.linspace(0, 1, self.num_timesteps + 1)
    #     t_shifted = 1 - (self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
    #     t_shifted = t_shifted.flip(0).to(signals.device)
    #
    #     # 2) Integrate ODE from t=1 → t=0
    #     for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
    #         step = t_prev - t_curr
    #         t_input = torch.tensor([t_curr*self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).to(signals.device).view(-1)
    #         v = self.output(zt.clone(), t_input, prototype_embeds, attn_mask)
    #
    #         #update missing region ONLY
    #         zt = zt + step * v * noise_mask.unsqueeze(-1)
    #         #restore known region
    #         zt = zt * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))
    #
    #     return zt


    @torch.no_grad()
    def posterior_impute(self, signals, posterior, attn_mask, noise_mask):
        """
        x_start: (B, T, C)
        anomaly_label: (B, T, C)   1 = missing, 0 = observed
        """
        self.eval()

        batch_size = signals.shape[0]
        prototype_embeds = posterior.reshape(batch_size, -1)


        zt = torch.randn_like(signals) * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))

        # --- identical timestep shifting as unconditional sample ---
        timesteps = torch.linspace(0, 1, self.num_timesteps + 1)
        t_shifted = 1 - (self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0).to(signals.device)

        # 2) Integrate ODE from t=1 → t=0
        for t_curr, t_prev in tqdm(zip(t_shifted[:-1], t_shifted[1:])):
            step = t_prev - t_curr
            t_input = torch.tensor([t_curr*self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).to(signals.device).view(-1)
            v = self.output(zt.clone(), t_input, prototype_embeds, attn_mask)

            #update missing region ONLY
            zt = zt + step * v * noise_mask.unsqueeze(-1)
            #restore known region
            zt = zt * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))

        return zt


    @torch.no_grad()
    def no_code_impute(self, signals, attn_mask, noise_mask):
        """
        x_start: (B, T, C)
        anomaly_label: (B, T, C)   1 = missing, 0 = observed
        """
        self.eval()

        # batch_size = signals.shape[0]
        # prototype_embeds = posterior.reshape(batch_size, -1)


        zt = torch.randn_like(signals) * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))

        # --- identical timestep shifting as unconditional sample ---
        timesteps = torch.linspace(0, 1, self.num_timesteps + 1)
        t_shifted = 1 - (self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0).to(signals.device)

        # 2) Integrate ODE from t=1 → t=0
        for t_curr, t_prev in tqdm(zip(t_shifted[:-1], t_shifted[1:])):
            step = t_prev - t_curr
            t_input = torch.tensor([t_curr*self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).to(signals.device).view(-1)
            v = self.output(zt.clone(), t_input, None, attn_mask)

            #update missing region ONLY
            zt = zt + step * v * noise_mask.unsqueeze(-1)
            #restore known region
            zt = zt * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))

        return zt


    def no_context_generation(self, signals, attn_mask):
        self.eval()

        batch_size = signals.shape[0]
        prototype_embeds = self.vqvae.encode(signals, attn_mask)
        prototype_embeds = prototype_embeds.reshape(batch_size, -1)

        zt = torch.randn_like(signals)

        # --- identical timestep shifting as unconditional sample ---
        timesteps = torch.linspace(0, 1, self.num_timesteps + 1)
        t_shifted = 1 - (self.alpha * timesteps) / (1 + (self.alpha - 1) * timesteps)
        t_shifted = t_shifted.flip(0).to(signals.device)

        # 2) Integrate ODE from t=1 → t=0
        for t_curr, t_prev in zip(t_shifted[:-1], t_shifted[1:]):
            step = t_prev - t_curr
            t_input = torch.tensor([t_curr * self.time_scalar]).unsqueeze(0).repeat(zt.shape[0], 1).to(
                signals.device).view(-1)
            v = self.output(zt.clone(), t_input, prototype_embeds, attn_mask)

            # update missing region ONLY
            zt = zt + step * v
            # restore known region
            zt = zt * attn_mask.unsqueeze(-1)

        return zt


    def forward(self, batch, mode):
        if mode=="uncond":
            return self._uncond_loss(signals=batch, attn_mask=None)

        elif mode=="cond":
            return self._cond_loss(batch)

        else:
            raise NotImplementedError("No such mode")


    def _no_context_loss(self, signals, attn_mask):
        # here we only take signals and attn_mask, we do discrete-code conditioned generation without context
        # to unify the length, we padded the signals in the dataset, this is why we need attn_mask
        batch_size = signals.shape[0]
        with torch.no_grad():
            prototype_embeds = self.vqvae.encode(signals, attn_mask)
        prototype_embeds = prototype_embeds.reshape(batch_size, -1)

        z0 = torch.randn_like(signals)
        z1 = signals

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)

        z_t = t * z1 + (1. - t) * z0
        target = z1 - z0
        model_out = self.output(
            z_t,
            t.view(-1) * self.time_scalar,
            prototype_embeds,
            padding_masks=attn_mask)

        # -------- length-aware mask --------
        # mask: [B, T, 1]
        B, T, C = signals.shape
        lengths = attn_mask.sum(1)
        # -------- masked MSE --------
        loss = (model_out - target) ** 2
        loss = loss.sum(2)
        loss = loss * attn_mask.to(dtype=torch.float32)
        # normalize by valid length
        loss = loss.sum(dim=1) / (lengths.float() * C)
        loss = loss.mean()

        return loss


    def _uncond_loss(self, signals, attn_mask):
        # here we only take signals and attn_mask, we do discrete-code conditioned generation without context
        # to unify the length, we padded the signals in the dataset, this is why we need attn_mask
        # batch_size = signals.shape[0]
        # with torch.no_grad():
            # prototype_embeds = self.vqvae.encode(signals)
        # prototype_embeds = prototype_embeds.reshape(batch_size, -1)

        z0 = torch.randn_like(signals)
        z1 = signals

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)

        z_t = t * z1 + (1. - t) * z0
        target = z1 - z0
        model_out = self.output(
            z_t,
            t.view(-1) * self.time_scalar,
            prototypes=None,
            padding_masks=attn_mask)

        # -------- length-aware mask --------
        # mask: [B, T, 1]
        B, T, C = signals.shape
        if attn_mask is not None:
            lengths = attn_mask.sum(1)
            # -------- masked MSE --------
            loss = (model_out - target) ** 2
            loss = loss.sum(2)
            loss = loss * attn_mask.to(dtype=torch.float32)
            # normalize by valid length
            loss = loss.sum(dim=1) / (lengths.float() * C)
            loss = loss.mean()
        else:
            loss = (model_out - target) ** 2
            loss = loss.mean()

        return loss


    def _cond_loss(self, batch):

        history = batch['history']
        signals = batch['target']
        attn_mask = None

        z0 = torch.randn_like(signals)
        z1 = signals

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)

        z_t = t * z1 + (1. - t) * z0
        target = z1 - z0
        breakpoint()
        model_out = self.output(
            z_t,
            t.view(-1) * self.time_scalar,
            history,
            batch['text_embed'],
            batch["attn_mask"],
            batch["cross_attn_mask"]
        )

        # -------- length-aware mask --------
        # mask: [B, T, 1]

        loss = (model_out - target) ** 2
        loss = loss.mean()

        return loss


    def _imputation_loss(self, signals, missing_signals, missing_signals_mask, attn_mask, noise_mask):

        batch_size = signals.shape[0]
        with torch.no_grad():
            prototype_embeds = self.vqvae.encode(missing_signals, missing_signals_mask)
        prototype_embeds = prototype_embeds.reshape(batch_size, -1)

        z0 = torch.randn_like(signals) * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))
        z1 = signals

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)

        z_t = t * z1 + (1. - t) * z0
        target = (z1 - z0) * noise_mask.unsqueeze(-1)
        model_out = self.output(
            z_t,
            t.view(-1) * self.time_scalar,
            prototype_embeds,
            padding_masks=attn_mask)
        model_out = model_out * noise_mask.unsqueeze(-1)

        # -------- masked MSE --------
        loss = (model_out - target) ** 2
        loss = loss.mean(-1)

        # 只对 anomaly 部分计算误差
        masked_loss = loss * noise_mask  # (B, T)
        # 每个样本 anomaly 的数量
        num_anomalies = reduce(noise_mask, 'b t -> b 1', 'sum')  # shape: (B, 1)

        # 每个样本的 loss = sum(masked_loss) / num_anomalies
        loss_per_sample = reduce(masked_loss, 'b t -> b 1', 'sum') / num_anomalies

        # 最终 batch loss = mean over batch
        return loss_per_sample.mean()


    def _no_code_imputation_loss(self, signals, attn_mask, noise_mask):

        # batch_size = signals.shape[0]
        # with torch.no_grad():
        #     prototype_embeds = self.vqvae.encode(missing_signals)
        # prototype_embeds = prototype_embeds.reshape(batch_size, -1)


        z0 = torch.randn_like(signals) * noise_mask.unsqueeze(-1) + signals * (1 - noise_mask.unsqueeze(-1))
        z1 = signals

        t = torch.rand(z0.shape[0], 1, 1).to(z0.device)
        if str(os.environ.get('hucfg_t_sampling', 'uniform')) == 'logitnorm':
            t = torch.sigmoid(torch.randn(z0.shape[0], 1, 1)).to(z0.device)

        z_t = t * z1 + (1. - t) * z0
        target = (z1 - z0) * noise_mask.unsqueeze(-1)
        model_out = self.output(
            z_t,
            t.view(-1) * self.time_scalar,
            prototypes=None,
            padding_masks=attn_mask)
        model_out = model_out * noise_mask.unsqueeze(-1)

        # -------- masked MSE --------
        loss = (model_out - target) ** 2
        loss = loss.mean(-1)

        # 只对 anomaly 部分计算误差
        masked_loss = loss * noise_mask  # (B, T)
        # 每个样本 anomaly 的数量
        num_anomalies = reduce(noise_mask, 'b t -> b 1', 'sum')  # shape: (B, 1)

        # 每个样本的 loss = sum(masked_loss) / num_anomalies
        loss_per_sample = reduce(masked_loss, 'b t -> b 1', 'sum') / num_anomalies

        # 最终 batch loss = mean over batch
        return loss_per_sample.mean()

