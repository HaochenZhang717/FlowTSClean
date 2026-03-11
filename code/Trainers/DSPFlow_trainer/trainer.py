import torch
import wandb
import os
import copy
from tqdm import tqdm


class DSPFlowTrainer(object):
    def __init__(
            self, optimizer,
            model, train_loader,
            val_loader, max_epochs,
            device, save_dir,
            wandb_project_name, wandb_run_name,
            grad_clip_norm,
            grad_accum_steps,
            compile=True
    ):
        self.optimizer = optimizer
        self.model = model.to(device)

        # -------------------------
        # EMA MODEL
        # -------------------------
        self.ema_decay = 0.999
        self.ema_model = copy.deepcopy(self.model).to(device)
        self.ema_model.requires_grad_(False)

        # -------------------------
        # compile only main model
        # -------------------------
        if compile:
            if torch.cuda.is_available():
                try:
                    print("Compiling model...")
                    self.model = torch.compile(self.model)
                except Exception as e:
                    print("Compile failed, fallback to eager mode")
                    print(e)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.max_epochs = max_epochs
        self.device = device
        self.save_dir = save_dir
        self.wandb_project_name = wandb_project_name
        self.wandb_run_name = wandb_run_name
        self.grad_clip_norm = grad_clip_norm
        self.grad_accum_steps = grad_accum_steps



    def update_ema(self):
        with torch.no_grad():
            for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
                ema_p.mul_(self.ema_decay).add_(model_p, alpha=1 - self.ema_decay)


    def uncond_train(self, config):

        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config=config,
        )

        os.makedirs(self.save_dir, exist_ok=True)

        best_val_loss = float("inf")
        no_improve_epochs = 0
        global_steps = 0

        model_dtype = next(self.model.parameters()).dtype

        for epoch in range(self.max_epochs):

            total_loss = 0
            tr_seen = 0

            self.model.train()
            self.optimizer.zero_grad()

            # ==========================
            # TRAIN
            # ==========================
            for batch in tqdm(self.train_loader, desc=f"Train Epoch {epoch}"):

                batch = batch[0].to(dtype=model_dtype, device=self.device)

                loss = self.model(batch, mode="uncond")

                total_loss += loss.item()
                tr_seen += 1
                global_steps += 1

                loss_backward = loss / self.grad_accum_steps
                loss_backward.backward()

                if global_steps % self.grad_accum_steps == 0:

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm
                    )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # -------------------------
                    # EMA UPDATE
                    # -------------------------
                    self.update_ema()

            train_total_avg = total_loss / tr_seen

            # ==========================
            # VALIDATION (EMA MODEL)
            # ==========================
            self.ema_model.eval()

            with torch.no_grad():

                val_total = 0
                val_seen = 0

                for batch in tqdm(self.val_loader, desc=f"Eval Epoch {epoch}"):

                    batch = batch[0].to(dtype=model_dtype, device=self.device)

                    loss = self.ema_model(batch, mode="uncond")

                    val_total += loss.item() * batch.shape[0]
                    val_seen += batch.shape[0]

                val_total /= val_seen

            # ==========================
            # LOG
            # ==========================
            wandb.log({
                "train/total_loss": train_total_avg,
                "val/total_loss": val_total,
                "epoch": epoch,
                "step": global_steps,
                "lr": self.optimizer.param_groups[0]["lr"],
            })

            print(
                f"Epoch {epoch} | "
                f"Train Loss: {train_total_avg:.6f} | "
                f"Val Loss: {val_total:.6f}"
            )

            # ==========================
            # SAVE BEST
            # ==========================
            if val_total < best_val_loss:
                best_val_loss = val_total
                torch.save(
                    self.model.state_dict(),
                    f"{self.save_dir}/ckpt_best.pth"
                )
                torch.save(
                    self.ema_model.state_dict(),
                    f"{self.save_dir}/ema_ckpt_best.pth"
                )


        wandb.finish()


    def cond_train(self, config):

        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config=config,
        )

        os.makedirs(self.save_dir, exist_ok=True)

        best_val_loss = float("inf")
        no_improve_epochs = 0
        global_steps = 0

        model_dtype = next(self.model.parameters()).dtype

        for epoch in range(self.max_epochs):

            total_loss = 0
            tr_seen = 0

            self.model.train()
            self.optimizer.zero_grad()

            # ==========================
            # TRAIN
            # ==========================
            for batch in tqdm(self.train_loader, desc=f"Train Epoch {epoch}"):

                # batch = batch[0].to(dtype=model_dtype, device=self.device)
                batch = {k: v.to(dtype=model_dtype, device=self.device) for k, v in batch.items()}
                loss = self.model(batch, mode="cond")

                total_loss += loss.item()
                tr_seen += 1
                global_steps += 1

                loss_backward = loss / self.grad_accum_steps
                loss_backward.backward()

                if global_steps % self.grad_accum_steps == 0:

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm
                    )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # -------------------------
                    # EMA UPDATE
                    # -------------------------
                    self.update_ema()

            train_total_avg = total_loss / tr_seen

            # ==========================
            # VALIDATION (EMA MODEL)
            # ==========================
            self.ema_model.eval()

            with torch.no_grad():

                val_total = 0
                val_seen = 0

                for batch in tqdm(self.val_loader, desc=f"Eval Epoch {epoch}"):

                    # batch = batch[0].to(dtype=model_dtype, device=self.device)
                    batch = {k: v.to(dtype=model_dtype, device=self.device) for k, v in batch.items()}

                    loss = self.ema_model(batch, mode="cond")

                    val_total += loss.item() * batch['ts'].shape[0]
                    val_seen += batch['ts'].shape[0]

                val_total /= val_seen

            # ==========================
            # LOG
            # ==========================
            wandb.log({
                "train/total_loss": train_total_avg,
                "val/total_loss": val_total,
                "epoch": epoch,
                "step": global_steps,
                "lr": self.optimizer.param_groups[0]["lr"],
            })

            print(
                f"Epoch {epoch} | "
                f"Train Loss: {train_total_avg:.6f} | "
                f"Val Loss: {val_total:.6f}"
            )

            # ==========================
            # SAVE BEST
            # ==========================
            if val_total < best_val_loss:
                best_val_loss = val_total
                torch.save(
                    self.model.state_dict(),
                    f"{self.save_dir}/ckpt_best.pth"
                )
                torch.save(
                    self.ema_model.state_dict(),
                    f"{self.save_dir}/ema_ckpt_best.pth"
                )


        wandb.finish()





    def cond_causal_train(self, config):

        wandb.init(
            project=self.wandb_project_name,
            name=self.wandb_run_name,
            config=config,
        )

        os.makedirs(self.save_dir, exist_ok=True)

        best_val_loss = float("inf")
        no_improve_epochs = 0
        global_steps = 0

        model_dtype = next(self.model.parameters()).dtype

        for epoch in range(self.max_epochs):

            total_loss = 0
            tr_seen = 0

            self.model.train()
            self.optimizer.zero_grad()

            # ==========================
            # TRAIN
            # ==========================
            for batch in tqdm(self.train_loader, desc=f"Train Epoch {epoch}"):

                # batch = batch[0].to(dtype=model_dtype, device=self.device)
                batch = {k: v.to(dtype=model_dtype, device=self.device) for k, v in batch.items()}
                loss = self.model(batch, mode="cond")

                total_loss += loss.item()
                tr_seen += 1
                global_steps += 1

                loss_backward = loss / self.grad_accum_steps
                loss_backward.backward()

                if global_steps % self.grad_accum_steps == 0:

                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.grad_clip_norm
                    )

                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # -------------------------
                    # EMA UPDATE
                    # -------------------------
                    self.update_ema()

            train_total_avg = total_loss / tr_seen

            # ==========================
            # VALIDATION (EMA MODEL)
            # ==========================
            self.ema_model.eval()

            with torch.no_grad():

                val_total = 0
                val_seen = 0

                for batch in tqdm(self.val_loader, desc=f"Eval Epoch {epoch}"):

                    # batch = batch[0].to(dtype=model_dtype, device=self.device)
                    batch = {k: v.to(dtype=model_dtype, device=self.device) for k, v in batch.items()}

                    loss = self.ema_model(batch, mode="cond")

                    val_total += loss.item() * batch['target'].shape[0]
                    val_seen += batch['target'].shape[0]

                val_total /= val_seen

            # ==========================
            # LOG
            # ==========================
            wandb.log({
                "train/total_loss": train_total_avg,
                "val/total_loss": val_total,
                "epoch": epoch,
                "step": global_steps,
                "lr": self.optimizer.param_groups[0]["lr"],
            })

            print(
                f"Epoch {epoch} | "
                f"Train Loss: {train_total_avg:.6f} | "
                f"Val Loss: {val_total:.6f}"
            )

            # ==========================
            # SAVE BEST
            # ==========================
            if val_total < best_val_loss:
                best_val_loss = val_total
                torch.save(
                    self.model.state_dict(),
                    f"{self.save_dir}/ckpt_best.pth"
                )
                torch.save(
                    self.ema_model.state_dict(),
                    f"{self.save_dir}/ema_ckpt_best.pth"
                )


        wandb.finish()