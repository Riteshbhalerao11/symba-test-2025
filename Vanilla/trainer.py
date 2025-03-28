from tqdm import tqdm
from data import Data
from fn_utils import calculate_line_params, collate_fn, create_mask, generate_eqn_mask, generate_unique_random_integers, get_model, decode_sequence
import torch
import os
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb
import numpy as np

from constants import BOS_IDX, PAD_IDX, EOS_IDX

def sequence_accuracy(config,test_ds,tgt_itos,load_best=True, epoch=None,test_size=100):
    """
    Calculate the sequence accuracy.

    Args:
        load_best (bool, optional): Whether to load the best model. Defaults to True.
        epochs (int, optional): Number of epochs. Defaults to None.

    Returns:
        float: Sequence accuracy.
    """
    predictor = Predictor(config,load_best, epoch)
    count = 0
    num_samples = 10 if config.debug else test_size 
    random_idx = generate_unique_random_integers(
        num_samples, start=0, end=len(test_ds))
    length = len(random_idx)
    pbar = tqdm(range(length))
    pbar.set_description("Seq_Acc_Cal")
    for i in pbar:
        original_tokens, predicted_tokens = predictor.predict(
            test_ds[random_idx[i]],tgt_itos, raw_tokens=True)
        original_tokens = original_tokens.detach().numpy().tolist()
        predicted_tokens = predicted_tokens.detach().cpu().numpy().tolist()
        original = decode_sequence(original_tokens,tgt_itos)
        predicted = decode_sequence(predicted_tokens,tgt_itos)
        if original == predicted:
            count = count + 1
        pbar.set_postfix(seq_accuracy=count / (i + 1))
    return count / length


class Predictor:
    """
    Class for generating predictions using a trained model and greedy decoding.

    Args:
        config (object): Configuration object containing model and inference settings.
        load_best (bool, optional): Whether to load the best model. Defaults to True.
        epoch (int, optional): Epoch number to load a specific checkpoint.

    Attributes:
        model (Model): Trained model for prediction.
        path (str): Path to the trained model checkpoint.
        device (str): Device for inference.
        checkpoint (str): Model checkpoint filename.
        max_len (int): Maximum target sequence length for inference.
    """

    def __init__(self, config, load_best=True, epoch=None):
        self.model = get_model(config)
        self.checkpoint = (
            f"{config.model_name}_best.pth"
            if load_best else f"{config.model_name}_ep{epoch + 1}.pth"
        )
        self.path = os.path.join(config.root_dir, self.checkpoint)
        self.device = config.device
        
        # Load model checkpoint
        state = torch.load(self.path, map_location=self.device)
        self.model.load_state_dict(state['state_dict'])
        self.model.to(self.device)
        self.max_len = config.tgt_max_len
        
        print(f"Using epoch {state['epoch']} model for predictions.")

    def greedy_decode(self, src, src_mask, src_padding_mask, start_symbol):
        """
        Performs greedy decoding to generate predictions.

        Args:
            src (Tensor): Source tensor.
            src_mask (Tensor): Source mask tensor.
            src_padding_mask (Tensor): Source padding mask.
            start_symbol (int): Start token index.

        Returns:
            Tensor: Generated token sequence.
        """
        src, src_mask, src_padding_mask = (
            src.to(self.device), src_mask.to(self.device), src_padding_mask.to(self.device)
        )
        
        memory = self.model.encode(src, src_mask, src_padding_mask).to(self.device)
        
        ys = torch.ones(1, 1, dtype=torch.long, device=self.device).fill_(start_symbol)
        
        for _ in range(self.max_len):
            tgt_mask = generate_eqn_mask(ys.size(0), self.device).bool().to(self.device)
            tgt_padding_mask = (ys == PAD_IDX).transpose(0, 1).to(self.device)
            
            out = self.model.decode(ys, memory, tgt_mask, None, tgt_padding_mask, src_padding_mask)
            out = out.transpose(0, 1)
            
            prob = self.model.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            
            ys = torch.cat([ys, torch.ones(1, 1, dtype=src.dtype, device=self.device).fill_(next_word)], dim=0)
            
            if next_word == EOS_IDX:
                break
        
        return ys

    def predict(self, test_example, itos, raw_tokens=False):
        """
        Generates predictions for a given test example.

        Args:
            test_example (tuple): Tuple containing source tensor and original tokens.
            itos (dict): Index-to-string vocabulary mapping.
            raw_tokens (bool, optional): Whether to return raw token outputs. Defaults to False.

        Returns:
            str or tuple: Decoded equation or tuple of original and generated tokens.
        """
        self.model.eval()
        
        src = test_example[0].unsqueeze(1)
        
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            src, torch.zeros((1, 1), dtype=torch.long, device=self.device), self.device
        )
        
        tgt_tokens = self.greedy_decode(src, src_mask, src_padding_mask, start_symbol=BOS_IDX).flatten()
        
        if raw_tokens:
            return test_example[1], tgt_tokens
        
        return ''.join(itos[int(t)] for t in tgt_tokens)


class Trainer():
    """
    Class for training a sequence-to-sequence model.

    Args:
        start_epoch (int, optional): Starting epoch number. Defaults to 0.

    Attributes:
        scaler (GradScaler): Gradient scaler for half-precision training.
        dtype (torch.dtype): Data type for training.
        dataloaders (dict): Dataloaders for train, validation, and test datasets.
        root_dir (str): Root directory for saving models and logs.
        device (str): Device for training.
        current_epoch (int): Current epoch number.
        best_val_loss (float): Best validation loss.
        train_loss_list (list): List of training losses.
        valid_loss_list (list): List of validation losses.
        valid_accuracy_tok_list (list): List of validation token accuracies.
        model (Model): Model for training.
        optimizer (Optimizer): Optimizer for training.
        scheduler (Scheduler): Learning rate scheduler.
        resume_best (bool): Whether to resume from the last best saved model
        save_freq (int): Frequency of saving in terms of epochs
        save_last (bool): Whether to save model after complete training

    """

    def __init__(self, config, df_train, df_test, df_valid, tokenizer, src_vocab, tgt_vocab, tgt_itos):

        # For half precision training
        self.scaler = GradScaler()
        self.is_constant_lr = config.is_constant_lr
        if config.use_half_precision:
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        if config.debug is not True:
            print(f"PROCESS ID : {int(os.environ['SLURM_PROCID'])} ; TORCH GLOBAL RANK : {self.global_rank} ; TORCH LOCAL RANK : {self.local_rank}")
        self.device = self.local_rank
        self.config = config
        self.is_master = self.local_rank == 0
        if self.is_master:
            wandb.login()
            self.run = wandb.init(
            # set the wandb project where this run will be logged
            project= config.project_name,
            name = config.run_name,
            dir=config.root_dir,
            # track hyperparameters and run metadata
            config=config.to_dict()
            )
        self.dataloaders,self.test_ds = self._prepare_dataloaders(
            df_train, df_test, df_valid, tokenizer, src_vocab, tgt_vocab)
        self.warmup_steps = int(config.warmup_ratio *
                                len(self.dataloaders['train']) * config.epochs)
        self.ep_steps = len(self.dataloaders['train'])
        self.root_dir = config.root_dir
        self.current_epoch = config.curr_epoch
        self.best_val_loss = 1e6
        self.train_loss_list = []
        self.valid_loss_list = []
        self.model, self.ddp_model = self._prepare_model()
        self.optimizer = self._prepare_optimizer()
        self.warm_scheduler, self.lr_scheduler = self._prepare_scheduler()
        self.save_freq = config.save_freq
        self.test_freq = config.test_freq
        self.resume_best = config.resume_best
        self.save_last = config.save_last
        self.lr = config.update_lr
        self.global_step = 0
        self.tgt_itos = tgt_itos
        self.ckp_paths = [file for file in os.listdir(config.root_dir) if ('best' not in file and config.model_name in file)]
        self.save_limit = config.save_limit

    def criterion(self, y_pred, y_true):
        """
        Calculate the loss between predicted and true values.

        Args:
            y_pred (Tensor): Predicted values.
            y_true (Tensor): True values.

        Returns:
            Tensor: Loss value.
        """
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        return loss_fn(y_pred, y_true)

    def _prepare_model(self):
        """
        Initialize and prepare the model for training.

        Returns:
            Model: Initialized model.
        """
        model = get_model(self.config)
        model.to(self.device)
        ddp_model = DDP(model, device_ids=[self.device])
        if self.is_master:
            self.run.watch(ddp_model.module,log_freq=20)
        return model, ddp_model

    def _prepare_optimizer(self):
        """
        Initialize the optimizer.

        Returns:
            Optimizer: Initialized optimizer.
        """
        param_optimizer = list(self.ddp_model.parameters())
        optimizer = torch.optim.AdamW(
            param_optimizer, lr=self.config.optimizer_lr, eps=1e-9, weight_decay = self.config.weight_decay)
        return optimizer

    def _prepare_scheduler(self):
        start_lr = self.config.optimizer_lr
        end_lr = self.config.end_lr

        if self.warmup_steps:
            m_warm, c_warm = calculate_line_params(
                (0, end_lr), (self.warmup_steps, start_lr))
            
            def lam_warm(step): return (1/start_lr)*(m_warm*step + c_warm)
            warm_scheduler = LambdaLR(self.optimizer, lr_lambda=lam_warm)

        else:
            warm_scheduler = None
        
        if self.is_constant_lr:
            lr_scheduler = None
        else:
            m_decay, c_decay = calculate_line_params(
                (0, start_lr), (self.config.epochs, end_lr))

            def lam(epoch): return (1/start_lr) * (m_decay*epoch + c_decay)
            lr_scheduler = LambdaLR(self.optimizer, lr_lambda=lam)

        return warm_scheduler, lr_scheduler

    def _prepare_dataloaders(self, df_train, df_test, df_valid, tokenizer, src_vocab, tgt_vocab):
        """
        Prepare dataloaders for training, validation, and testing.

        Returns:
            dict: Dictionary containing train, validation, and test dataloaders.
        """
        datasets = Data.get_data(
            df_train, df_test, df_valid, self.config, tokenizer,src_vocab, tgt_vocab)
        sampler_train = torch.utils.data.DistributedSampler(datasets['train'], num_replicas=self.config.world_size,
                                                            rank=self.device, shuffle=self.config.train_shuffle, seed=self.config.seed)

        train_loader = torch.utils.data.DataLoader(datasets['train'], batch_size=self.config.training_batch_size,
                                                   sampler=sampler_train, num_workers=self.config.num_workers,
                                                   pin_memory=self.config.pin_memory, collate_fn=collate_fn)

        dataloaders = {
            'train': train_loader,
            'valid': torch.utils.data.DataLoader(datasets['valid'],
                                                 batch_size=self.config.valid_batch_size, shuffle=self.config.valid_shuffle,
                                                 num_workers=self.config.num_workers, pin_memory=self.config.pin_memory, collate_fn=collate_fn),
        }
        return dataloaders,datasets['test']

    def load_model(self, resume=False, epoch=None, lr=None):
        """
        Load the most recent model checkpoint.

        Args:
            resume (bool, optional): Whether to resume training. Defaults to False.
            epoch (int, optional): Load model from a particular epoch
        """
        checkpoint_name = f"{self.config.model_name}_best.pth" if resume else f"{self.config.model_name}_ep{epoch}.pth"
        file = os.path.join(self.root_dir, checkpoint_name)
        device_name = f"cuda:{self.device}"
        state = torch.load(file, map_location=device_name)
        self.model.load_state_dict(state['state_dict'])
        if resume or (epoch != None):
            self.train_loss_list = state['train_loss_list']
            self.valid_loss_list = state['valid_loss_list']
            self.best_val_loss = np.array(self.valid_loss_list).min()
            self.optimizer.load_state_dict(state['optimizer'])
            
            if state['decay_scheduler'] is not None:
                self.lr_scheduler.load_state_dict(state['decay_scheduler'])
            if state['warm_scheduler'] is not None:
                self.warm_scheduler.load_state_dict(state['warm_scheduler'])
            self.global_step = state['global_step']

            if epoch == None:
                self.current_epoch = state['epoch']
            
            if lr:
                for g in self.optimizer.param_groups:
                    g['lr'] = lr
                print("Lr_changed :)")

            print(checkpoint_name)
            print("Loaded :)")

    def _train_epoch(self):
        """
        Perform a single training epoch.

        Returns:
            float: Average training loss for the epoch.
        """
        self.ddp_model.train()
        pbar = tqdm(self.dataloaders['train'],
                    total=len(self.dataloaders['train']),disable= (not self.is_master))
        pbar.set_description(
            f"[{self.current_epoch+1}/{self.config.epochs}] Train")
        running_loss = 0.0
        total_samples = 0

        for src, tgt in pbar:
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            bs = src.size(1)

            with torch.autocast(device_type='cuda', dtype=self.dtype):
                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                    src, tgt[:-1, :], self.device)

                logits = self.ddp_model(
                    src, tgt[:-1, :], src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), tgt[1:, :].reshape(-1))
            if ((self.global_step % self.config.log_freq == 0) and self.is_master):
                self.run.log({'train/loss': loss.item(),
                          'global_step': self.global_step})
            running_loss += loss.item() * bs
            total_samples += bs
            avg_loss = running_loss / total_samples
            pbar.set_postfix(loss=avg_loss)

            # Backward
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.config.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.ddp_model.parameters(), self.config.clip_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
           
            grads = [
                param.grad.detach().flatten()
                for param in self.ddp_model.module.parameters()
                if param.grad is not None ]
            norm = torch.cat(grads).norm()

            if (self.global_step <= self.warmup_steps):
                if self.is_master:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.run.log({'train/lr': lr , 'global_step': self.global_step})
                if self.warmup_steps:
                    self.warm_scheduler.step()
            else:
                if (self.global_step % self.config.log_freq == 0) and self.is_master:
                    lr = self.optimizer.param_groups[0]['lr']
                    self.run.log({'train/lr': lr, 'global_step': self.global_step})

            if ((self.global_step % self.config.log_freq) == 0) and self.is_master:
                self.run.log({'train/epoch': self.global_step /
                          self.ep_steps, 'global_step': self.global_step})
                self.run.log({'train/grad_norm': norm, 'global_step': self.global_step})

            self.global_step += 1

        return avg_loss

    def evaluate(self, phase):
        """
        Evaluate the model on the validation or test set.

        Args:
            phase (str): Phase of evaluation ('valid' or 'test').

        Returns:
            tuple: Tuple containing average token accuracy and average loss.
        """
        self.ddp_model.eval()
        pbar = tqdm(self.dataloaders[phase],
                    total=len(self.dataloaders[phase]), disable= (not self.is_master))
        pbar.set_description(
            f"[{self.current_epoch+1}/{self.config.epochs}] {phase.capitalize()}")
        running_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for src, tgt in pbar:
                src = src.to(self.device)
                tgt = tgt.to(self.device)
                bs = src.size(1)

                src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                    src, tgt[:-1, :], self.device)
                logits = self.ddp_model(
                    src, tgt[:-1, :], src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
                loss = self.criterion(
                    logits.reshape(-1, logits.shape[-1]), tgt[1:, :].reshape(-1))

                running_loss += loss.item() * bs
                total_samples += bs
                avg_loss = running_loss / total_samples

        return avg_loss

    def _save_model(self, checkpoint_name):
        """
        Save the model checkpoint.

        Args:
            checkpoint_name (str): Name of the checkpoint file.
        """
        ckp_path = os.path.join(self.root_dir, checkpoint_name)
        state_dict = self.ddp_model.module.state_dict()
        torch.save({
            "epoch": self.current_epoch + 1,
            "state_dict": state_dict,
            'optimizer': self.optimizer.state_dict(),
            'decay_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            'warm_scheduler': self.warm_scheduler.state_dict() if self.warm_scheduler else None,
            "train_loss_list": self.train_loss_list,
            "valid_loss_list": self.valid_loss_list,
            "global_step": self.global_step
        }, ckp_path)
        
        if "best" not in  checkpoint_name:
            self.ckp_paths.append(ckp_path)
        
        # Remove oldest checkpoint if exceeding save_limit
        if len(self.ckp_paths) > self.save_limit:
            oldest_checkpoint = self.ckp_paths.pop(0)
            if os.path.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)
                print(f"Deleted old checkpoint: {oldest_checkpoint}")

    def _test_seq_acc(self, load_best=True, epochs=None):
        """
        Test sequence accuracy and save results to a file.
        """

        test_accuracy_seq = sequence_accuracy(self.config,self.test_ds,self.tgt_itos,load_best, epochs)
        self.run.log({'test/acc': test_accuracy_seq,
                  'global_step': self.global_step})
        print(f"Test Accuracy: {round(test_accuracy_seq, 4)}")

    def fit(self):
        """
        Train the model.
        """
        if self.is_master:

            self.run.define_metric("global_step")
            self.run.define_metric("validation/*", step_metric="global_step")
            self.run.define_metric("train/*", step_metric="global_step")
            self.run.define_metric("test/*", step_metric="global_step")
        
        
        if self.current_epoch != 0:
                self.load_model(epoch=self.current_epoch, lr=self.lr)
        
        elif self.resume_best:
                self.load_model(resume=True, lr=self.lr)

        for self.current_epoch in range(self.current_epoch, self.config.epochs):

            training_loss = self._train_epoch()
            valid_loss = self.evaluate("valid")
            if self.global_step >= self.warmup_steps:
                if self.is_constant_lr is False:
                    self.lr_scheduler.step(self.current_epoch)
            if self.is_master:
                self.run.log({'valid/loss': valid_loss,
                        'global_step': self.global_step})

            self.train_loss_list.append(round(training_loss, 4))
            self.valid_loss_list.append(round(valid_loss, 4))

            if self.is_master:
                if valid_loss <= self.best_val_loss:
                            self.best_val_loss = valid_loss
                            self._save_model(f"{self.config.model_name}_best.pth")

                if self.save_freq:
                    if (self.current_epoch+1) % self.save_freq == 0:
                        self._save_model(f"{self.config.model_name}_ep{self.current_epoch + 1}.pth")
                        self._test_seq_acc(load_best=False,epochs=self.current_epoch)

                    elif (self.current_epoch+1) % self.test_freq == 0:
                        self._test_seq_acc()                            

            
            torch.distributed.barrier()

            print(f"Epoch {self.current_epoch + 1}/{self.config.epochs}, "
                  f"Training Loss: {training_loss:.4f}, "
                  f"Validation Loss: {valid_loss:.4f}, ")
        if self.is_master:
            if self.save_last:
                self._save_model(f"{self.config.model_name}_ep{self.current_epoch + 1}.pth")
            self._test_seq_acc(load_best=False, epochs=self.current_epoch)

        wandb.finish()