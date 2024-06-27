import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from TorchJaekwon.Util.UtilData import UtilData
from TorchJaekwon.Train.Trainer.Trainer import Trainer, TrainState

from TorchJaekwon.Train.AverageMeter import AverageMeter
#from TorchJaekwon.Util.UtilAudio import UtilAudio
#from TorchJaekwon.Util.UtilData import UtilData 
#from TorchJaekwon.Util.UtilTorch import UtilTorch   
#from TorchJaekwon.Util.UtilAudioMelSpec import UtilAudioMelSpec
#from DataProcess.Util.UtilAudioLowPassFilter import UtilAudioLowPassFilterNVSR
class GANTrainer(Trainer):
    def __init__(self, discriminator_freeze_step:int = 0, **kwargs):
        super().__init__(**kwargs)
        self.discriminator_freeze_step:int = discriminator_freeze_step
    
    def run_epoch(self, dataloader: DataLoader, train_state:TrainState, metric_range:str = "step"):
        if self.discriminator_freeze_step < self.global_step:
            self.log_writer.print_and_log('discriminator training')
        else:
            self.log_writer.print_and_log('only generator training')
            
        assert metric_range in ["step","epoch"], "metric range should be 'step' or 'epoch'"

        if train_state == TrainState.TRAIN:
            self.set_model_train_valid_mode(self.model, 'train')
        else:
            self.set_model_train_valid_mode(self.model, 'valid')

        try: dataset_size = len(dataloader)
        except: dataset_size = dataloader.dataset.__len__()


        if metric_range == "epoch":
            metric = dict()

        for step,data in enumerate(dataloader):

            if metric_range == "step":
                metric = dict()

            if step >= len(dataloader):
                break

            self.local_step = step
            metric = self.run_step(data,metric,train_state)
        
            if train_state == TrainState.TRAIN:
                
                if self.local_step % self.h_params.log.log_every_local_step == 0:
                    self.log_metric(metrics=metric,data_size=dataset_size)
                
                self.global_step += 1

                self.lr_scheduler_step(call_state='step')
            
            if train_state == TrainState.TRAIN and self.save_model_every_step is not None and self.global_step % self.save_model_every_step == 0:
                self.save_module(self.model, name=f"step{self.global_step}")
                self.log_current_state()
        
        if train_state == TrainState.VALIDATE or train_state == TrainState.TEST:
            self.save_module(self.model, name=f"step{self.global_step}")
            self.log_metric(metrics=metric,data_size=dataset_size,train_state=train_state)
            self.log_current_state(train_state)

        return metric
    
    def run_generator_step(self, data, metric, train_state):
        raise NotImplementedError
'''
class GANTrainer(Trainer):

    def __init__(self, discriminator_freeze_step:int = 0):
        super().__init__()
        self.discriminator_freeze_step:int = discriminator_freeze_step
    
    def init_optimizer(self) -> None:
        learning_rate = 2.0e-4 #0.0001
        adam_b1 = 0.8
        adam_b2 = 0.99

        self.optim_g = torch.optim.Adam(params=itertools.chain(*[self.model['generator'][model_name].parameters() for model_name in self.model['generator']]),lr= 2.0e-4,betas=[0.5, 0.9],weight_decay=0.0) #torch.optim.AdamW(self.model['BigVGANMelAudioUpsampler'].parameters(), learning_rate, betas=[adam_b1, adam_b2])
        self.optim_d = torch.optim.Adam(params=itertools.chain(*[self.model['discriminator'][model_name].parameters() for model_name in self.model['discriminator']]),lr= 2.0e-4,betas=[0.5, 0.9],weight_decay=0.0) #torch.optim.AdamW(itertools.chain(self.model['MultiResolutionDiscriminator'].parameters(), self.model['MultiPeriodDiscriminator'].parameters()),learning_rate, betas=[adam_b1, adam_b2])
        self.scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optim_g,gamma=0.5,milestones=[100000, 200000, 300000, 400000]) #torch.optim.lr_scheduler.ExponentialLR(self.optim_g, gamma=0.999, last_epoch=-1)
        self.scheduler_d = torch.optim.lr_scheduler.MultiStepLR(optimizer=self.optim_d,gamma=0.5,milestones=[100000, 200000, 300000, 400000])#torch.optim.lr_scheduler.ExponentialLR(self.optim_d, gamma=0.999, last_epoch=-1)
    
    def metric_init(self):
        loss_name_list = []
        initialized_metric = dict()

        for loss_name in loss_name_list:
            initialized_metric[loss_name] = AverageMeter()

        return initialized_metric
    
    @staticmethod
    def discriminator_loss(disc_real_outputs, disc_generated_outputs):
        loss = 0
        r_losses = []
        g_losses = []
        for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
            r_loss = torch.mean((1-dr)**2)
            g_loss = torch.mean(dg**2)
            loss += (r_loss + g_loss)
            r_losses.append(r_loss.item())
            g_losses.append(g_loss.item())

        return loss, r_losses, g_losses
    
    @staticmethod
    def feature_loss(fmap_r, fmap_g):
        loss = 0
        for dr, dg in zip(fmap_r, fmap_g):
            for rl, gl in zip(dr, dg):
                loss += torch.mean(torch.abs(rl - gl))

        return loss*2
    
    @staticmethod
    def generator_loss(disc_outputs):
        loss = 0
        gen_losses = []
        for dg in disc_outputs:
            l = torch.mean((1-dg)**2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses
    
    def run_step(self,data,metric,train_state):
        """
        run 1 step
        1. get data
        2. use model
        3. calculate loss
        4. put the loss in metric (append)
        return loss,metric
        """
        #print('-------------------------------train step---------------------------------')
        data_dict = self.data_dict_to_device(data)
        pred_hr_audio:Tensor = self.model['BigVGANMelAudioUpsampler'](data_dict['lr_audio'])
        current_loss_dict = dict()
        current_loss_dict['disc_total'], current_loss_dict['disc_mrd'], current_loss_dict['disc_mpd'] = self.discriminator_step(data_dict,pred_hr_audio,train_state)
        current_loss_dict['gen_total'], current_loss_dict['gen_mel'], current_loss_dict['gen_mpd'], current_loss_dict['gen_mrd'], current_loss_dict['gen_mpd_fm'], current_loss_dict['gen_mrd_fm'] = self.generator_step(data_dict,pred_hr_audio,train_state)

        for loss_name in current_loss_dict:
            if loss_name not in metric:
                metric[loss_name] = AverageMeter()
            metric[loss_name].update(current_loss_dict[loss_name].item(),data_dict['lr_audio'].shape[0])

        return current_loss_dict["gen_total"],metric
    
    def lr_scheduler_step(self, call_state:Literal['step','epoch'], args = None):
        #if call_state == 'epoch':
        #    self.scheduler_g.step()
        #    self.scheduler_d.step()
        print('')
    
    def discriminator_step(self,data_dict,pred_hr_audio,train_state):
        self.optim_d.zero_grad()

        # MPD
        y_df_hat_r, y_df_hat_g, _, _ = self.model['MultiPeriodDiscriminator'](data_dict['hr_audio'].unsqueeze(1), pred_hr_audio.unsqueeze(1).detach())
        loss_disc_f, losses_disc_f_r, losses_disc_f_g = BigVGANSRTrainer.discriminator_loss(y_df_hat_r, y_df_hat_g)

        # MRD
        y_ds_hat_r, y_ds_hat_g, _, _ = self.model['MultiResolutionDiscriminator'](data_dict['hr_audio'].unsqueeze(1), pred_hr_audio.unsqueeze(1).detach())
        loss_disc_s, losses_disc_s_r, losses_disc_s_g = BigVGANSRTrainer.discriminator_loss(y_ds_hat_r, y_ds_hat_g)

        loss_disc_all = loss_disc_s + loss_disc_f

        if train_state == TrainState.TRAIN:
            # whether to freeze D for initial training steps
            if self.global_step >= self.discriminator_freeze_step:
                loss_disc_all.backward()
                grad_norm_mpd = torch.nn.utils.clip_grad_norm_(self.model['MultiPeriodDiscriminator'].parameters(), 1000.)
                grad_norm_mrd = torch.nn.utils.clip_grad_norm_(self.model['MultiResolutionDiscriminator'].parameters(), 1000.)
                self.optim_d.step()
                self.scheduler_d.step()
            else:
                print("WARNING: skipping D training for the first {} steps".format(self.discriminator_freeze_step))
                grad_norm_mpd = 0.
                grad_norm_mrd = 0.
        
        return loss_disc_all, loss_disc_s, loss_disc_f
    
    def generator_step(self,data_dict,pred_hr_audio,train_state):
        self.optim_g.zero_grad()
        loss_mel = F.l1_loss(self.mel_spec.get_hifigan_mel_spec(data_dict['hr_audio']), self.mel_spec.get_hifigan_mel_spec(pred_hr_audio)) * 45
        # MPD loss
        y_df_hat_r, y_df_hat_g, fmap_f_r, fmap_f_g = self.model['MultiPeriodDiscriminator'](data_dict['hr_audio'].unsqueeze(1), pred_hr_audio.unsqueeze(1))
        loss_fm_f = BigVGANSRTrainer.feature_loss(fmap_f_r, fmap_f_g)
        loss_gen_f, losses_gen_f = BigVGANSRTrainer.generator_loss(y_df_hat_g)

        # MRD loss
        y_ds_hat_r, y_ds_hat_g, fmap_s_r, fmap_s_g = self.model['MultiResolutionDiscriminator'](data_dict['hr_audio'].unsqueeze(1), pred_hr_audio.unsqueeze(1))
        loss_fm_s = BigVGANSRTrainer.feature_loss(fmap_s_r, fmap_s_g)
        loss_gen_s, losses_gen_s = BigVGANSRTrainer.generator_loss(y_ds_hat_g)
        if self.global_step >= self.discriminator_freeze_step:
            loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel
        else:
            print("WARNING: using regression loss only for G for the first {} steps".format(self.discriminator_freeze_step))
            loss_gen_all = loss_mel
        if train_state == TrainState.TRAIN:
            loss_gen_all.backward()
            grad_norm_g = torch.nn.utils.clip_grad_norm_(self.model['BigVGANMelAudioUpsampler'].parameters(), 1000.)
            self.optim_g.step()
            self.scheduler_g.step()
        return loss_gen_all, loss_mel, loss_gen_f, loss_gen_s, loss_fm_f, loss_fm_s
    
    def backprop(self,loss):
        print('')

    def get_current_lr(self):
        return self.optim_g.param_groups[0]['lr']
    
    def save_checkpoint(self,save_name:str = 'train_checkpoint.pth'):
        train_state = {
            'epoch': self.current_epoch,
            'step': self.global_step,
            'seed': self.seed,
            'generator_optimizers': self.optim_g.state_dict(),
            'generator_lr_scheduler': self.scheduler_g.state_dict(),
            'discriminator_optimizers': self.optim_d.state_dict(),
            'discriminator_lr_scheduler': self.scheduler_d.state_dict(),
            'best_metric': self.best_valid_metric,
            'best_model_epoch' :  self.best_valid_epoch,
        }

        train_state.update(self.get_model_state_dict(self.model))

        path = os.path.join(self.log_writer.log_path["root"],save_name)
        self.log_writer.print_and_log(save_name)
        torch.save(train_state,path)
    
    def load_train(self,filename:str):
        cpt = torch.load(filename,map_location='cpu')
        self.seed = cpt['seed']
        self.set_seeds(self.h_params.train.seed_strict)
        self.current_epoch = cpt['epoch']
        self.global_step = cpt['step']

        if isinstance(self.model, dict):
            for model_name in self.model:
                self.model[model_name].to(torch.device('cpu'))
                self.model[model_name].load_state_dict(cpt[f'model_{model_name}'])
                self.model[model_name].to(self.device)
        self.optim_g.load_state_dict(cpt['generator_optimizers'])
        self.scheduler_g.load_state_dict(cpt['generator_lr_scheduler'])
        self.optim_d.load_state_dict(cpt['discriminator_optimizers'])
        self.scheduler_d.load_state_dict(cpt['discriminator_lr_scheduler'])

        self.best_valid_result = cpt['best_metric']
        self.best_valid_epoch = cpt['best_model_epoch']

    def log_media(self) -> None:
        cutoff_freq = 4000
        filter_name = "cheby"
        filter_order =  8
        lowpass_filter = UtilAudioLowPassFilterNVSR()
        for i,data_path in tqdm(enumerate(self.log_test_data_list),desc='plot audio'):
            audio, sr = UtilAudio.read(data_path, mono=True)
            assert sr == 48000, f"sr is {sr}"
            audio = UtilData.fix_length(audio,19200 * 10)
            lr_audio = lowpass_filter.lowpass(audio, sr, filter_name=filter_name, filter_order=filter_order, cutoff_freq=cutoff_freq)
            pred_hr_audio = self.model['BigVGANMelAudioUpsampler'](torch.from_numpy(lr_audio).unsqueeze(0).float().to(self.device))
            self.log_writer.plot_audio(name = f'testcase{i}', audio_dict = {'lr_audio': lr_audio, 'hr_audio': audio, 'pred_hr_audio': UtilTorch.to_np(pred_hr_audio)}, global_step=self.global_step, sample_rate=48000)

    def save_best_model(self,prev_best_metric, current_metric):
        """
        compare what is the best metric
        If current_metric is better, 
            1.save best model
            2. self.best_valid_epoch = self.current_epoch
        Return
            better metric
        """
        return None
'''