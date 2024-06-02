'''
2021_ICML_Improved denoising diffusion probabilistic models
Code Reference: https://github.com/facebookresearch/DiT
'''
#type
from typing import Optional, Union
from torch import Tensor
#package
import torch
#torchjaekwon
from TorchJaekwon.Util.UtilData import UtilData
from TorchJaekwon.Model.Diffusion.DDPM.DDPM import DDPM

class DDPMLearningVariances(DDPM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def p_losses(self, 
                 x_start:Tensor,
                 cond:Optional[Union[dict,Tensor]],
                 is_cond_unpack:bool,
                 t:Tensor, 
                 noise:Optional[Tensor] = None):
        noise:Tensor = UtilData.default(noise, lambda: torch.randn_like(x_start))
        x_noisy:Tensor = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output:Tensor = self.apply_model(x_noisy, t, cond, is_cond_unpack)

        batch_size, channel_size = x_noisy.shape[:2]
        assert model_output.shape == (batch_size, channel_size * 2, *x_noisy.shape[2:]), 'Model output size is expected to be (batch_size, channel_size * 2, ...), because it also predicts variance.'
        model_output, model_var_values = torch.split(model_output, channel_size, dim=1)
        # Learn the variance using the variational bound, but don't let it affect our mean prediction.
        mean_frozen_output = torch.cat([model_output.detach(), model_var_values], dim=1)
        vlb_loss = self._vb_terms_bpd(model=lambda *args, fixed_return_value = mean_frozen_output: fixed_return_value,
                                      x_start=x_start,
                                      x_t=x_t,
                                      t=t,
                                      clip_denoised=False,
                                      )["output"]

        if self.model_output_type == 'x_start':
            target:Tensor = x_start
        elif self.model_output_type == 'noise':
            target:Tensor = noise
        elif self.model_output_type == 'v_prediction':
            target:Tensor = self.get_v(x_start, noise, t)
        else:
            print(f'''model output type is {self.model_output_type}. It should be in [x_start, noise]''')
            raise NotImplementedError()
        if target.shape != model_output.shape: print(f'warning: target shape({target.shape}) and model shape({model_output.shape}) are different')
        return self.loss_func(target, model_output) + vlb_loss
    
    def _vb_terms_bpd(
            self, model, x_start, x_t, t, clip_denoised=True, model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound. 
        bits per dimension (bpd).
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        true_mean, _, true_log_variance_clipped = self.q_posterior_mean_variance(
            x_start=x_start, x_t=x_t, t=t
        )
        out = self.p_mean_variance(
            model, x_t, t, clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        kl = normal_kl(
            true_mean, true_log_variance_clipped, out["mean"], out["log_variance"]
        )
        kl = mean_flat(kl) / np.log(2.0)

        decoder_nll = -discretized_gaussian_log_likelihood(
            x_start, means=out["mean"], log_scales=0.5 * out["log_variance"]
        )
        assert decoder_nll.shape == x_start.shape
        decoder_nll = mean_flat(decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        output = torch.where((t == 0), decoder_nll, kl)
        return {"output": output, "pred_xstart": out["pred_xstart"]}