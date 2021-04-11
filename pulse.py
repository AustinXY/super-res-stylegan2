from model import Generator
from dataclasses import dataclass
from SphericalOptimizer import SphericalOptimizer
from pathlib import Path
import numpy as np
import time
import torch
from loss import LossBuilder
from functools import partial
from tqdm import tqdm

# from drive import open_url


class PULSE(torch.nn.Module):
    def __init__(self, generator, device, verbose=True):
        super(PULSE, self).__init__()

        self.generator = generator
        self.verbose = verbose

        for param in self.generator.parameters():
            param.requires_grad = False

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

        with torch.no_grad():
            torch.manual_seed(0)
            latent = torch.randn((1000000, 512),dtype=torch.float32, device=device)
            latent_out = torch.nn.LeakyReLU(5)(self.generator.style(latent))
            self.gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}

    def forward(self,
                ref_im,
                seed,
                loss_str,
                eps,
                noise_type,
                num_trainable_noise_layers,
                tile_latent,
                bad_noise_layers,
                opt_name,
                learning_rate,
                steps,
                lr_schedule,
                save_intermediate):

        if seed:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True

        batch_size = ref_im.shape[0]
        device = ref_im.device
        n_latent = self.generator.n_latent

        # Generate latent tensor
        if tile_latent:
            latent = torch.randn(
                (batch_size, 1, 512), dtype=torch.float, requires_grad=True, device=device)
        else:
            latent = torch.randn(
                (batch_size, n_latent, 512), dtype=torch.float, requires_grad=True, device=device)

        noises_single = self.generator.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(ref_im.shape[0], 1, 1, 1).normal_())
        for noise in noises:
            noise.requires_grad = True

        var_list = [latent] + noises

        opt_dict = {
            'sgd': torch.optim.SGD,
            'adam': torch.optim.Adam,
            'sgdm': partial(torch.optim.SGD, momentum=0.9),
            'adamax': torch.optim.Adamax
        }
        opt_func = opt_dict[opt_name]
        opt = SphericalOptimizer(opt_func, var_list, lr=learning_rate)

        schedule_dict = {
            'fixed': lambda x: 1,
            'linear1cycle': lambda x: (9*(1-np.abs(x/steps-1/2)*2)+1)/10,
            'linear1cycledrop': lambda x: (9*(1-np.abs(x/(0.9*steps)-1/2)*2)+1)/10 if x < 0.9*steps else 1/10 + (x-0.9*steps)/(0.1*steps)*(1/1000-1/10),
        }
        schedule_func = schedule_dict[lr_schedule]
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt.opt, schedule_func)

        loss_builder = LossBuilder(ref_im, loss_str, n_latent, eps).cuda()

        min_loss = np.inf
        best_summary = ""
        start_t = time.time()
        gen_im = None

        if self.verbose: print("Optimizing")

        pbar = tqdm(range(steps))
        for j in pbar:
            opt.opt.zero_grad()

            # Duplicate latent in case tile_latent = True
            if (tile_latent):
                latent_in = latent.expand(-1, n_latent, -1)
            else:
                latent_in = latent

            # Apply learned linear mapping to match latent distribution to that of the mapping network
            latent_in = self.lrelu(latent_in*self.gaussian_fit["std"] + self.gaussian_fit["mean"])

            # Normalize image to [0,1] instead of [-1,1]
            gen_im, _ = self.generator([latent_in], input_is_latent=True, noise=noises)
            gen_im = (gen_im + 1) / 2

            # Calculate Losses
            loss, loss_dict = loss_builder(latent_in, gen_im)
            loss_dict['TOTAL'] = loss

            # Save best summary for log
            if loss < min_loss:
                min_loss = loss
                best_summary = f'BEST ({j+1}) | '+' | '.join(
                [f'{x}: {y:.4f}' for x, y in loss_dict.items()])
                best_im = gen_im.clone()

            # Save intermediate HR and LR images
            # if save_intermediate:
            #     yield (best_im.cpu().detach().clamp(0, 1),loss_builder.D(best_im).cpu().detach().clamp(0, 1))

            loss.backward()
            opt.step()
            scheduler.step()

        total_t = time.time()-start_t
        current_info = f' | time: {total_t:.1f} | it/s: {(j+1)/total_t:.2f} | batchsize: {batch_size}'
        if self.verbose: print(best_summary+current_info)

        return gen_im.clamp(0, 1)

        # yield (gen_im.clone().cpu().detach().clamp(0, 1),loss_builder.D(best_im).cpu().detach().clamp(0, 1))
