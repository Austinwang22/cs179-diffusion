import torch
import os
from tqdm import tqdm 
from argparse import ArgumentParser
from unet import EDMPrecond, UNet
from utils import save_image_batch

# Parallelizing this with CUDA just requires the neural network forward pass and the tensor operations to be parallelized.
# We detailed how to do this in the unet.py file. 

def euler_sampler(net, num_samples, num_steps=18, sigma_min=0.002, 
                sigma_max=80, rho=7, device='cuda'):
    model.eval()
    model.to(torch.float32)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]).to(torch.float32) # t_N = 0

    x_next = torch.randn([num_samples, *net.unet.shape], device=device) * sigma_max
    pbar = tqdm(range(num_steps))
    for i in pbar:
        t_cur = t_steps[i]
        t_next = t_steps[i + 1]

        x_cur = x_next
        
        denoised = net(x_cur, t_cur.repeat(num_samples))
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

    return x_next

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img_resolution', type=int, default=28)
    parser.add_argument('--num_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_folder', type=str, default='exp/ckpts/')
    parser.add_argument('--filename', type=str, default='model.pt')
    parser.add_argument('--Pmean', type=float, default=-1.2)
    parser.add_argument('--Pstd', type=float, default=1.2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--sigma_min', type=float, default=0.002)
    parser.add_argument('--sigma_max', type=int, default=80.)
    parser.add_argument('--rho', type=int, default=7)
    args = parser.parse_args()
    
    net = UNet(args.img_resolution)
    device = torch.device(args.device)
    model = EDMPrecond(net).to(device)
    
    model.load_state_dict(torch.load(os.path.join(args.save_folder, args.filename)))
    
    samples = euler_sampler(model, args.batch_size, args.num_steps, args.sigma_min, args.sigma_max, args.rho, args.device)
    save_image_batch(samples)