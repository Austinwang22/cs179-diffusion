import torch
import os
from tqdm import tqdm 
from argparse import ArgumentParser
from unet import EDMPrecond, UNet
from utils import save_image_batch

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--img_resolution', type=int, default=28)
    parser.add_argument('--save_folder', type=str, default='exp/ckpts/')
    parser.add_argument('--filename', type=str, default='model.pt')
    parser.add_argument('--Pmean', type=float, default=-1.2)
    parser.add_argument('--Pstd', type=float, default=1.2)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--sigma_min', type=float, default=0.002)
    parser.add_argument('--sigma_max', type=int, default=80.)
    parser.add_argument('--rho', type=int, default=7)
    args = parser.parse_args()
    
    net = UNet(args.img_resolution)
    device = torch.device(args.device)
    model = EDMPrecond(net).to(device)
    
    model.load_state_dict(torch.load(os.path.join(args.save_folder, args.filename)))
    
    t = torch.tensor([32. for _ in range(args.batch_size)], device=args.device)
    x = torch.ones([1,1,28,28], device=args.device)
    out = model.unet(x, t)
    print('\nFinal output: ', out.flatten()[0:16])