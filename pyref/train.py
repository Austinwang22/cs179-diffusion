import os 
import torch 
from tqdm import tqdm
from datasets import get_mnist_loaders
from unet import EDMPrecond, UNet
from torch.optim import Adam
from argparse import ArgumentParser

def edm_loss(batch, net:EDMPrecond, Pmean=-1.2, Pstd=1.2):
    x0 = batch
    B = x0.shape[0]
    
    sigma_data = net.sigma_data

    log_sigma = torch.randn(B, device=x0.device) * Pstd + Pmean
    sigma = torch.exp(log_sigma)

    noise = torch.randn_like(x0)
    x_noisy = x0 + noise * sigma.view(B, 1, 1, 1)

    denoised = net(x_noisy, sigma)

    lam = (sigma**2 + sigma_data**2) / (sigma * sigma_data)**2
    lam = lam.view(B, 1, 1, 1)
    loss = (lam * (denoised - x0).pow(2)).mean()

    return loss

def train(model, train_loader, optim, num_epochs, device='cuda', Pmean=-1.2, Pstd=1.2,
          save_folder='exp/ckpts/', filename='model.pt'):
    model.train()
    pbar = tqdm(range(num_epochs))
    for e in pbar:
        
        total_loss = 0.0
        
        for data, y in train_loader:
            optim.zero_grad()
            data = data.to(device)
            loss = edm_loss(data, model, Pmean, Pstd)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        pbar.set_description(f'Epoch {e + 1}: {avg_loss}')
    
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, filename)
    torch.save(model.state_dict(), save_path)
    
    
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
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    
    net = UNet(args.img_resolution)
    device = torch.device(args.device)
    model = EDMPrecond(net).to(device)
    
    train_loader, test_loader = get_mnist_loaders(args.batch_size)
    optim = Adam(model.parameters(), args.lr)
    
    train(model, train_loader, optim, args.num_epochs)
    print(f'Model weights saved at {os.path.join(args.save_folder, args.filename)}')