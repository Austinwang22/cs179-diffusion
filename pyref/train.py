import os 
import torch 
from tqdm import tqdm
from datasets import get_mnist_loaders
from unet import EDMPrecond, UNet
from torch.optim import Adam
from argparse import ArgumentParser


#   To move the Python training loop, including data loading and optimizer steps, into a fully C++/CUDA implementation,
#   parallelize both the DataLoader and the optimizer.

#   1) DATA LOADER PARALLELISM (HOST SIDE):
#      - Use a thread pool to spin up multiple worker threads (num_workers) that each:
#          – Read image batches from disk into host memory.
#          – Enqueue completed batches into a lock‐free circular buffer (producer/consumer queue).
#      - The main training thread (consumer) dequeues ready batches and immediately issues
#        asynchronous cudaMemcpyAsync() calls to transfer data to device memory on dedicated CUDA streams,
#        overlapping data transfers with GPU computation.
#      - This pipelining ensures that I/O, CPU transforms, host→device copies, and GPU compute all run concurrently.

#   2) OPTIMIZER PARALLELISM (DEVICE SIDE):
#      • Represent each learnable parameter tensor as a contiguous region in GPU global memory.
#      • After the backward pass, gradients for every parameter are resident on the device.
#      • Launch a single fused CUDA kernel (or one per parameter group) that:
#          – Reads gradient and current weight for each element.
#          – Applies the optimizer update rule.
#          – Writes updated weight (and velocity buffer if used) back to global memory.
#        • Use one CUDA thread per tensor element (or block‐wide tiling) to maximize throughput.

#   Once we have the CUDA code for the DataLoader and the optimizer, we can parallelize the training of our model.

#   TESTING: to verify the correctness of our GPU implementation we can initialize both the PyTorch and CUDA models
#   with the same weights and the same optimizer hyperparameters, then train for one iteration. We can then verify
#   that the model weights should still be the same (within some small threshold). 


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