import torch 
import torch.nn.functional as F
import math

#   To re-implement this PyTorch UNet + EDM pipeline in C++/CUDA from scratch, parallelize every tensor operation
#   by launching CUDA kernels or using highly-optimized cuBLAS/cuDNN routines.

#   1) HOST SETUP
#      - Allocate device buffers for inputs, weights, biases, and intermediate activations.
#      - Transfer model parameters (conv kernels, linear weights, etc.) to GPU global memory.
#      - For each forward pass, copy input batch to device (or use persistent GPU memory).

#   2) LINEAR LAYERS (e.g. time_proj, MLP)
#      - Represent as matrix–vector or matrix–matrix multiplies.
#      - Launch a GEMM kernel (cuBLAS sgemm) or a custom tiled kernel:
#          – GridDim = ceil(M/TILE_M) × ceil(N/TILE_N), BlockDim = TILE_M × TILE_N threads.
#          – Each thread-block loads a TILE_M×K submatrix of A (input) and a K×TILE_N submatrix of B (weights)
#            into shared memory, performs partial dot products, then writes TILE_M×TILE_N results to global memory.
#      - Follow with elementwise bias add and activation (SiLU): one thread per output element; uses fast CUDA math.

#   3) 2D CONVOLUTION (Conv2D and ConvTranspose2d)
#      - Convolution kernel
#          – Kernel launch: GridDim = (N, out_channels, H_out, W_out), BlockDim = (tile_W, tile_H) threads.
#          – Each thread-block:
#              - Loads a tile of the input feature map into shared memory (including halo for kernel radius).
#              - Loads corresponding filter weights into registers or shared memory.
#              - Each thread computes one (or a small tile of) output pixel by sliding the kernel window.
#              - Accumulates over input channels.
#          – Use loop unrolling and warp-level reductions to speed up the inner product.
#      - For ConvTranspose2d (upsampling):
#          – Similar to convolution but reversed indexing: each thread scatters its input into the appropriate output locations,
#            or implement deconvolution by swapping roles of input/output in a direct conv kernel.

#   4) ACTIVATIONS (ReLU, SiLU)
#      - Elementwise operations: launch one CUDA thread per element.
#      - Each thread loads one or more values from global memory, applies the activation, writes back.
#      - Perfectly parallel with no inter-thread communication.

#   5) POOLING (MaxPool2d)
#      - For 2×2 max-pool:
#          – GridDim = (N, C, H_out, W_out), BlockDim = (pool_H, pool_W) threads.
#          – Each thread reads its 2×2 patch from input and computes max.
#          – Write one scalar to output per thread.

#   6) CONCATENATION (skip connections)
#      - Treated as a memory copy kernel:
#          – GridDim = (batch, channels, H, W) split across two sources.
#          – Each thread writes from either input A or input B into the correct region of the output tensor.

#   7) ELEMENTWISE OPERATIONS (adding time embeddings, c_skip*x, c_out*raw)
#      - Use a broadcast-aware kernel: each thread multiplies or adds a scalar or vector across a tile of the tensor.

#   TESTS: we will verify that the UNet with CUDA has the same output as the UNet in PyTorch when running a forward pass
#   from the same input tensors. We will verify that the specific encoder and decoder blocks as well have the same outputs
#   from given input tensors. 


class TimeEmbedding(torch.nn.Module):
    
    def __init__(self, emb_dim, mlp_dim=None):
        super().__init__()
        if mlp_dim is None:
            mlp_dim = emb_dim * 4
        self.emb_dim = emb_dim
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, mlp_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(mlp_dim, emb_dim)
        )
    
    def forward(self, t):
        '''
        t: tensor of shape [B] containing timesteps
        returns: FloatTensor of shape [B, emb_dim]
        '''
        device = t.device
        half_dim = self.emb_dim // 2
        exp_term = -math.log(10000) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * exp_term)
        args = t.unsqueeze(1).float() * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.emb_dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)

class EncoderBlock(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, t_emb_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1) 
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1) 
        self.act = torch.nn.ReLU()
        self.downsample = torch.nn.MaxPool2d(2)
        self.time_proj = torch.nn.Linear(t_emb_dim, out_channels)
        
    
    def forward(self, x, t_emb):
        print('\nForward Pass\n')
        print(f't_emb: ', t_emb.flatten()[0:7], t_emb.shape)
        t_proj = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        print(f'tbias: ', t_proj.flatten()[0:7], t_proj.shape)
        print(x.flatten()[0:16], x.shape)
        h = (self.conv1(x))
        print(h.flatten()[0:16], h.shape)
        h += t_proj
        print(h.flatten()[0:16], h.shape)
        h = self.act(h)
        print(h.flatten()[0:16], h.shape, '\n')
        h = (self.conv2(h))
        print(h.flatten()[0:16], h.shape)
        print(f'tbias: ', t_proj.flatten()[0:7], t_proj.shape)
        h += t_proj
        print(h.flatten()[0:16], h.shape)
        h = self.act(h)
        print(h.flatten()[0:16], h.shape)
        h = self.downsample(h)
        print(h.flatten()[0:16], h.shape)
        return h
         
    
class DecoderBlock(torch.nn.Module):
    
    def __init__(self, in_channels, out_channels, t_emb_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, 3, padding=1) 
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, 3, padding=1) 
        self.act = torch.nn.ReLU()
        self.upsample = torch.nn.ConvTranspose2d(out_channels, out_channels, 2, stride=2)
        self.time_proj = torch.nn.Linear(t_emb_dim, out_channels)
    
    def forward(self, x, t_emb, skip):
        x = torch.cat([x, skip], dim=1)
        t_proj = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.conv1(x) + t_proj)
        h = self.act(self.conv2(h) + t_proj)
        h = self.upsample(h)
        return h
    
class Bottleneck(torch.nn.Module):
    def __init__(self, channels, t_emb_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.act  = torch.nn.ReLU()
        self.time_proj = torch.nn.Linear(t_emb_dim, channels)

    def forward(self, x, t_emb):
        t = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.conv1(x) + t)
        h = self.act(self.conv2(h) + t)
        return h

class UNet(torch.nn.Module):
    
    def __init__(self, img_resolution, enc_channels=[1, 64, 128], dec_channels=[128, 64], out_channels=1,
                 t_emb_dim=128, t_mlp_dim=None):
        super().__init__()
        self.img_resolution = img_resolution
        self.in_channels = enc_channels[0]
        self.out_channels = out_channels
        self.shape = (self.in_channels, img_resolution, img_resolution)
        
        self.time_emb = TimeEmbedding(t_emb_dim, t_mlp_dim)
        self.enc_blocks = torch.nn.ModuleList([EncoderBlock(enc_channels[i], enc_channels[i + 1], t_emb_dim) 
                                               for i in range(len(enc_channels) - 1)])
        self.bottleneck = Bottleneck(enc_channels[-1], t_emb_dim)
        self.dec_blocks = torch.nn.ModuleList()
        for i, o_channels in enumerate(dec_channels):
            skip_channels = enc_channels[-(len(enc_channels) - 2) - i]
            in_channels = (dec_channels[i - 1] if i > 0 else enc_channels[-1]) + skip_channels
            self.dec_blocks.append(DecoderBlock(in_channels, o_channels, t_emb_dim))
        self.out_conv = torch.nn.Conv2d(dec_channels[-1], out_channels, kernel_size=1)
    
    def forward(self, x, t):
        '''
        x: shape [B,C,H,W] tensor of images
        t: shape [B] tensor of timesteps
        '''
        t_emb = self.time_emb(t)
        enc_outs = []
        for enc_block in self.enc_blocks:
            out = enc_block(x, t_emb)
            enc_outs.append(out)
            x = out
        x = self.bottleneck(x, t_emb)
        for i, dec_block in enumerate(self.dec_blocks):
            x = dec_block(x, t_emb, enc_outs.pop())
        return self.out_conv(x)
    
class EDMPrecond(torch.nn.Module):
    '''
    Wraps a backbone UNet (F_theta) to implement the
    EDM denoiser D_theta(x; sigma) = c_skip x + c_out * F_theta(c_in x, ln(sigma))
    '''
    def __init__(self, backbone_unet, sigma_data=0.5):
        super().__init__()
        self.unet = backbone_unet
        self.sigma_data = sigma_data

    def forward(self, x_noisy: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        x_noisy: [B, C, H, W] noisy images x
        sigma:   [B] noise levels per sample
        returns: D_theta(x; sigma)
        """
        b = sigma.shape[0]
        sd2 = self.sigma_data**2

        c_skip = sd2 / (sigma**2 + sd2) 
        c_out  = sigma * self.sigma_data / torch.sqrt(sigma**2 + sd2) 
        c_in   = 1.0 / torch.sqrt(sigma**2 + sd2)  
        c_noise= torch.log(sigma) 

        c_skip = c_skip.view(b, 1, 1, 1)
        c_out  = c_out.view(b, 1, 1, 1)
        c_in   = c_in.view(b, 1, 1, 1)

        x_input = c_in * x_noisy

        raw = self.unet(x_input, c_noise)
        return c_skip * x_noisy + c_out * raw

# if __name__ == '__main__':
#     net = UNet(28)
#     model = EDMPrecond(net)
#     batch = torch.randn(128,1,28,28)
#     t = torch.randn(128)
#     out = model(batch, t)
#     print(out.shape)
    