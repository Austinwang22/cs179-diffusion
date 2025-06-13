import torch 
import torch.nn.functional as F
import math

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
        t_proj = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.act(self.conv1(x) + t_proj)
        h = self.act(self.conv2(h) + t_proj)
        h = self.downsample(h)
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
        h = self.conv1(x)
        h += t_proj
        h = self.act(h)
        h = self.conv2(h)
        h += t_proj
        h = self.act(h)  
        # print(f'first 16 elements of relu output: {h.flatten()[0:16]}')
        h = self.upsample(h)
        # print(f'first 16 elements of upconv output: {h.flatten()[0:16]}')
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
            
        # print(f'encoder output: {x.flatten()[0:16]}')
            
        x = self.bottleneck(x, t_emb)
        
        # print(f'\nbottleneck output: {x.flatten()[0:16]}')
        
        for i, dec_block in enumerate(self.dec_blocks):
            x = dec_block(x, t_emb, enc_outs.pop())
            # print(f'\ndecoder {i} output: {x.flatten()[0:16]}')
        
        
        # Print weights of out_conv
        # print(f'\nout_conv weights: {self.out_conv.weight.flatten()[0:16]}')    
        
        out = self.out_conv(x)
        # print(f'\noutconv output: {out.flatten()[0:16]}')
        return out
    
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
    