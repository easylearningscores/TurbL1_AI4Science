 # <p align=center> Turb-L1: Achieving Long-term Turbulence Tracing By Tackling Spectral Bias</p>
<p align="center" width="100%">
  <img src='Figures/nips_t1.png' width="100%">
</p>

This repo is the official PyTorch implementation of Turb-L1.


## 📑 Data
| Dataset       | Task                                    | Geometry        | Link                                                         |  Original Data shape | 
| ------------- | --------------------------------------- | --------------- | ------------------------------------------------------------ |------------------------------------------------------------
| Decaying Isotropic Turbulence| Predict future fluid vorticity          | Regular Grid    | [[Hugging Face]](https://huggingface.co/datasets/scaomath/navier-stokes-dataset/blob/main/McWilliams2d_fp32_128x128_N1280_Re5000_T100.pt) | (1280, 100, 128, 128) |
| Forced Isotropic Turbulence | redict future fluid vorticity           | Regular Grid    | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |  (1200, 64, 64, 20)  |


Here's a brief description of the datasets:

**Dataset 1**  
- Original dimensions: 1,280 samples × 100 timesteps  
- Spatial resolution: 128×128  
- Training strategy: 1-step prediction (1→1)  
- Inference strategy: Multi-step prediction (1→99)  
- Variable: Vorticity (single variable)

**Dataset 2**  
- Original dimensions: 1,200 samples × 20 timesteps  
- Spatial resolution: 64×64  
- Training strategy: 1-step prediction (1→1)  
- Inference strategy: Multi-step prediction (1→19)  
- Variable: Vorticity (single variable)


The expected output dimensions of the dataloader are as follows:
```bash
# --- Testing DataLoader Output ---

# Train DataLoader
train_input_shape = (32, 1, 1, 64, 64)  # (B, T_input, C, H, W)
train_target_shape = (32, 1, 1, 64, 64)  # (B, T_target, C, H, W)
batch_size = 32                          # B
input_timesteps = 1                      # T_input
channels = 1                             # C
height = 64                              # H
width = 64                               # W

# Validation DataLoader
val_input_shape = (32, 1, 1, 64, 64)    # (B, T_input, C, H, W)

# Test DataLoader
test_input_shape = (32, 1, 1, 64, 64)   # (B, T_input, C, H, W)
```
  
