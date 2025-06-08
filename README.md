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

