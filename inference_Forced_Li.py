import torch
import numpy as np
import yaml
import os
import scipy.io
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

# ================== 基础配置 ==================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('config_Forced_Li.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 配置参数解析
selected_model = config['selected_model']
model_config = config['models'][selected_model]
data_config = config['datas'][selected_model]
training_config = config['trainings'][selected_model]
logging_config = config['loggings'][selected_model]

# 路径配置
backbone = logging_config['backbone']
checkpoint_dir = logging_config['checkpoint_dir']
result_dir = logging_config['result_dir']
os.makedirs(result_dir, exist_ok=True)

# ================== 数据加载 ==================
def load_raw_data(data_path):
    """直接加载原始MAT数据并进行预处理"""
    mat_data = scipy.io.loadmat(data_path)
    print("数据文件包含的变量:", mat_data.keys())
    
    if 'u' not in mat_data:
        raise ValueError("数据文件中未找到'u'变量")
    
    # 原始维度转换 (N, H, W, T) -> (N, T, C, H, W)
    U_original = mat_data['u']
    print("原始数据维度:", U_original.shape)
    
    # 维度调整并转换为Tensor
    U_processed = torch.from_numpy(np.transpose(U_original, (0, 3, 1, 2)))  # [N, T, H, W]

    U_processed = U_processed.unsqueeze(2).float()  # 添加通道维度 [N, T, 1, H, W]
    return U_processed

# 加载完整数据集
full_data = load_raw_data(data_config['data_path'])

# 数据集分割参数
train_ratio = data_config.get('train_split_ratio', 0.8)
val_ratio = data_config.get('val_split_ratio', 0.1)
total_samples = full_data.size(0)

# 计算分割点
train_end = int(train_ratio * total_samples)
val_end = train_end + int(val_ratio * total_samples)

# 提取测试集数据 (保持原始时间序列完整性)
test_data = full_data[val_end:]  # [N_test, T_total=20, C=1, H, W]
print(f"\n测试集数据维度: {test_data.shape}")

# ================== 模型初始化 ==================
from model.turb_l1 import TurbL1
from model_baselines.fno import FNO2d 
from model_baselines.dit import Dit
from model_baselines.simvp import SimVP
from model_baselines.cno import CNO
from model_baselines.mgno import MgNO
from model_baselines.lsm import LSM
from model_baselines.pastnet import PastNetModel
from model_baselines.resnet import ResNet
from model_baselines.unet import U_net
from model_baselines.vit import VisionTransformer
from model_baselines.convlstm import ConvLSTM_NS

model_dict = {
    'TurbL1': TurbL1,
    'FNO': FNO2d,
    'DiT': Dit,
    'SimVP': SimVP,
    'CNO': CNO,
    'MGNO': MgNO,
    'LSM': LSM,
    'PastNet': PastNetModel,
    'ResNet': ResNet,
    'U_net': U_net,
    'ViT': VisionTransformer,
    'ConvLSTM': ConvLSTM_NS,
}

# 模型实例化
ModelClass = model_dict[selected_model]
model = ModelClass(**model_config['parameters']).to(device)
model.eval()
print(f"\n成功加载模型: {selected_model}")

# ================== 加载训练权重 ==================
def load_model_weights(model, checkpoint_path):
    """处理可能的并行训练权重"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if all(k.startswith('module.') for k in checkpoint.keys()):
        # 去除并行训练的前缀
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            new_state_dict[k[7:]] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(checkpoint)
    print(f"成功加载模型权重: {checkpoint_path}")

# 自动选择最佳检查点
checkpoint_path = os.path.join(checkpoint_dir, f"{backbone}_best_model.pth")
if not os.path.exists(checkpoint_path):
    checkpoint_path = os.path.join(checkpoint_dir, f"{backbone}_epoch_50.pth")

load_model_weights(model, checkpoint_path)

# ================== 推理参数配置 ==================
input_length = data_config['input_length']  # 应为1
target_length = 19  # 总时间步20 = 1输入 + 19预测
downsample_factor = data_config.get('downsample_factor', 1)
original_H, original_W = test_data.shape[-2], test_data.shape[-1]
H = original_H // downsample_factor
W = original_W // downsample_factor

# ================== 推理核心逻辑 ==================
def autoregressive_predict(sample_data, model, steps):
    """自回归预测函数"""
    predictions = []
    current_input = sample_data[:input_length]  # 初始输入 [T=1, C=1, H, W]
    
    for step in range(steps):
        # 准备模型输入 [1, T, C, H, W]
        model_input = current_input.unsqueeze(0).to(device).float()
        
        # 模型推理
        with torch.no_grad():
            pred = model(model_input)
        
        # 提取最后一个预测步 [C=1, H, W]
        last_pred = pred[0, -1]
        predictions.append(last_pred.cpu())
        
        # 更新输入序列
        current_input = torch.cat([current_input[1:].to(device), last_pred.unsqueeze(0).to(device)], dim=0)
    
    return torch.stack(predictions)

# ================== 结果存储与可视化 ==================
def visualize_prediction(input_seq, pred_seq, true_seq, save_dir):
    """生成预测过程可视化"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 合并输入和预测序列
    full_seq = torch.cat([input_seq, pred_seq], dim=0)
    
    plt.figure(figsize=(18, 6))
    for t in range(0, full_seq.size(0), 5):  # 每5步保存一帧
        plt.clf()
        
        # 输入序列
        plt.subplot(1, 3, 1)
        plt.imshow(input_seq.squeeze().numpy(), cmap='coolwarm')
        plt.title(f'Initial Input (t=0)')
        
        # 预测结果
        plt.subplot(1, 3, 2)
        plt.imshow(full_seq[t].squeeze().numpy(), cmap='coolwarm')
        plt.title(f'Prediction (t={t})')
        
        # 真实值
        plt.subplot(1, 3, 3)
        plt.imshow(true_seq[t].squeeze().numpy(), cmap='coolwarm')
        plt.title(f'Ground Truth (t={t})')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"step_{t:03d}.png"))
        plt.close()

# ================== 主推理循环 ==================
# 结果容器
all_inputs = []
all_preds = []
all_targets = []

# 进度条设置
pbar = tqdm(total=test_data.size(0), desc="Processing Samples")

for sample_idx in range(test_data.size(0)):
    # 当前样本数据 [T=20, C=1, H, W]
    full_sequence = test_data[sample_idx]
    
    # 下采样处理
    if downsample_factor > 1:
        full_sequence = full_sequence[..., ::downsample_factor, ::downsample_factor]
    
    # 分割输入和真实值
    input_sequence = full_sequence[:input_length]          # [1, 1, H, W]
    ground_truth = full_sequence[input_length:input_length+target_length]  # [19, 1, H, W]
    
    # 执行自回归预测
    predictions = autoregressive_predict(input_sequence, model, target_length)
    
    # 存储结果
    all_inputs.append(input_sequence.numpy())
    all_preds.append(predictions.numpy())
    all_targets.append(ground_truth.numpy())
    
    # 生成可视化
    sample_viz_dir = os.path.join(result_dir, "visualizations", f"sample_{sample_idx:04d}")
    visualize_prediction(input_sequence.squeeze(1), predictions.squeeze(1), ground_truth.squeeze(1), sample_viz_dir)
    
    pbar.update(1)

pbar.close()

# ================== 结果保存 ==================
print("\n保存结果文件...")
np.savez_compressed(
    os.path.join(result_dir, f"{backbone}_predictions.npz"),
    inputs=np.concatenate(all_inputs, axis=0),
    predictions=np.concatenate(all_preds, axis=0),
    targets=np.concatenate(all_targets, axis=0)
)

print("\n✅ 推理完成! 结果保存至:", os.path.abspath(result_dir))
