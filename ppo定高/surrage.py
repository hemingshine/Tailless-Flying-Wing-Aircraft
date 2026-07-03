#coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

print("===========================================")
print("  启动工业级 PyTorch 气动代理模型训练")
print("===========================================")

# 1. 加载第一步采好的数据
dataset_file = 'aero_dataset.npz'
if not os.path.exists(dataset_file):
    print(f"❌ 找不到数据文件 '{dataset_file}'，请先运行 1_sample_data.py！")
    exit()

print("\n[1/3] 正在加载并预处理数据集...")
data = np.load(dataset_file)
X_valid = data['X']
Y_valid = data['Y']
print(f"成功加载 {len(X_valid)} 条飞行状态数据。")

# 转换为 Tensor
X_tensor = torch.FloatTensor(X_valid)
Y_tensor = torch.FloatTensor(Y_valid)

# 标准化 (非常关键)
x_mean = X_tensor.mean(dim=0)
x_std = X_tensor.std(dim=0) + 1e-6
y_mean = Y_tensor.mean(dim=0)
y_std = Y_tensor.std(dim=0) + 1e-6

X_norm = (X_tensor - x_mean) / x_std
Y_norm = (Y_tensor - y_mean) / y_std

# 划分数据集 (90% 训练，10% 验证)
dataset = TensorDataset(X_norm, Y_norm)
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2048)

# 2. 构建深度神经网络
print("\n[2/3] 构建神经网络与优化器...")
class AeroSurrogate(nn.Module):
    def __init__(self):
        super(AeroSurrogate, self).__init__()
        # 残差块定义
        class ResBlock(nn.Module):
            def __init__(self, in_dim, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),  # 批归一化：稳定训练、加速收敛
                    nn.GELU(),
                    nn.Dropout(0.2)
                )
                # 残差连接的维度适配
                self.shortcut = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
            
            def forward(self, x):
                return self.net(x) + self.shortcut(x)  # 残差连接
        
        # 分层残差网络（宽度递减，更符合气动数据的非线性拟合规律）
        self.net = nn.Sequential(
            ResBlock(9, 256),
            ResBlock(256, 512),
            ResBlock(512, 256),
            ResBlock(256, 128),
            nn.Linear(128, 6)  # 输出6个气动参数
        )

    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前使用的计算硬件: {device}")

model = AeroSurrogate().to(device)
criterion = nn.HuberLoss(delta=0.1) 
optimizer = optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999),weight_decay=1e-4)

# 动态学习率调度器：如果验证集 Loss 不下降，自动砍半学习率
warmup_epochs = 10
total_epochs = 300
scheduler_warmup = LinearLR(optimizer, start_factor=1e-4, total_iters=warmup_epochs)
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs, eta_min=1e-5)
scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
# 3. 开始炼丹
print("\n[3/3] 开始训练...")
EPOCHS = 300 
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    train_loss /= len(train_loader.dataset)
    
    # 验证集评估
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    
    # 调度器步进
    scheduler.step()
    
    # 抓取最优模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_surrogate_weights.pth')
    
    if (epoch + 1) % 10 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch [{epoch+1:03d}/{EPOCHS}] | LR: {current_lr:.6f} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

print(f"\n🎉 训练圆满结束！最优 Val Loss 突破至: {best_val_loss:.6f}")

# 4. 打包导出
model.load_state_dict(torch.load('best_surrogate_weights.pth'))
model.to('cpu') 

print("\n正在将模型与标准化参数打包...")
save_dict = {
    'model_state_dict': model.state_dict(),
    'x_mean': x_mean, 'x_std': x_std,
    'y_mean': y_mean, 'y_std': y_std
}
torch.save(save_dict, 'aero_surrogate.pth')
print("✅ 终极武器 'aero_surrogate.pth' 锻造成功！请准备接入强化学习模拟器！")