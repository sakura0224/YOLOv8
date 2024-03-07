import matplotlib.pyplot as plt
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


# 定义自动编码器模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # 编码器层
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2),  # 解码器层
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# 加载 MNIST 数据集并添加噪声
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

# 转换为 FloatTensor 类型并添加噪声
noise_factor = 0.5
train_noisy_data = train_dataset.data.float() / 255.  # 转换为浮点数类型
train_noisy_data += noise_factor * torch.randn_like(train_noisy_data)  # 添加噪声

test_noisy_data = test_dataset.data.float() / 255.
test_noisy_data += noise_factor * torch.randn_like(test_noisy_data)

# 数据加载器
batch_size = 128
train_loader = torch.utils.data.DataLoader(train_noisy_data.unsqueeze(1), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_noisy_data.unsqueeze(1), batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for data in train_loader:
        img = data.float().to(device)
        output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# 测试模型
with torch.no_grad():
    for data in test_loader:
        img = data.float().to(device)
        output = model(img)
        break

# 显示结果
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # 原始图像
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(test_noisy_data[i].cpu().numpy().squeeze(), cmap='gray')
    plt.title("Noisy")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # 去噪图像
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(output[i].cpu().numpy().squeeze(), cmap='gray')
    plt.title("Denoised")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
