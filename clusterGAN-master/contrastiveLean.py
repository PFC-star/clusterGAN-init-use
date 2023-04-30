import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 图像变换（可自行根据需求修改）
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 加载数据集，并分成两个子集
train_data = dsets.ImageFolder(root='path/to/train/data', transform=transform)
test_data = dsets.ImageFolder(root='path/to/test/data', transform=transform)

train_data1 = train_data[::2]
train_data2 = train_data[1::2]

test_data1 = test_data[::2]
test_data2 = test_data[1::2]

data1 = DataLoader(train_data1, batch_size=32, shuffle=True)
data2 = DataLoader(train_data2, batch_size=32, shuffle=True)

test_data1 = DataLoader(test_data1, batch_size=32, shuffle=False)
test_data2 = DataLoader(test_data2, batch_size=32, shuffle=False)

# 定义对比学习模型
class Contrastive(nn.Module):
    def __init__(self):
        super(Contrastive, self).__init__()
        # 加载预训练模型
        self.resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50', pretrained=True)
        # 替换最后一层全连接层
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, 512)
        # 添加对比学习头部
        self.linear = nn.Linear(512, 128)

    def forward_once(self, x):
        output = self.resnet(x)
        output = self.linear(output)
        return output

    def forward(self, x1, x2):
        out1 = self.forward_once(x1)
        out2 = self.forward_once(x2)
        return out1, out2

# 训练模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Contrastive().to(device)

criterion = nn.CosineEmbeddingLoss(margin=0.5)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
num_epochs = 10

for epoch in range(num_epochs):
    for i, ((image1, _), (image2, _)) in enumerate(zip(data1, data2)):
        image1 = image1.to(device)
        image2 = image2.to(device)

        optimizer.zero_grad()
        out1, out2 = model(image1, image2)
        target = torch.ones(out1.size()[0]).to(device)
        loss = criterion(out1, out2, target)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print("Epoch: ", epoch+1, " Iteration: ", i+1, " Loss: ", loss.item())

# 测试模型
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for i, ((image1, _), (image2, _)) in enumerate(zip(test_data1, test_data2)):
        image1 = image1.to(device)
        image2 = image2.to(device)

        out1, out2 = model(image1, image2)
        cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        sim_score = cos_sim(out1, out2)

        predicted = torch.round(sim_score).to(device)
        total += image1.size(0)
        correct += (predicted == torch.ones_like(predicted)).sum().item()

print('Accuracy: {:.2f}%'.format(correct / total * 100))