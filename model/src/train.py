import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN

transform = transforms.Compose([transforms.Resize((28,28)), transforms.ToTensor()])
train_data = datasets.FakeData(transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(num_classes=10).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(2):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
torch.save(model.state_dict(), '../outputs/simple_cnn.pth')
