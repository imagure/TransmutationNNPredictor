from Model import *
from FeatureDataset import FeatureDataset

N, D_in, H, D_out = 64, 18, 10, 4

dataset = FeatureDataset("torch_dataset.csv")
trainloader = torch.utils.data.DataLoader(dataset, batch_size=N, shuffle=True)

net = Model(D_in, H, D_out)

criterion = torch.nn.MSELoss(reduction='sum')
optimizer = torch.optim.SGD(net.parameters(), lr=1e-4)

epochs = 500
for epoch in range(epochs):
    running_loss=0

    for features, labels in trainloader:
        y_pred = net(features)

        # Compute and print loss
        loss = criterion(y_pred, labels)
        if epoch % 100 == 99:
            print("Epoch: ", epoch+1, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

PATH = './trained_model.pth'
torch.save(net.state_dict(), PATH)
