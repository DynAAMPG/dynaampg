import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dynaampg import DynAAMPG
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
from config import *
from utils import *
# Training loop
def train(train_loader, model, optimizer, criterion, device):
    model = model.to(device)
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Testing loop
@torch.no_grad()
def test(test_loader, model, criterion, device):
    model = model.to(device)
    model.eval()
    correct = 0
    total_loss = 0
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        loss = criterion(out, data.y)
        total_loss += loss.item()
        pred = out.argmax(dim=1)
        correct += int((pred == data.y.argmax(dim=1)).sum())
    return total_loss / len(test_loader), correct / len(test_loader.dataset)



if __name__ == "__main__":

    batch_size = 32
    epochs = 500
    dk = 512
    C = 3
    num_layers = 3
    num_heads = 8
    dataset = ISCX_VPN_DATASET_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(SAVED_MODELS_DIR):
        os.mkdir(SAVED_MODELS_DIR)
    
    if os.path.exists(TENSORBOARD_LOG_DIR):
        shutil.rmtree(TENSORBOARD_LOG_DIR)

  
    writer = SummaryWriter()

    dataset = SessionDataset(root=dataset, class_labels=iscx_vpn_get_unique_labels())
    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    # Split dataset into train and test
    train_dataset = dataset[:int(len(dataset) * 0.7)]
    test_dataset = dataset[int(len(dataset) * 0.7):]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    model = DynAAMPG(input_dim=dataset.num_node_features, hidden_dim=dk, output_dim=dataset.num_classes, num_layers=num_layers, num_heads=num_heads, C=C)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()


    # Main training and testing process
    max_train_acc = 0.0
    for epoch in range(1, epochs+1):
        train_loss = train(train_loader, model, optimizer, criterion, device)
        _, train_acc = test(train_loader, model, criterion, device)
        val_loss, val_acc = test(test_loader, model, criterion, device)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        writer.add_scalars('Loss', {'Train Loss':train_loss, 'Val Loss':val_loss}, epoch)
        writer.add_scalars('Accuracy', {'Train Acc':train_acc, 'Val Acc':val_acc} , epoch)

        if train_acc > max_train_acc:
            torch.save(model.state_dict(), os.path.join(SAVED_MODELS_DIR, "gformer_model_weights_iscx_vpn_" + str(epoch) + ".pth"))
            max_train_acc = train_acc


    writer.flush()
    writer.close()