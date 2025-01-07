import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from dynaampg import DynAAMPG
from session_dataset import SessionDataset
from torch.utils.tensorboard import SummaryWriter
import shutil
import os
from config import *
from models.base_model import BaseModel
from gram_matrix import GRAM_TYPE
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



def train_loop(model_name, model, train_loader, test_loader, device, writer, epochs):

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = torch.nn.CrossEntropyLoss()

        print(f'Training {model_name}...')
        max_train_acc = 0.0
        for epoch in range(1, epochs+1):
            train_loss = train(train_loader, model, optimizer, criterion, device)
            _, train_acc = test(train_loader, model, criterion, device)
            val_loss, val_acc = test(test_loader, model, criterion, device)
            print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            writer.add_scalars(f'{model_name} Loss', {'Train Loss':train_loss, 'Val Loss':val_loss}, epoch)
            writer.add_scalars(f'{model_name} Accuracy', {'Train Acc':train_acc, 'Val Acc':val_acc} , epoch)

            if train_acc > max_train_acc:
                torch.save(model.state_dict(), os.path.join(SAVED_MODELS_DIR, "{model_name}_weights_" + str(epoch) + ".pth"))
                max_train_acc = train_acc





if __name__ == "__main__":

    # Load Dataset
    batch_size = 32
    epochs = 500
    dk = 512
    num_layers = 3
    num_heads = 8
    dataset = ISCX_VPN_DATASET_DIR

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(SAVED_MODELS_DIR):
        os.mkdir(SAVED_MODELS_DIR)
    
    if os.path.exists(TENSORBOARD_LOG_DIR):
        shutil.rmtree(TENSORBOARD_LOG_DIR)
  
    writer = SummaryWriter()

    dataset_id = SessionDataset(root=dataset, class_labels=iscx_vpn_get_unique_labels(), exclude_classes=['email'])
    dataset_ood = SessionDataset(root=dataset, class_labels=iscx_vpn_get_unique_labels(), exclude_classes=["chat", "streaming", "file_transfer", "voip", "p2p","vpn_email", "vpn_chat", "vpn_streaming", "vpn_file_transfer", "vpn_voip", "vpn_p2p"])
    torch.manual_seed(12345)
    dataset_id = dataset_id.shuffle()
    dataset_ood = dataset_ood.shuffle()

    # Split dataset into train and test
    train_dataset = dataset_id[:int(len(dataset_id) * 0.7)]
    test_dataset = dataset_id[int(len(dataset_id) * 0.7):]

    train_id_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_id_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_ood_loader = DataLoader(dataset_ood, batch_size=batch_size, shuffle=False)


    # Train BaseModel + DynAAM (C=3) + Penalized Gram Matrix
    C = 3
    G = GRAM_TYPE.PENALIZED_GRAM

    dynaam_c3_penalized_gram = DynAAMPG(input_dim=dataset.num_node_features, hidden_dim=dk, output_dim=dataset.num_classes, num_layers=num_layers, num_heads=num_heads, C=C, G=G)
    train_loop("DynAAM_C3_PenalizedGram", dynaam_c3_penalized_gram, train_id_loader, test_id_loader, device, writer, epochs)

    _, ood_accuracy = test(test_ood_loader, dynaam_c3_penalized_gram, criterion = torch.nn.CrossEntropyLoss(), device=device)

    writer.flush()
    writer.close()