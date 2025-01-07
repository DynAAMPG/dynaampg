import torch
from torch_geometric.loader import DataLoader
from session_dataset import SessionDataset
from dynaampg import DynAAMPG
from config import *
from utils import *
import numpy as np


if __name__ == "__main__":
    batch_size = 32
    dk = 512
    C = 3
    num_layers = 3
    num_heads = 8
    dataset = ISCX_VPN_DATASET_DIR
    class_labels =iscx_vpn_get_unique_labels()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(SAVED_LOGITS_DIR):
        os.makedirs(SAVED_LOGITS_DIR)

    dataset = SessionDataset(root=dataset, class_labels=class_labels)
    torch.manual_seed(12345)
    dataset = dataset.shuffle()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = DynAAMPG(input_dim=dataset.num_node_features, hidden_dim=dk, output_dim=dataset.num_classes, num_layers=num_layers, num_heads=num_heads, C=C,  model_state_path=BEST_MODEL_STATE_PATH_ISCX_VPN)
    model.to(device)

    labels = []
    actual_logits_pca = torch.tensor([])
    actual_logits_tsne = torch.tensor([])
    modified_logits_pca = torch.tensor([])
    modified_logits_tsne = torch.tensor([])

    for data in loader:
        data = data.to(device)
        _ = model(data)

        labels_batch = data.y.detach().cpu().numpy()

        for label in labels_batch:
            label_index = int((label == 1.0).nonzero()[0][0])            
            labels.append(label_index)

        actual_logit = model.actual_logits.detach().cpu()
        modified_logit = model.modified_logits.detach().cpu()

        reduced_actual_logit_pca = reduce_dimentions(actual_logit, method='PCA', n_components=2)
        reduced_actual_logit_tsne = reduce_dimentions(actual_logit, method='t-SNE', n_components=2)
        reduced_modified_logit_pca = reduce_dimentions(modified_logit, method='PCA', n_components=2)
        reduced_modified_logit_tsne = reduce_dimentions(modified_logit, method='t-SNE', n_components=2)
        
        # labels = torch.cat((labels, data.y.detach().cpu()), 0)

        actual_logits_pca = torch.cat((actual_logits_pca, reduced_actual_logit_pca), 0)
        actual_logits_tsne = torch.cat((actual_logits_tsne, reduced_actual_logit_tsne), 0)
        modified_logits_pca = torch.cat((modified_logits_pca, reduced_modified_logit_pca), 0)
        modified_logits_tsne = torch.cat((modified_logits_tsne, reduced_modified_logit_tsne), 0)
        
    
    labels = np.array(labels)
    actual_logits_pca = actual_logits_pca.numpy()
    actual_logits_tsne = actual_logits_tsne.numpy()
    modified_logits_pca = modified_logits_pca.numpy()
    modified_logits_tsne = modified_logits_tsne.numpy()

    np.save(os.path.join(SAVED_LOGITS_DIR, "actual_logits_pca.npy"), actual_logits_pca) 
    np.save(os.path.join(SAVED_LOGITS_DIR, "actual_logits_tsne.npy"), actual_logits_tsne)
    np.save(os.path.join(SAVED_LOGITS_DIR, "modified_logits_pca.npy"), modified_logits_pca)
    np.save(os.path.join(SAVED_LOGITS_DIR, "modified_logits_tsne.npy"), modified_logits_tsne)
    np.save(os.path.join(SAVED_LOGITS_DIR, "labels.npy"), labels)


