import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import matplotlib.pyplot as plt
from config import *

plt.rcParams.update({
    "text.usetex": True,
    "font.family": 'Times New Roman'})


train_acc_path = os.path.join(SAVED_TRAIN_VAL_CURVES_DIR, 'train_acc.csv')
val_acc_path = os.path.join(SAVED_TRAIN_VAL_CURVES_DIR, 'val_acc.csv')
train_loss_path = os.path.join(SAVED_TRAIN_VAL_CURVES_DIR, 'train_loss.csv')
val_loss_path = os.path.join(SAVED_TRAIN_VAL_CURVES_DIR, 'val_loss.csv')


train_acc_data = pd.read_csv(train_acc_path)
val_acc_data = pd.read_csv(val_acc_path)
train_loss_data = pd.read_csv(train_loss_path)
val_loss_data = pd.read_csv(val_loss_path)


       
plt.figure(figsize=(10, 5))
plt.plot(train_acc_data['Value'], label='Train Accuracy')
plt.plot(val_acc_data['Value'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('visualization/train_val_acc.png')
plt.show()


plt.figure(figsize=(10, 5))
plt.plot(train_loss_data['Value'], label='Train Loss')
plt.plot(val_loss_data['Value'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('visualization/train_val_loss.png')
plt.show()