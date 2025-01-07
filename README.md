# DynAAM-PG
![Alt text](assets/model.png?raw=true "Model")

## 1. Download and install Wireshark
* While installing wireshark, CHECK the following option: Install Npcap in WinPcap API-compatible mode
## 2. Set Wireshark to environment variable
* In start menu search environment
* Edit the system environment variables > Environment Variables
* In user variable double-click Path
* Click New
* Paste the Wireshark installation path: (C:\Program Files\Wireshark)
* Click Ok > OK

## 3. Download the datasets
### 3A. (ISCX-VPN-NonVPN-2016)

http://205.174.165.80/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/PCAPs/

* Download following files:
```
1. NonVPN-PCAPs-01.zip
2. NonVPN-PCAPs-02.zip
3. NonVPN-PCAPs-03.zip
4. VPN-PCAPS-01.zip
5. VPN-PCAPS-02.zip
```
### 3B. (VNAT-VPN)
https://archive.ll.mit.edu/datasets/vnat/VNAT_release_1.zip
* Download following files:
```
1. VNAT_release_1.zip
```

## 4. Prepare the datasets
### 4A. (ISCX-VPN-NonVPN-2016)

* Extract the following files in folder datasets/ISCX_VPN
```
NonVPN-PCAPs-01.zip
NonVPN-PCAPs-02.zip
NonVPN-PCAPs-03.zip
VPN-PCAPS-01.zip
VPN-PCAPS-02.zip
```

* Finally the dataset structure should look like:
```
-datasets
 - ISCX
  - aim_chat_3a.pcap
  - aim_chat_3b.pcap
	...
  - youtubeHTML5_1.pcap
  - vpn_aim_chat1a.pcap
  - vpn_aim_chat1b.pcap
	...
  - vpn_youtube_A.pcap
```
### 4B. (VNAT-VPN)
* Extract the following files in folder datasets/VNAT
```
VNAT_release_1.zip
```

## 5. Create conda environment and install packages
* Open miniconda and run following commands one-by-one:
```
create conda -n gformer python=3.11
conda activate gformer
pip install matplotlib, scapy, tqdm, pandas, numpy
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install tensorboard
```

## 6. Convert pcapng files to pcap and split sessions
* Download splitcap utility
   https://www.netresec.com/?page=SplitCap
* Place SplitCap.exe file in .\gformer\
* Open project folder in VSCode
* Select python interpreter as "gformer"
* Run the script split_sessions.py

## 7. Process packets

* Run the script process_data.py
* This script will perform following tasks:
```
Read each sessions pcap file and extract first 10 packets.
For each packet, it
  Removes the Ethernet header.
  Masks source and destination IPs.
  Pads the UDP packets with zeros.
  Convert each packet to byte format (0-255 values), if any packet is less than 1500 it pads with zeros.
  Normalize each byte value into the range of (0.0 - 1.0).
  Generates JSON file where each file represents a session graph.
```

## 8. Generate graphs
* Run the script generate_graphs.py
* This script will perform following tasks:
```
Move all JSON files in to "{dataset_path}\raw" directory
Generate graphs in "{dataset_path}\processed" directory

-{dataset_path}\raw
 - 1.json
 - 2.json
 ...
-{dataset_path}\processed
 - data.pt
 - pre_filter.pt
 - pre_transform.pt
```

## 9. Train the model
* Run the script train.py

![Alt text](assets/training.png?raw=true "Training model")

## 10. Launch tensorboard
* Open new powershell window from VS Code by clicking "+" button
* Run the following command
```
tensorboard --logdir=runs
```
```
Now CTRL+Click http://localhost:6006/ or copy this address in browser
```
![Alt text](assets/tensorboard.png?raw=true "Launch Tensorboard")
![Alt text](assets/graph3.png?raw=true "Tensorboard Training and Validation Accuracy Visualization")
![Alt text](assets/graph4.png?raw=true "Tensorboard Training and Validation Loss Visualization")




# Generate OOD dataset

1. Download the OOD dataset from following link:
```
https://drive.google.com/file/d/1h0Q55pba1zF8Q4D7xJvenscLodof4CDS/view?usp=sharing
```
2. Extract and place the dataset in datasets/OOD

3. Run the script 
```
generate_ood_dataset.py
```


# Plot All Gram Matrices

1. Download the pre-trained model from following link:
```
https://drive.google.com/file/d/1fgvWeshlc10raOoQfPfEXtsp73O4fa4L/view?usp=sharing
```
2. Extract and place the model in saved_models/

3. Run the script
```
 visualization/plot_all_grams.py
```
![Alt text](assets/fig_gram_matrices.png?raw=true "Gram Matrices")



# Plot Dataset Distribution

1. Run the script
```
 visualization/plot_data_distribution.py
```
![Alt text](assets/fig_dataset_distribution.png?raw=true "Data Distribution")

# Plot Dynamic Angular Margins

1. Run the script
```
 visualization/plot_dynamic_angular_margins.py
``` 
![Alt text](assets/fig_dynamic_angular_margins.png?raw=true "Dynamic Angular Margins")

# Plot Confusion Matrices

1. Run the script (ID Eval)
```
 visualization/plot_confusion_matrx.py
``` 
![Alt text](assets/fig_confusion_matrices.png?raw=true "Confusion Matrices")
2. Run the script (OOD Eval)
```
 visualization/plot_ood_confusion_matrx.py
``` 
![Alt text](assets/fig_ood_confusion_matrices.png?raw=true "OOD Confusion Matrices")
# Plot Threshold

1. Run the script
```
 visualization/plot_threshold.py
``` 
![Alt text](assets/fig_threshold_plot.png?raw=true "Threshold")
