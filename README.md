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

## 3. Download and prepare the datasets
### 3A. (ISCX-VPN-NonVPN-2016)
* Download the following files from the link:
http://205.174.165.80/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/PCAPs/

```
1. NonVPN-PCAPs-01.zip
2. NonVPN-PCAPs-02.zip
3. NonVPN-PCAPs-03.zip
4. VPN-PCAPS-01.zip
5. VPN-PCAPS-02.zip
```
* Extract these following files in folder datasets/ISCX_VPN
* Finally the dataset structure should look like:
```
-datasets
 - ISCX_VPN
  - aim_chat_3a.pcap
  - aim_chat_3b.pcap
	...
  - youtubeHTML5_1.pcap
  - vpn_aim_chat1a.pcap
  - vpn_aim_chat1b.pcap
	...
  - vpn_youtube_A.pcap
```

### 3B. (VNAT)
* Download the following files from the link:
https://archive.ll.mit.edu/datasets/vnat/VNAT_release_1.zip

```
1. VNAT_release_1.zip
```
* Extract these files in folder datasets/VNAT

### 3C. (Mendeley NetworkTraffic)
* Download the following files from the link:
https://data.mendeley.com/datasets/5pmnkshffm/3

```
1. pcap_bulk-1.tar.gz
2. pcap_bulk-2.tar.gz
3. pcap_idle.tar.gz
4. pcap_interactive.tar.gz
5. pcap_video.tar.gz
6. pcap_web.tar.gz
```
* Extract these files in folder datasets/Mendeley

### 3D. (US Mobile APP)
* Download the following files from the link:
https://recon.meddle.mobi/appversions/details.html

```
1. china.zip
1. us.zip
1. india.zip
```
* Extract these files in folder datasets/USMobileApp

### 3E. (Custom/Realtime Dataset)

1. Download the Custom/Realtime dataset from following link:
```
https://drive.google.com/file/d/12888hICcowDk2Ye1rfnEHQHictyh7TB9/view?usp=sharing
```
2. Extract and place the dataset in datasets/Realtime


### 3F. (OOD Dataset)

1. Download the OOD dataset from following link:
```
https://drive.google.com/file/d/1OrhHZ-BHBVxkXLAQnukRI6coB_SQuxRW/view?usp=sharing
```
2. Extract and place the dataset in datasets/OOD




## 4. Create conda environment and install packages
* Open miniconda and run following commands one-by-one:
```
create conda -n dynaampg python=3.11
conda activate dynaampg
pip install matplotlib scapy tqdm pandas numpy pyshark networkx
pip install psutil scikit-learn seaborn umap-learn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu118.html
pip install tensorboard
```

## 5. Convert pcapng files to pcap and split sessions
* Download splitcap utility
   https://www.netresec.com/?page=SplitCap
* Place SplitCap.exe file in .\dynaampg\
* Open project folder in VSCode
* Select python interpreter as "dynaampg"
* Run the script split_sessions.py

## 6. Process packets

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

## 7. Generate graphs
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

## 8. Train the model
* Run the script train.py

![Alt text](assets/training.png?raw=true "Training model")

## 9. Launch tensorboard
* Open new powershell window from VS Code by clicking "+" button
* Run the following command
```
tensorboard --logdir=runs
```

Now CTRL+Click http://localhost:6006/ or copy this address in browser
```
![Alt text](assets/tensorboard.png?raw=true "Launch Tensorboard")
![Alt text](assets/graph3.png?raw=true "Tensorboard Training and Validation Accuracy Visualization")
![Alt text](assets/graph4.png?raw=true "Tensorboard Training and Validation Loss Visualization")
```


## Plot All Gram Matrices

1. Download the pre-trained model from following link:
```
https://drive.google.com/file/d/10Gcpi1Kln-yCCMk-Sahl-f0i9dI_iHzm/view?usp=sharing
```
2. Extract and place the model in saved_models/

3. Run the script
```
 visualization/plot_all_grams.py
```
![Alt text](assets/fig_gram_matrices.png?raw=true "Gram Matrices")



## Plot Dataset Distribution

1. Run the script
```
 visualization/plot_data_distribution.py
```
![Alt text](assets/fig_dataset_distribution.png?raw=true "Data Distribution")

## Plot Dynamic Angular Margins

1. Run the script
```
 visualization/plot_dynamic_angular_margins.py
``` 
![Alt text](assets/fig_dynamic_angular_margins.png?raw=true "Dynamic Angular Margins")

## Plot Confusion Matrices

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
## Plot Threshold

1. Run the script
```
 visualization/plot_threshold.py
``` 
![Alt text](assets/fig_threshold_plot.png?raw=true "Threshold")

# Ablation Study

To assess the contribution of each component in the DynAAM-PG framework, we conducted an ablation study on the VNAT dataset, chosen for its significant class imbalance. The study investigates the impact of: (i) dynamic angular margins, (ii) sub-centers, and (iii) penalized Gram matrices on classification accuracy and OOD detection.

![Alt text](assets/ablation-table.png?raw=true "Ablation Results")

## Baseline Model. 
We use a baseline model with the same session encoder but without angular margin adjustments or penalized Gram matrices. This model employs a standard \textit{Softmax} loss, enabling us to clearly quantify improvements introduced by DynAAM-PG.

### Effect of Dynamic Angular Margins. 
Incorporating Dynamic Additive Angular Margins (DynAAM) significantly boosts classification performance over the baseline, as shown in {Ablation Table}. DynAAM addresses class imbalance by dynamically adjusting angular margins, particularly for minority classes, improving class separation and overall classification accuracy.

### Effect of Sub-centers. 
The inclusion of sub-centers helps address intra-class variability. Testing various numbers of sub-centers ($C=2$ to $C=15$) revealed that $C=3$ provides optimal results, with an $F1_m$ of 0.9816 and an $ACC$ of 0.9814. Increasing $C$ beyond 3 reduces performance, as seen with $C=15$ ($F1_m$ = 0.8563, $ACC$ = 0.8495). This suggests that a moderate number of sub-centers balances intra-class compactness without introducing noise.

### Effect of Penalized Gram Matrices. 
We evaluated three types of Gram matrices for OOD detection: (i) standard Gram matrix $G$, (ii) higher-order Gram matrices $G^{p}$, and (iii) our proposed normalized penalized Gram matrix $G^{np}$. As shown in Table \ref{tableAblation}, $G^{np}$ with $C=3$ achieves the best OOD detection performance, with an $F1_m$ of 0.9501 and $ACC$ of 0.9498. Standard and higher-order Gram matrices ($G$ and $G^p$) show lower performance, particularly as the order increases. The effectiveness of $G^{np}$ highlights the importance of penalization and normalization for better inter-feature correlation and OOD detection.

The ablation study confirms the importance of each DynAAM-PG component. Dynamic angular margins enhance class separation, sub-centers optimize intra-class compactness, and the normalized penalized Gram matrix is crucial for reliable OOD detection. Together, these components contribute to improved classification accuracy and robust OOD detection, demonstrating the efficacy of the DynAAM-PG framework.

