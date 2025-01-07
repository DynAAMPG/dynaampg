import numpy as np
import os
import json
from scapy.all import *
from scapy.compat import raw
from scapy.layers.inet import IP, UDP
from scapy.layers.inet6 import IPv6
from pathlib import Path
from tqdm import tqdm
from scapy.layers.l2 import Ether
from utils import *
from config import *

def myconverter(o):
    if isinstance(o, np.float32):
        return float(o)
    
def preprocess(dataset_path, packets_per_session, dataset):

    # Get list of all PCAP session file paths
    files = list(Path(dataset_path).rglob('*.pcap'))
    pbar = tqdm(total=len(files), desc='Files Done: ')

    for file_number, file in enumerate(files, start=1):

        pcap_reader = PcapReader(file.__str__())
        # Read first n packets of PCAP file, n=packets_per_session, we use n=10
        packet_list = pcap_reader.read_all(count=packets_per_session)

        features = []
        max = 1500
        json_str = {}

        # Session atleast has 10 packets
        if len(packet_list) >= packets_per_session:
            for packet in packet_list[0:packets_per_session]:

                # Remove ethernet header
                #########################################################
                if Ether in packet:
                    packet = packet[Ether].payload


                # Mask IP
                #########################################################
                if IPv6 in packet:
                    packet[IPv6].dst = '0000::0:0'
                    packet[IPv6].src = '0000::0000:0000:0000:0000'

                if IP in packet:
                    packet[IP].dst = '0.0.0.0'
                    packet[IP].src = '0.0.0.0'


                # Pad UDP
                #########################################################
                if UDP in packet:
                    layer_after = packet[UDP].payload.copy()

                    pad = Padding()
                    pad.load = '\x00' * 12

                    layer_before = packet.copy()
                    layer_before[UDP].remove_payload()
                    packet = layer_before / raw(pad) / layer_after


                # Convert PCAP to Byte format and normalize
                #########################################################
                t = np.frombuffer(raw(packet), dtype=np.uint8)[0: max] / 255
                if len(t) <= max:
                    pad_width = max - len(t)
                    t = np.pad(t, pad_width=(0, pad_width), constant_values=0)
                    features.append(t.tolist())

            # Save PCAP as JSON files
            #########################################################
            
            class_label = ''
            one_hot_vector = []

            if dataset == DATASET.ISCX_VPN:
                class_label = iscx_vpn_get_class_label(Path(file).name)
                one_hot_vector = iscx_vpn_get_one_hot(class_label)
            elif dataset == DATASET.VNAT:
                class_label = vnat_get_class_label(Path(file).name)
                one_hot_vector = vnat_get_one_hot(class_label)
            elif dataset == DATASET.ISCX_TOR:
                class_label = iscx_tor_get_class_label(Path(file).name)
                one_hot_vector = iscx_tor_get_one_hot(class_label)
            elif dataset == DATASET.NETWORK_TRAFFIC:
                class_label = network_traffic_get_class_label(Path(file).name)
                one_hot_vector = network_traffic_get_one_hot(class_label)
            elif dataset == DATASET.REALTIME:
                class_label = realtime_get_class_label(Path(file).name)
                one_hot_vector = realtime_get_one_hot(class_label)
            
            
            id = str(file_number)
            json_str['id'] = id
            json_str['features'] = features
            json_str['edge_indices'] = num_packets_to_edge_indices(len(features))
            json_str['class'] = class_label
            json_str['class_vector'] = one_hot_vector  

            with open(dataset_path + '\\' + id + '.json', 'w') as f:
                f.write(json.dumps(json_str, default=myconverter))

            features.clear()

        pcap_reader.close()

        # Remove processed PCAP session file
        if os.path.isfile(file.__str__()):
            os.remove(file.__str__())

        # Update progress bar
        pbar.update(1)
        


if __name__=='__main__':
    packets_per_session = 5

    # Process ISCX-VPN dataset
    preprocess(ISCX_VPN_DATASET_DIR, packets_per_session, DATASET.ISCX_VPN)

    # Process VNAT dataset
    preprocess(VNAT_DATASET_DIR, packets_per_session, DATASET.VNAT)

    # Process ISCX-Tor dataset
    # preprocess(ISCX_TOR_DATASET_DIR, packets_per_session, DATASET.ISCX_TOR)
    
    # Process NetworkTraffic dataset
    # preprocess(NETWORK_TRAFFIC_DATASET_DIR, packets_per_session, DATASET.NETWORK_TRAFFIC)

    # Process Realtime dataset
    # preprocess(REALTIME_DATASET_DIR, packets_per_session, DATASET.REALTIME)
    




    print('ALL DONE!!!!!')