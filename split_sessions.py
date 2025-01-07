
import os
from scapy.all import *
from scapy.compat import raw
from pathlib import Path
from config import *


def pcapng_to_pcap(dataset_path):

    if not os.path.exists(os.path.join(dataset_path, 'Splitted')):
        os.mkdir(os.path.join(dataset_path, 'Splitted'))

    print("")
    print('STEP 01: STARTED (Converting PCAPNG files to PCAP)')
    print('================================================================================')

    files = list(Path(dataset_path).rglob('*.pcap*'))
    
    count = 0
    for file in files:

        # Remove corrupted files
        if file.name in ['vpn_hangouts_audio1.pcap', 'vpn_hangouts_audio2.pcap']:
            os.remove(file.__str__())
            continue

        old_file = file.resolve().__str__()
        new_file = old_file.replace("pcapng", "pcap")

        if old_file[-7:len(old_file)] == '.pcapng':
            command = "editcap -F libpcap -T ether " + old_file + " " + new_file
            os.system(command)
            os.remove(old_file)
            count = count + 1
            print("File converted : ", old_file)

    print("No. of PCAPNG files converted to PCAP  : ", count)
    print('================================================================================')
    print('STEP 01: COMPLETED')
    print('')



def split_sessions(dataset_path, splitcap_path):
    print("")
    print('STEP 02: STARTED (Splitting PCAP files into sessions)')
    print('================================================================================')

    splitcap_path = "\"" + splitcap_path + "\""

    split_dir = dataset_path + r'\Splitted'

    cmd1 = splitcap_path + " -r " +  dataset_path + " -o " + split_dir + " -recursive -s session"    
    cmd2 = "del /Q " + dataset_path + "\\*.pcap"
    cmd3 = "move /Y " + split_dir + "\\*.pcap " + dataset_path
    cmd4 = "rmdir /s /Q " + split_dir

    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)
    os.system(cmd4)

    print('================================================================================')
    print('STEP 02: COMPLETED')
    print('')




if __name__=='__main__':
    
    # Split sessions for ISCX-VPN dataset
    # pcapng_to_pcap(ISCX_VPN_DATASET_DIR)
    # split_sessions(ISCX_VPN_DATASET_DIR, SPLITCAP_PATH)
    # print('ALL DONE!!!!!')

    # Split sessions for VNAT dataset
    # pcapng_to_pcap(VNAT_DATASET_DIR)
    # split_sessions(VNAT_DATASET_DIR, SPLITCAP_PATH)
    # print('ALL DONE!!!!!')

    # Split sessions for Mendeley_NetworkTraffic dataset
    pcapng_to_pcap(MENDELEY_NETWORKTRAFIIC_DATASET_DIR)
    split_sessions(MENDELEY_NETWORKTRAFIIC_DATASET_DIR, SPLITCAP_PATH)
    print('ALL DONE!!!!!')