import threading
import pyshark
import torch
from torch_geometric.data import Data
from utils import num_packets_to_edge_indices, iscx_vpn_get_class_label, iscx_vpn_get_one_hot
from session_dataset import SessionDataset

class NetworkTrafficCapture:
    def __init__(self, interface, model, class_labels):
        self.interface = interface
        self.model = model
        self.class_labels = class_labels
        self.capture = pyshark.LiveCapture(interface=self.interface)
        self.sessions = {}
        self.lock = threading.Lock()

    def start_capture(self):
        capture_thread = threading.Thread(target=self._capture_traffic)
        capture_thread.start()

    def _capture_traffic(self):
        for packet in self.capture.sniff_continuously():
            self._process_packet(packet)

    def _process_packet(self, packet):
        session_id = packet.ip.src + '-' + packet.ip.dst
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = []
            self.sessions[session_id].append(packet)
            if len(self.sessions[session_id]) >= 10:  # Process session after 10 packets
                self._process_session(session_id)

    def _process_session(self, session_id):
        packets = self.sessions.pop(session_id, [])
        features = [self._extract_features(packet) for packet in packets]
        edge_indices = num_packets_to_edge_indices(len(packets))
        class_label = iscx_vpn_get_class_label(session_id)
        class_vector = iscx_vpn_get_one_hot(class_label)

        data = Data(x=torch.tensor(features, dtype=torch.float), edge_index=torch.tensor(edge_indices, dtype=torch.long), y=torch.tensor(class_vector, dtype=torch.long))
        self._classify_session(data)

    def _extract_features(self, packet):
        # Extract relevant features from the packet
        return [float(packet.length), float(packet.ip.ttl)]

    def _classify_session(self, data):
        self.model.eval()
        with torch.no_grad():
            output = self.model(data)
            predicted_class = output.argmax(dim=1)
            print(f"Session classified as: {self.class_labels[predicted_class]}")

# Load pretrained model
model = torch.load('pretrained_dynAAMPG_model.pth')
model.eval()

# Define class labels
class_labels = iscx_vpn_get_unique_labels()

# Start capturing network traffic
network_capture = NetworkTrafficCapture(interface='eth0', model=model, class_labels=class_labels)
network_capture.start_capture()