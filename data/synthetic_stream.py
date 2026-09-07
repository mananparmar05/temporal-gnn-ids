"""
Dynamic Network Traffic Stream Generator
Simulates multi-host network traffic streams with realistic benign background traffic 
and injected attack patterns (Port Scans, DDoS, Botnet C2, Lateral Movement).
"""

import time
import random
import numpy as np
import pandas as pd

class TrafficStreamGenerator:
    def __init__(self, num_hosts=50, seed=42):
        np.random.seed(seed)
        random.seed(seed)
        self.num_hosts = num_hosts
        self.hosts = [f"192.168.1.{i}" for i in range(1, num_hosts + 1)]
        self.protocols = [6, 17, 1] # TCP, UDP, ICMP
        
        # Categorize hosts into roles (proportional to num_hosts)
        n_servers = max(2, num_hosts // 10)
        n_attackers = max(3, num_hosts // 8)
        self.servers = self.hosts[:n_servers]
        self.attackers = self.hosts[n_servers:n_servers + n_attackers]
        self.workstations = self.hosts[n_servers + n_attackers:]
        
    def generate_flow_stream(self, num_windows=20, window_duration_sec=30, attack_ratio=0.35):
        """
        Generates a sequence of connection flow records across time windows.
        Returns a list of window flow dictionaries containing flow DataFrames & ground truth labels.
        """
        window_streams = []
        base_timestamp = 1700000000.0
        
        for w_idx in range(num_windows):
            window_start = base_timestamp + w_idx * window_duration_sec
            flows = []
            
            # 1. Benign background traffic (Workstations talking to Servers, peer-to-peer)
            num_benign = np.random.randint(150, 300)
            for _ in range(num_benign):
                src = random.choice(self.workstations)
                dst = random.choice(self.servers + self.workstations)
                if src == dst:
                    continue
                
                protocol = random.choice(self.protocols)
                byte_count = float(np.random.exponential(scale=1500) + 64)
                packet_count = float(np.random.poisson(lam=10) + 1)
                duration = float(np.random.exponential(scale=2.5) + 0.01)
                tcp_flags = float(random.choice([2, 16, 24, 18])) # SYN, ACK, PSH-ACK, SYN-ACK
                
                ts = window_start + np.random.uniform(0, window_duration_sec)
                
                flows.append({
                    'timestamp': ts,
                    'src_ip': src,
                    'dst_ip': dst,
                    'protocol': protocol,
                    'byte_count': byte_count,
                    'packet_count': packet_count,
                    'duration': duration,
                    'tcp_flags': tcp_flags,
                    'label': 0, # Benign
                    'attack_type': 'BENIGN'
                })
            
            # 2. Inject Attack Topologies (Port Scan, DDoS, Botnet C2, Lateral Movement)
            is_attack_window = (w_idx > 2) and (random.random() < attack_ratio)
            
            if is_attack_window:
                attack_category = random.choice(['PORT_SCAN', 'DDOS', 'BOTNET_C2', 'LATERAL_MOVEMENT'])
                attacker = random.choice(self.attackers)
                
                if attack_category == 'PORT_SCAN':
                    # One host connects rapidly to dozens of target hosts (High Out-Degree structural signature)
                    targets = random.sample(self.workstations + self.servers, k=min(25, len(self.hosts)-1))
                    for tgt in targets:
                        ts = window_start + np.random.uniform(0, window_duration_sec)
                        flows.append({
                            'timestamp': ts,
                            'src_ip': attacker,
                            'dst_ip': tgt,
                            'protocol': 6.0, # TCP SYN
                            'byte_count': 64.0,
                            'packet_count': 1.0,
                            'duration': 0.001,
                            'tcp_flags': 2.0, # SYN
                            'label': 1,
                            'attack_type': 'PORT_SCAN'
                        })
                        
                elif attack_category == 'DDOS':
                    # Many compromised workstations burst traffic to a single server (High In-Degree)
                    victim = random.choice(self.servers)
                    bots = random.sample(self.workstations + self.attackers, k=20)
                    for bot in bots:
                        for _ in range(5):
                            ts = window_start + np.random.uniform(0, window_duration_sec)
                            flows.append({
                                'timestamp': ts,
                                'src_ip': bot,
                                'dst_ip': victim,
                                'protocol': 17.0, # UDP flood
                                'byte_count': 1400.0,
                                'packet_count': 50.0,
                                'duration': 0.1,
                                'tcp_flags': 0.0,
                                'label': 1,
                                'attack_type': 'DDOS'
                            })
                            
                elif attack_category == 'BOTNET_C2':
                    # Low-volume persistent periodic beaconing from multiple bots to C2 master
                    c2_master = attacker
                    bots = self.attackers[:3]
                    for bot in bots:
                        for b_i in range(8):
                            ts = window_start + b_i * 3.5
                            flows.append({
                                'timestamp': ts,
                                'src_ip': bot,
                                'dst_ip': c2_master,
                                'protocol': 6.0,
                                'byte_count': 256.0,
                                'packet_count': 4.0,
                                'duration': 0.05,
                                'tcp_flags': 24.0,
                                'label': 1,
                                'attack_type': 'BOTNET_C2'
                            })
                            
                elif attack_category == 'LATERAL_MOVEMENT':
                    # Step-by-step infection chain: Attacker -> Host A -> Host B -> Server
                    chain = [attacker] + random.sample(self.workstations, k=3) + [self.servers[0]]
                    for i in range(len(chain)-1):
                        ts = window_start + i * 5.0
                        flows.append({
                            'timestamp': ts,
                            'src_ip': chain[i],
                            'dst_ip': chain[i+1],
                            'protocol': 6.0,
                            'byte_count': 5000.0,
                            'packet_count': 30.0,
                            'duration': 1.2,
                            'tcp_flags': 18.0,
                            'label': 1,
                            'attack_type': 'LATERAL_MOVEMENT'
                        })

            # Convert window to DataFrame
            df = pd.DataFrame(flows)
            df = df.sort_values(by='timestamp').reset_index(drop=True)
            window_streams.append({
                'window_idx': w_idx,
                'window_start': window_start,
                'window_end': window_start + window_duration_sec,
                'df': df
            })
            
        return window_streams, self.hosts

if __name__ == '__main__':
    gen = TrafficStreamGenerator(num_hosts=30)
    windows, hosts = gen.generate_flow_stream(num_windows=5)
    print(f"Generated {len(windows)} windows over {len(hosts)} hosts.")
    print("Sample flow window 0 shape:", windows[0]['df'].shape)
    print("Sample flow head:\n", windows[0]['df'].head())
