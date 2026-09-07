"""
Flow Data Log Parser & Preprocessor
Parses raw network traffic logs (CIC-IDS2017, UNSW-NB15, or CSV streams)
into normalized connection flow records: [protocol, byte_count, packet_count, duration, TCP_flags].
"""

import numpy as np
import pandas as pd

class FlowLogParser:
    def __init__(self):
        # Canonical feature schema required by Section 6.1
        self.feature_columns = ['protocol', 'byte_count', 'packet_count', 'duration', 'tcp_flags']
        
    def normalize_features(self, df):
        """
        Extracts and normalizes the core 5 flow features specified in Section 6.1:
        e_ij(t) = [ protocol, byte_count, packet_count, duration, TCP_flags ]
        """
        features_df = df[self.feature_columns].copy()
        
        # Log1p scaling for skewed numeric counts
        features_df['byte_count'] = np.log1p(np.maximum(0, features_df['byte_count'].values))
        features_df['packet_count'] = np.log1p(np.maximum(0, features_df['packet_count'].values))
        features_df['duration'] = np.log1p(np.maximum(0, features_df['duration'].values))
        
        # Min-max scale protocol and tcp_flags
        features_df['protocol'] = features_df['protocol'] / 17.0
        features_df['tcp_flags'] = features_df['tcp_flags'] / 32.0
        
        return features_df.values.astype(np.float32)

    def parse_cicids2017_csv(self, csv_path):
        """
        Parses raw CIC-IDS2017 CSV export into standardized flow DataFrame format.
        """
        df = pd.read_csv(csv_path)
        # Strip column whitespace common in CIC-IDS2017
        df.columns = [c.strip() for c in df.columns]
        
        column_mapping = {
            'Source IP': 'src_ip',
            'Destination IP': 'dst_ip',
            'Timestamp': 'timestamp',
            'Protocol': 'protocol',
            'Total Length of Fwd Packets': 'byte_count',
            'Total Fwd Packets': 'packet_count',
            'Flow Duration': 'duration',
            'FIN Flag Count': 'tcp_flags',
            'Label': 'label'
        }
        
        df = df.rename(columns=column_mapping)
        df['label'] = df['label'].apply(lambda x: 0 if str(x).upper() == 'BENIGN' else 1)
        return df

    def parse_unswnb15_csv(self, csv_path):
        """
        Parses raw UNSW-NB15 CSV export into standardized flow DataFrame format.
        """
        df = pd.read_csv(csv_path)
        column_mapping = {
            'srcip': 'src_ip',
            'dstip': 'dst_ip',
            'Stime': 'timestamp',
            'proto': 'protocol',
            'sbytes': 'byte_count',
            'spkts': 'packet_count',
            'dur': 'duration',
            'label': 'label'
        }
        df = df.rename(columns=column_mapping)
        if 'tcp_flags' not in df.columns:
            df['tcp_flags'] = 0.0
        return df
