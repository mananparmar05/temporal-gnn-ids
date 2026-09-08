"""
Flow Data Log Parser & Preprocessor
Parses raw network traffic logs (CIC-IDS2017, UNSW-NB15, or CSV streams)
into normalized connection flow records: [protocol, byte_count, packet_count, duration, tcp_flags].
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
        e_ij(t) = [ protocol, byte_count, packet_count, duration, tcp_flags ]
        """
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0.0

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
        Handles both full PCAP exports and MachineLearningCSV exports.
        """
        df = pd.read_csv(csv_path, encoding='latin1', low_memory=False)
        # Clean column names (strip spaces and BOM)
        df.columns = [c.strip().replace('ï»¿', '') for c in df.columns]

        column_mapping = {
            'Source IP': 'src_ip',
            'Source_IP': 'src_ip',
            'src_ip': 'src_ip',
            'Destination IP': 'dst_ip',
            'Destination_IP': 'dst_ip',
            'dst_ip': 'dst_ip',
            'Timestamp': 'timestamp',
            'Protocol': 'protocol',
            'Total Length of Fwd Packets': 'byte_count',
            'Total Fwd Packets': 'packet_count',
            'Flow Duration': 'duration',
            'FIN Flag Count': 'fin_flag',
            'SYN Flag Count': 'syn_flag',
            'RST Flag Count': 'rst_flag',
            'PSH Flag Count': 'psh_flag',
            'ACK Flag Count': 'ack_flag',
            'URG Flag Count': 'urg_flag',
            'Label': 'label'
        }

        df = df.rename(columns=column_mapping)

        # Fallback protocol
        if 'protocol' not in df.columns:
            df['protocol'] = 6.0  # Default to TCP
        else:
            df['protocol'] = pd.to_numeric(df['protocol'], errors='coerce').fillna(6.0)

        # Build aggregated TCP flags if individual flag counts exist
        flag_cols = [c for c in ['fin_flag', 'syn_flag', 'rst_flag', 'psh_flag', 'ack_flag', 'urg_flag'] if c in df.columns]
        if flag_cols:
            df['tcp_flags'] = df[flag_cols].sum(axis=1)
        elif 'tcp_flags' not in df.columns:
            df['tcp_flags'] = 0.0

        # Label encoding: 0 = BENIGN, 1 = ANOMALY
        if 'label' in df.columns:
            df['label'] = df['label'].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)
        else:
            df['label'] = 0

        # Create structured host IP topology if raw CSV lacks explicit IP columns
        if 'src_ip' not in df.columns:
            # Map based on Destination Port or index distribution
            dst_port = df['Destination Port'] if 'Destination Port' in df.columns else range(len(df))
            df['src_ip'] = [f"192.168.1.{(i % 25) + 1}" for i in range(len(df))]
            df['dst_ip'] = [f"10.0.0.{(int(p) % 10) + 1}" if str(p).isdigit() else "10.0.0.1" for p in dst_port]

        return df

    def parse_unswnb15_csv(self, csv_path):
        """
        Parses raw UNSW-NB15 CSV export into standardized flow DataFrame format.
        """
        df = pd.read_csv(csv_path, encoding='latin1', low_memory=False)
        df.columns = [c.strip().replace('ï»¿', '') for c in df.columns]

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

        if 'src_ip' not in df.columns:
            df['src_ip'] = [f"192.168.1.{(i % 25) + 1}" for i in range(len(df))]
        if 'dst_ip' not in df.columns:
            df['dst_ip'] = [f"10.0.0.{(i % 10) + 1}" for i in range(len(df))]
        if 'tcp_flags' not in df.columns:
            df['tcp_flags'] = 0.0
        if 'protocol' in df.columns:
            df['protocol'] = pd.to_numeric(df['protocol'], errors='coerce').fillna(6.0).astype(float)
        else:
            df['protocol'] = 6.0

        return df
