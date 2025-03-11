from etherscan import Etherscan
from web3 import Web3
import pandas as pd
import numpy as np
from datetime import datetime
import time

# Etherscan API Key
ETHERSCAN_API_KEY = "848HZHG8QKCE4DIMBV7P13CDSUARHXDX4T"
eth = Etherscan(ETHERSCAN_API_KEY)

# Function to fetch transactions and generate attributes for ML model
def fetch_wallet_data(wallet_address):
    # Fetch normal transactions
    print("Fetching normal transactions...")
    normal_txs = eth.get_normal_txs_by_address(wallet_address, startblock=0, endblock=99999999, sort='asc')
    normal_df = pd.DataFrame(normal_txs)
    
    # Sometimes the API returns empty data if there are no transactions
    if not normal_df.empty:
        normal_df['eth_value'] = normal_df['value'].apply(lambda x: float(Web3.from_wei(int(x), 'ether')))
        normal_df['timestamp'] = normal_df['timeStamp'].astype(int)
        normal_df['datetime'] = normal_df['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x))
        normal_df['is_contract_creation'] = normal_df['to'].apply(lambda x: 1 if x == '' else 0)
        normal_df['is_contract_interaction'] = normal_df['input'].apply(lambda x: 1 if x != '0x' else 0)
    else:
        normal_df = pd.DataFrame(columns=['from', 'to', 'eth_value', 'timestamp', 'datetime', 'is_contract_creation', 'is_contract_interaction'])
    
    # Separate sent and received transactions
    sent_txns = normal_df[normal_df['from'].str.lower() == wallet_address.lower()] if not normal_df.empty else pd.DataFrame()
    received_txns = normal_df[normal_df['to'].str.lower() == wallet_address.lower()] if not normal_df.empty else pd.DataFrame()
    
    # To avoid rate limiting
    time.sleep(1)
    
    # Fetch ERC20 Token Transfer Events
    print("Fetching ERC20 transactions...")
    try:
        erc20_txs = eth.get_erc20_token_transfer_events_by_address(wallet_address, startblock=0, endblock=99999999, sort='asc')
        erc20_df = pd.DataFrame(erc20_txs)
    except:
        # Handle case when there are no ERC20 transactions
        erc20_df = pd.DataFrame(columns=['from', 'to', 'value', 'tokenName', 'tokenSymbol', 'timeStamp', 'contractAddress'])
    
    if not erc20_df.empty:
        # Process ERC20 transactions
        erc20_df['timestamp'] = erc20_df['timeStamp'].astype(int)
        erc20_df['datetime'] = erc20_df['timestamp'].apply(lambda x: datetime.utcfromtimestamp(x))
        # Convert token value based on decimals
        erc20_df['token_value'] = erc20_df.apply(
            lambda row: float(row['value']) / (10 ** int(row.get('tokenDecimal', 18))), axis=1
        )
        erc20_df['is_contract'] = erc20_df['to'].apply(lambda x: 1 if x.startswith('0x') and len(x) == 42 else 0)
    
    # Separate ERC20 sent and received transactions
    erc20_sent = erc20_df[erc20_df['from'].str.lower() == wallet_address.lower()] if not erc20_df.empty else pd.DataFrame()
    erc20_received = erc20_df[erc20_df['to'].str.lower() == wallet_address.lower()] if not erc20_df.empty else pd.DataFrame()
    
    # Contract interactions in normal transactions
    contract_txns = sent_txns[sent_txns['is_contract_interaction'] == 1] if not sent_txns.empty else pd.DataFrame()
    
    # Calculate features
    features = {
        "Index": 0,
        "Address": wallet_address,
        "FLAG": 0,
        "Avg min between sent tnx": sent_txns['timestamp'].diff().mean() / 60 if len(sent_txns) > 1 else 0,
        "Avg min between received tnx": received_txns['timestamp'].diff().mean() / 60 if len(received_txns) > 1 else 0,
        "Time Diff between first and last (Mins)": (normal_df['timestamp'].max() - normal_df['timestamp'].min()) / 60 if not normal_df.empty else 0,
        "Sent tnx": len(sent_txns),
        "Received Tnx": len(received_txns),
        "Number of Created Contracts": sent_txns['is_contract_creation'].sum() if not sent_txns.empty else 0,
        "Unique Received From Addresses": received_txns['from'].nunique() if not received_txns.empty else 0,
        "Unique Sent To Addresses": sent_txns['to'].nunique() if not sent_txns.empty else 0,
        
        # Value statistics for normal transactions
        "min value received": received_txns['eth_value'].min() if not received_txns.empty else 0,
        "max value received ": received_txns['eth_value'].max() if not received_txns.empty else 0,  # Note the space after "received "
        "avg val received": received_txns['eth_value'].mean() if not received_txns.empty else 0,
        "min val sent": sent_txns['eth_value'].min() if not sent_txns.empty else 0,
        "max val sent": sent_txns['eth_value'].max() if not sent_txns.empty else 0,
        "avg val sent": sent_txns['eth_value'].mean() if not sent_txns.empty else 0,
        
        # Contract interaction values
        "min value sent to contract": contract_txns['eth_value'].min() if not contract_txns.empty else 0,
        "max val sent to contract": contract_txns['eth_value'].max() if not contract_txns.empty else 0,
        "avg value sent to contract": contract_txns['eth_value'].mean() if not contract_txns.empty else 0,
        
        # Total transaction stats
        "total transactions (including tnx to create contract": len(normal_df),
        "total Ether sent": sent_txns['eth_value'].sum() if not sent_txns.empty else 0,
        "total ether received": received_txns['eth_value'].sum() if not received_txns.empty else 0,
        "total ether sent contracts": contract_txns['eth_value'].sum() if not contract_txns.empty else 0,
        "total ether balance": (received_txns['eth_value'].sum() if not received_txns.empty else 0) - 
                              (sent_txns['eth_value'].sum() if not sent_txns.empty else 0),
        
        # ERC20 transaction counts - note the space before these column names
        " Total ERC20 tnxs": len(erc20_df),
        " ERC20 total Ether received": erc20_received['token_value'].sum() if not erc20_received.empty else 0,
        " ERC20 total ether sent": erc20_sent['token_value'].sum() if not erc20_sent.empty else 0,
        " ERC20 total Ether sent contract": erc20_sent[erc20_sent['is_contract'] == 1]['token_value'].sum() if not erc20_sent.empty else 0,
        
        # ERC20 unique addresses - with spaces
        " ERC20 uniq sent addr": erc20_sent['to'].nunique() if not erc20_sent.empty else 0,
        " ERC20 uniq rec addr": erc20_received['from'].nunique() if not erc20_received.empty else 0,
        " ERC20 uniq sent addr.1": erc20_sent['to'].nunique() if not erc20_sent.empty else 0,
        " ERC20 uniq rec contract addr": erc20_received[erc20_received['is_contract'] == 1]['from'].nunique() if not erc20_received.empty else 0,
        
        # ERC20 time differences - with spaces
        " ERC20 avg time between sent tnx": erc20_sent['timestamp'].diff().mean() / 60 if len(erc20_sent) > 1 else 0,
        " ERC20 avg time between rec tnx": erc20_received['timestamp'].diff().mean() / 60 if len(erc20_received) > 1 else 0,
        " ERC20 avg time between rec 2 tnx": erc20_received['timestamp'].diff().mean() / 60 if len(erc20_received) > 1 else 0,
        " ERC20 avg time between contract tnx": erc20_sent[erc20_sent['is_contract'] == 1]['timestamp'].diff().mean() / 60 
                                             if len(erc20_sent[erc20_sent['is_contract'] == 1]) > 1 else 0,
        
        # ERC20 value statistics - with spaces
        " ERC20 min val rec": erc20_received['token_value'].min() if not erc20_received.empty else 0,
        " ERC20 max val rec": erc20_received['token_value'].max() if not erc20_received.empty else 0,
        " ERC20 avg val rec": erc20_received['token_value'].mean() if not erc20_received.empty else 0,
        " ERC20 min val sent": erc20_sent['token_value'].min() if not erc20_sent.empty else 0,
        " ERC20 max val sent": erc20_sent['token_value'].max() if not erc20_sent.empty else 0,
        " ERC20 avg val sent": erc20_sent['token_value'].mean() if not erc20_sent.empty else 0,
        
        # ERC20 contract value statistics - with spaces
        " ERC20 min val sent contract": erc20_sent[erc20_sent['is_contract'] == 1]['token_value'].min() 
                                     if not erc20_sent[erc20_sent['is_contract'] == 1].empty else 0,
        " ERC20 max val sent contract": erc20_sent[erc20_sent['is_contract'] == 1]['token_value'].max() 
                                     if not erc20_sent[erc20_sent['is_contract'] == 1].empty else 0,
        " ERC20 avg val sent contract": erc20_sent[erc20_sent['is_contract'] == 1]['token_value'].mean() 
                                     if not erc20_sent[erc20_sent['is_contract'] == 1].empty else 0,
        
        # ERC20 token types - with spaces
        " ERC20 uniq sent token name": erc20_sent['tokenName'].nunique() if not erc20_sent.empty else 0,
        " ERC20 uniq rec token name": erc20_received['tokenName'].nunique() if not erc20_received.empty else 0,
    }
    
    # Most frequent token types
    if not erc20_sent.empty and 'tokenName' in erc20_sent.columns and not erc20_sent['tokenName'].empty:
        features["ERC20 most sent token type"] = erc20_sent['tokenName'].value_counts().idxmax()
    else:
        features["ERC20 most sent token type"] = "None"
        
    if not erc20_received.empty and 'tokenName' in erc20_received.columns and not erc20_received['tokenName'].empty:
        features["ERC20_most_rec_token_type"] = erc20_received['tokenName'].value_counts().idxmax()
    else:
        features["ERC20_most_rec_token_type"] = "None"
    
    return features

def process_wallet_addresses(addresses):
    all_data = []
    for idx, address in enumerate(addresses):
        print(f"Processing wallet {idx+1}/{len(addresses)}: {address}")
        try:
            data = fetch_wallet_data(address)
            data["Index"] = idx + 1
            all_data.append(data)
            # Sleep to avoid API rate limits
            time.sleep(1)
        except Exception as e:
            print(f"Error processing {address}: {str(e)}")
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    return df

# Main execution
if __name__ == "__main__":
    # Option to process either a single wallet or multiple
    mode = input("Process a single wallet (1) or multiple wallets from a file (2)? Enter 1 or 2: ")
    
    if mode == "1":
        wallet_address = input("Enter Ethereum Wallet Address: ")
        df = process_wallet_addresses([wallet_address])
    else:
        file_path = input("Enter path to file containing wallet addresses (one per line): ")
        with open(file_path, 'r') as f:
            addresses = [line.strip() for line in f.readlines()]
        df = process_wallet_addresses(addresses)
    
    output_file = "wallet_features.csv"
    df.to_csv(output_file, index=False)
    print(f"Feature data saved to {output_file}")
