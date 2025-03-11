import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class EthereumFraudDetectorNN:
    def __init__(self, model_path='ethereum_fraud_model_nn.pth'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.features = None
        self.verified_data = []

    def preprocess_data(self, data):
        data = data.copy()
        unnamed_cols = [col for col in data.columns if col.startswith('Unnamed')]
        if unnamed_cols:
            data = data.drop(columns=unnamed_cols)
        data.columns = [col.strip() for col in data.columns]
        excluded_columns = ['Index', 'Address', 'FLAG', 'ERC20 most sent token type', 'ERC20_most_sent_token_type',
                            'ERC20_most_rec_token_type', 'ERC20 most received token type']
        data = data.dropna(how='all')
        data = data.fillna(0)
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_numeric(data[col])
                except:
                    if col not in excluded_columns:
                        print(f"Dropping non-numeric column: {col}")
                        data = data.drop(columns=[col])
        self.features = [col for col in data.columns if col not in excluded_columns]
        return data

    def train(self, data_path, test_size=0.2, epochs=50, batch_size=32):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        df = self.preprocess_data(df)
        X = df[self.features]
        y = df['FLAG']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        input_shape = X_train_scaled.shape[1]
        self.model = nn.Sequential(
            nn.Linear(input_shape, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        print("Training the model...")
        self.model.train()
        for epoch in range(epochs):
            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        self.model.eval()
        with torch.no_grad():
            y_pred_prob = self.model(X_test_tensor).numpy()
        y_pred = (y_pred_prob > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix_nn.png')
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        joblib.dump(self.scaler, 'scaler_nn.pkl')
        joblib.dump(self.features, 'features_nn.pkl')
        torch.save(self.model.state_dict(), self.model_path)
        print(f"Model saved to neural_model.pt at {self.model_path}")
        return accuracy, precision, recall, f1

    def load_model(self):
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.features = joblib.load('features_nn.pkl')  # Load features first
            input_shape = len(self.features)
            self.model = nn.Sequential(
                nn.Linear(input_shape, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            self.model.load_state_dict(torch.load(self.model_path))
            self.scaler = joblib.load('scaler_nn.pkl')
            return True
        else:
            print(f"Model file {self.model_path} not found.")
            return False

    def predict_risk(self, wallet_data, threshold=0.5):
        if self.model is None:
            loaded = self.load_model()
            if not loaded:
                raise Exception("No model loaded or trained.")
        if isinstance(wallet_data, dict):
            wallet_data = pd.DataFrame([wallet_data])
        wallet_data = wallet_data.copy()
        unnamed_cols = [col for col in wallet_data.columns if col.startswith('Unnamed')]
        if unnamed_cols:
            wallet_data = wallet_data.drop(columns=unnamed_cols)
        wallet_data.columns = [col.strip() for col in wallet_data.columns]
        address = wallet_data['Address'].iloc[0] if 'Address' in wallet_data.columns else "Unknown"
        for feature in self.features:
            feature_stripped = feature.strip()
            if feature not in wallet_data.columns:
                possible_matches = [col for col in wallet_data.columns if col.strip() == feature_stripped]
                if possible_matches:
                    wallet_data[feature] = wallet_data[possible_matches[0]]
                else:
                    print(f"Column {feature} not found in input data. Creating with default values.")
                    wallet_data[feature] = 0
        wallet_features = wallet_data[self.features].copy()
        wallet_features_scaled = self.scaler.transform(wallet_features)
        wallet_features_tensor = torch.tensor(wallet_features_scaled, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            risk_prob = self.model(wallet_features_tensor).numpy()[0, 0]
        risk_label = 1 if risk_prob >= threshold else 0
        result = {
            'address': address,
            'risk_score': risk_prob,
            'risk_label': risk_label,
            'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'feature_contributions': {}
        }
        return result

    def add_verified_data(self, wallet_data, actual_label):
        if isinstance(wallet_data, dict):
            wallet_data = pd.DataFrame([wallet_data])
        wallet_data['FLAG'] = actual_label
        self.verified_data.append(wallet_data)
        verified_df = pd.concat(self.verified_data) if len(self.verified_data) > 1 else self.verified_data[0]
        verified_df.to_csv('verified_wallet_data.csv', index=False)
        print(f"Added verified data for wallet {wallet_data['Address'].iloc[0]}. " 
              f"Total verified samples: {len(self.verified_data)}")
        return len(self.verified_data)

    def retrain_with_verified_data(self, original_data_path):
        if len(self.verified_data) == 0:
            print("No verified data available for retraining.")
            return False
        original_df = pd.read_csv(original_data_path)
        verified_df = pd.concat(self.verified_data)
        combined_df = pd.concat([original_df, verified_df])
        combined_df.drop_duplicates(subset=['Address'], keep='last', inplace=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = f"combined_training_data_{timestamp}.csv"
        combined_df.to_csv(combined_path, index=False)
        print(f"Retraining model with {len(combined_df)} samples...")
        self.train(combined_path)
        return True

    def generate_report(self, wallet_data, prediction_result):
        address = wallet_data['Address'].iloc[0] if 'Address' in wallet_data.columns else "Unknown"
        print("\n" + "="*50)
        print(f"RISK ASSESSMENT REPORT FOR {address}")
        print("="*50)
        print(f"Risk Score: {prediction_result['risk_score']:.4f}")
        print(f"Risk Label: {'HIGH RISK' if prediction_result['risk_label'] == 1 else 'LOW RISK'}")
        print(f"Generated on: {prediction_result['prediction_time']}")
        print("\nTop Risk Factors:")
        sorted_contributions = sorted(
            prediction_result['feature_contributions'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        for feature, contribution in sorted_contributions[:5]:
            print(f"- {feature}: {contribution:.4f}")
        def get_column_value(column_names):
            for col in column_names:
                if col in wallet_data.columns:
                    return wallet_data[col].iloc[0]
            return "N/A"
        print("\nTransaction Summary:")
        print(f"- Total Transactions: {get_column_value(['total transactions (including tnx to create contract'])}")
        print(f"- Sent Transactions: {get_column_value(['Sent tnx'])}")
        print(f"- Received Transactions: {get_column_value(['Received Tnx'])}")
        print(f"- Total Ether Sent: {get_column_value(['total Ether sent'])}")
        print(f"- Total Ether Received: {get_column_value(['total ether received'])}")
        print("\nWallet Activity Pattern:")
        print(f"- Time Span: {get_column_value(['Time Diff between first and last (Mins)'])/60/24:.2f} days")
        print(f"- Avg. Time Between Sent Txns: {get_column_value(['Avg min between sent tnx']):.2f} minutes")
        print(f"- Avg. Time Between Received Txns: {get_column_value(['Avg min between received tnx']):.2f} minutes")
        print("\nERC20 Token Activity:")
        print(f"- Total ERC20 Transactions: {get_column_value([' Total ERC20 tnxs', 'Total ERC20 tnxs'])}")
        print(f"- Most Sent Token Type: {get_column_value(['ERC20_most_sent_token_type', 'ERC20 most sent token type'])}")
        print(f"- Most Received Token Type: {get_column_value(['ERC20_most_rec_token_type'])}")
        print("="*50)
        print("NOTE: This report is for informational purposes only.")
        print("Further investigation may be required for conclusive evidence.")
        print("="*50 + "\n")
        return True

    def check_illegal_transactions(self, wallet_data):
        address = wallet_data['Address'].iloc[0] if 'Address' in wallet_data.columns else "Unknown"
        suspicious_indicators = []
        if wallet_data['max value received'].iloc[0] > 1000:
            suspicious_indicators.append(f"Unusually large incoming transaction: {wallet_data['max value received'].iloc[0]} ETH")
        if wallet_data['max val sent'].iloc[0] > 1000:
            suspicious_indicators.append(f"Unusually large outgoing transaction: {wallet_data['max val sent'].iloc[0]} ETH")
        if wallet_data['Sent tnx'].iloc[0] > 0:
            avg_time_between_sent = wallet_data['Avg min between sent tnx'].iloc[0]
            if avg_time_between_sent < 5 and wallet_data['Sent tnx'].iloc[0] > 50:
                suspicious_indicators.append(f"High frequency outgoing transactions: {wallet_data['Sent tnx'].iloc[0]} transactions with average {avg_time_between_sent:.2f} minutes between them")
        if wallet_data['Sent tnx'].iloc[0] > 0 and wallet_data['Received Tnx'].iloc[0] > 0:
            ratio = wallet_data['Received Tnx'].iloc[0] / wallet_data['Sent tnx'].iloc[0]
            if ratio > 5:
                suspicious_indicators.append(f"Unusual received/sent ratio: {ratio:.2f}")
        if wallet_data['Unique Sent To Addresses'].iloc[0] > 100:
            suspicious_indicators.append(f"Unusually high number of unique addresses sent to: {wallet_data['Unique Sent To Addresses'].iloc[0]}")
        result = {
            'address': address,
            'suspicious_indicators': suspicious_indicators,
            'has_suspicious_activity': len(suspicious_indicators) > 0
        }
        return result

    def track_ip_address(self, wallet_address):
        print(f"IP tracking functionality would be activated for {wallet_address}")
        return "IP tracking activated (placeholder)"

    def alert_authorities(self, wallet_address, risk_score, illegal_transactions_report):
        print(f"ALERT: Suspicious wallet detected: {wallet_address}")
        print(f"Risk Score: {risk_score}")
        print("Suspicious Indicators:")
        for indicator in illegal_transactions_report['suspicious_indicators']:
            print(f"- {indicator}")
        return "Alert sent to authorities (placeholder)"

    def log_activity(self, wallet_address, risk_score, prediction_time):
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'wallet_address': wallet_address,
            'risk_score': risk_score,
            'prediction_time': prediction_time
        }
        log_df = pd.DataFrame([log_entry])
        log_file = 'wallet_analysis_log.csv'
        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)
        print(f"Activity logged for wallet {wallet_address}")
        return True

# Example usage
if __name__ == "__main__":
    detector = EthereumFraudDetectorNN()
    train_model = input("Do you want to train a new model? (y/n): ").lower() == 'y'
    if train_model:
        training_data = input("Enter path to training data (default: trained_dataset.csv): ") or "trained_dataset.csv"
        detector.train(training_data)
    else:
        if not detector.load_model():
            print("No existing model found. Training a new model...")
            training_data = input("Enter path to training data (default: trained_dataset.csv): ") or "trained_dataset.csv"
            detector.train(training_data)
    while True:
        print("\n" + "="*50)
        print("ETHEREUM WALLET RISK ASSESSMENT")
        print("="*50)
        print("1. Analyze a wallet address")
        print("2. Retrain model with verified data")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")
        if choice == '1':
            wallet_file = input("Enter path to wallet data CSV: ")
            try:
                wallet_data = pd.read_csv(wallet_file)
                for i, row in wallet_data.iterrows():
                    address = row['Address']
                    print(f"\nAnalyzing wallet: {address}")
                    single_wallet = pd.DataFrame([row])
                    risk_result = detector.predict_risk(single_wallet)
                    print(f"Risk Score: {risk_result['risk_score']:.4f}")
                    print(f"Risk Label: {'HIGH RISK' if risk_result['risk_label'] == 1 else 'LOW RISK'}")
                    if risk_result['risk_label'] == 1:
                        illegal_transactions_report = detector.check_illegal_transactions(single_wallet)
                        if illegal_transactions_report['has_suspicious_activity']:
                            detector.alert_authorities(address, risk_result['risk_score'], illegal_transactions_report)
                    detector.log_activity(address, risk_result['risk_score'], risk_result['prediction_time'])
            except Exception as e:
                print(f"Error analyzing wallet: {str(e)}")
        elif choice == '2':
            original_data_path = input("Enter path to original training data (default: trained_dataset.csv): ") or "trained_dataset.csv"
            detector.retrain_with_verified_data(original_data_path)
        elif choice == '3':
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")