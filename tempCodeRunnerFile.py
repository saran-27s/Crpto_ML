import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

class EthereumFraudDetector:
    def __init__(self, model_path='ethereum_fraud_model.pkl'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.features = None
        self.verified_data = []  # Store new verified data for retraining
        
    def preprocess_data(self, data):
        """Preprocess the data by handling missing values and scaling features"""
        # Remove any rows with all NaN values
        data = data.dropna(how='all')
        
        # Fill remaining NaN values with appropriate values
        data = data.fillna(0)
        
        # Convert object columns to numeric where possible
        for col in data.columns:
            if data[col].dtype == 'object':
                try:
                    data[col] = pd.to_numeric(data[col])
                except:
                    # For columns that can't be converted to numeric, we'll drop them
                    if col not in ['Address', 'ERC20_most_sent_token_type', 'ERC20_most_rec_token_type']:
                        print(f"Dropping non-numeric column: {col}")
                        data = data.drop(columns=[col])
        
        # Save column names for future use
        self.features = [col for col in data.columns if col not in ['Index', 'Address', 'FLAG', 
                                                             'ERC20_most_sent_token_type', 
                                                             'ERC20_most_rec_token_type']]
        
        return data
    
    def train(self, data_path, test_size=0.2, optimize=True):
        """Train the Random Forest model on the dataset"""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Preprocess the data
        df = self.preprocess_data(df)
        
        # Split features and target
        X = df[self.features]
        y = df['FLAG']
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
        
        # Scale the features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize the model
        if optimize:
            print("Optimizing hyperparameters...")
            # Define the parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, 'balanced']
            }
            
            # Use GridSearchCV to find the best parameters
            rf = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                                      cv=5, n_jobs=-1, verbose=1, scoring='f1')
            grid_search.fit(X_train_scaled, y_train)
            
            # Get the best parameters
            best_params = grid_search.best_params_
            print(f"Best parameters: {best_params}")
            
            # Initialize model with best parameters
            self.model = RandomForestClassifier(**best_params, random_state=42)
        else:
            # Use default parameters
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        print("Training the model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on test set
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Display confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': self.features,
            'Importance': self.model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        # Save the model
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, 'scaler.pkl')
        joblib.dump(self.features, 'features.pkl')
        
        print(f"Model saved to {self.model_path}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Top 20 Feature Importance')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        
        return accuracy, precision, recall, f1
    
    def load_model(self):
        """Load a pre-trained model"""
        if os.path.exists(self.model_path):
            print(f"Loading model from {self.model_path}")
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load('scaler.pkl')
            self.features = joblib.load('features.pkl')
            return True
        else:
            print(f"Model file {self.model_path} not found.")
            return False
    
    def predict_risk(self, wallet_data, threshold=0.5):
        """Predict risk score for a wallet address"""
        # Ensure model is loaded
        if self.model is None:
            loaded = self.load_model()
            if not loaded:
                raise Exception("No model loaded or trained.")
        
        # Convert to DataFrame if it's a dictionary
        if isinstance(wallet_data, dict):
            wallet_data = pd.DataFrame([wallet_data])
        
        # Extract address for reporting
        address = wallet_data['Address'].iloc[0] if 'Address' in wallet_data.columns else "Unknown"
        
        # Keep only the features used during training
        wallet_features = wallet_data[self.features].copy()
        
        # Scale the features
        wallet_features_scaled = self.scaler.transform(wallet_features)
        
        # Get prediction probability
        risk_prob = self.model.predict_proba(wallet_features_scaled)[0, 1]
        
        # Make binary prediction
        risk_label = 1 if risk_prob >= threshold else 0
        
        result = {
            'address': address,
            'risk_score': risk_prob,
            'risk_label': risk_label,
            'prediction_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'feature_contributions': {}
        }
        
        # Get feature contributions to the risk score
        for i, feature in enumerate(self.features):
            # Get the importances from all trees for this feature
            importances = [tree.feature_importances_[i] for tree in self.model.estimators_]
            avg_importance = np.mean(importances)
            result['feature_contributions'][feature] = avg_importance * wallet_features.iloc[0, i]
        
        return result
    
    def add_verified_data(self, wallet_data, actual_label):
        """Add verified wallet data for future retraining"""
        if isinstance(wallet_data, dict):
            wallet_data = pd.DataFrame([wallet_data])
        
        # Add actual label
        wallet_data['FLAG'] = actual_label
        
        # Add to verified data
        self.verified_data.append(wallet_data)
        
        # Save verified data to CSV
        verified_df = pd.concat(self.verified_data) if len(self.verified_data) > 1 else self.verified_data[0]
        verified_df.to_csv('verified_wallet_data.csv', index=False)
        
        print(f"Added verified data for wallet {wallet_data['Address'].iloc[0]}. " 
              f"Total verified samples: {len(self.verified_data)}")
        
        return len(self.verified_data)
    
    def retrain_with_verified_data(self, original_data_path):
        """Retrain the model with original data plus verified data"""
        if len(self.verified_data) == 0:
            print("No verified data available for retraining.")
            return False
        
        # Load original training data
        original_df = pd.read_csv(original_data_path)
        
        # Combine with verified data
        verified_df = pd.concat(self.verified_data)
        combined_df = pd.concat([original_df, verified_df])
        
        # Remove duplicates
        combined_df.drop_duplicates(subset=['Address'], keep='last', inplace=True)
        
        # Save combined dataset
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined_path = f"combined_training_data_{timestamp}.csv"
        combined_df.to_csv(combined_path, index=False)
        
        # Retrain model
        print(f"Retraining model with {len(combined_df)} samples...")
        self.train(combined_path)
        
        return True
    
    def generate_report(self, wallet_data, prediction_result):
        """Generate a PDF report for the wallet"""
        # This function would generate a PDF report
        # For this example, we'll just print what would be in the report
        
        address = wallet_data['Address'].iloc[0] if 'Address' in wallet_data.columns else "Unknown"
        
        print("\n" + "="*50)
        print(f"RISK ASSESSMENT REPORT FOR {address}")
        print("="*50)
        print(f"Risk Score: {prediction_result['risk_score']:.4f}")
        print(f"Risk Label: {'HIGH RISK' if prediction_result['risk_label'] == 1 else 'LOW RISK'}")
        print(f"Generated on: {prediction_result['prediction_time']}")
        print("\nTop Risk Factors:")
        
        # Sort feature contributions
        sorted_contributions = sorted(
            prediction_result['feature_contributions'].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        
        for feature, contribution in sorted_contributions[:5]:
            print(f"- {feature}: {contribution:.4f}")
        
        print("\nTransaction Summary:")
        print(f"- Total Transactions: {wallet_data['total transactions (including tnx to create contract'].iloc[0]}")
        print(f"- Sent Transactions: {wallet_data['Sent tnx'].iloc[0]}")
        print(f"- Received Transactions: {wallet_data['Received Tnx'].iloc[0]}")
        print(f"- Total Ether Sent: {wallet_data['total Ether sent'].iloc[0]}")
        print(f"- Total Ether Received: {wallet_data['total ether received'].iloc[0]}")
        
        print("\nWallet Activity Pattern:")
        print(f"- Time Span: {wallet_data['Time Diff between first and last (Mins)'].iloc[0]/60/24:.2f} days")
        print(f"- Avg. Time Between Sent Txns: {wallet_data['Avg min between sent tnx'].iloc[0]:.2f} minutes")
        print(f"- Avg. Time Between Received Txns: {wallet_data['Avg min between received tnx'].iloc[0]:.2f} minutes")
        
        print("\nERC20 Token Activity:")
        print(f"- Total ERC20 Transactions: {wallet_data['Total ERC20 tnxs'].iloc[0]}")
        print(f"- Most Sent Token Type: {wallet_data['ERC20_most_sent_token_type'].iloc[0]}")
        print(f"- Most Received Token Type: {wallet_data['ERC20_most_rec_token_type'].iloc[0]}")
        
        print("="*50)
        print("NOTE: This report is for informational purposes only.")
        print("Further investigation may be required for conclusive evidence.")
        print("="*50 + "\n")
        
        return True
    
    def check_illegal_transactions(self, wallet_data):
        """Check for potentially illegal transactions based on patterns"""
        # This is a placeholder for more sophisticated illegal transaction detection
        # In a real system, this would involve checking against known fraud patterns,
        # blacklists, and other indicators of illegal activity
        
        # Some simple heuristics for demonstration:
        address = wallet_data['Address'].iloc[0] if 'Address' in wallet_data.columns else "Unknown"
        suspicious_indicators = []
        
        # Check for extremely large single transactions
        if wallet_data['max value received'].iloc[0] > 1000:
            suspicious_indicators.append(f"Unusually large incoming transaction: {wallet_data['max value received'].iloc[0]} ETH")
        
        if wallet_data['max val sent'].iloc[0] > 1000:
            suspicious_indicators.append(f"Unusually large outgoing transaction: {wallet_data['max val sent'].iloc[0]} ETH")
        
        # Check for high frequency of transactions in a short time
        if wallet_data['Sent tnx'].iloc[0] > 0:
            avg_time_between_sent = wallet_data['Avg min between sent tnx'].iloc[0]
            if avg_time_between_sent < 5 and wallet_data['Sent tnx'].iloc[0] > 50:
                suspicious_indicators.append(f"High frequency outgoing transactions: {wallet_data['Sent tnx'].iloc[0]} transactions with average {avg_time_between_sent:.2f} minutes between them")
        
        # Check for a high ratio of received to sent (could indicate money laundering)
        if wallet_data['Sent tnx'].iloc[0] > 0 and wallet_data['Received Tnx'].iloc[0] > 0:
            ratio = wallet_data['Received Tnx'].iloc[0] / wallet_data['Sent tnx'].iloc[0]
            if ratio > 5:
                suspicious_indicators.append(f"Unusual received/sent ratio: {ratio:.2f}")
        
        # Check for unusual number of unique addresses
        if wallet_data['Unique Sent To Addresses'].iloc[0] > 100:
            suspicious_indicators.append(f"Unusually high number of unique addresses sent to: {wallet_data['Unique Sent To Addresses'].iloc[0]}")
        
        result = {
            'address': address,
            'suspicious_indicators': suspicious_indicators,
            'has_suspicious_activity': len(suspicious_indicators) > 0
        }
        
        return result
    
    def track_ip_address(self, wallet_address):
        """Placeholder for IP address tracking functionality"""
        # This would be implemented by integrating with blockchain node data
        # or other sources that could link wallet addresses to IP addresses
        print(f"IP tracking functionality would be activated for {wallet_address}")
        return "IP tracking activated (placeholder)"
    
    def alert_authorities(self, wallet_address, risk_score, illegal_transactions_report):
        """Placeholder for alerting authorities about suspicious wallets"""
        print(f"ALERT: Suspicious wallet detected: {wallet_address}")
        print(f"Risk Score: {risk_score}")
        print("Suspicious Indicators:")
        for indicator in illegal_transactions_report['suspicious_indicators']:
            print(f"- {indicator}")
        return "Alert sent to authorities (placeholder)"
    
    def log_activity(self, wallet_address, risk_score, prediction_time):
        """Log wallet analysis activity to database"""
        # This would typically write to a database
        # For this example, we'll append to a CSV file
        log_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'wallet_address': wallet_address,
            'risk_score': risk_score,
            'prediction_time': prediction_time
        }
        
        log_df = pd.DataFrame([log_entry])
        log_file = 'wallet_analysis_log.csv'
        
        # Append or create log file
        if os.path.exists(log_file):
            log_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            log_df.to_csv(log_file, index=False)
        
        print(f"Activity logged for wallet {wallet_address}")
        return True

# Example usage
if __name__ == "__main__":
    # Initialize the detector
    detector = EthereumFraudDetector()
    
    # Training mode
    train_model = input("Do you want to train a new model? (y/n): ").lower() == 'y'
    
    if train_model:
        # Train a new model
        training_data = input("Enter path to training data (default: trained_dataset.csv): ") or "trained_dataset.csv"
        detector.train(training_data, optimize=True)
    else:
        # Load existing model
        if not detector.load_model():
            print("No existing model found. Training a new model...")
            training_data = input("Enter path to training data (default: trained_dataset.csv): ") or "trained_dataset.csv"
            detector.train(training_data)
    
    # Analysis mode
    while True:
        print("\n" + "="*50)
        print("ETHEREUM WALLET RISK ASSESSMENT")
        print("="*50)
        print("1. Analyze a wallet address")
        print("2. Retrain model with verified data")
        print("3. Exit")
        choice = input("Enter your choice (1-3): ")
        
        if choice == '1':
            # Get wallet features (in a real system, this would come from your wallet data fetching code)
            wallet_file = input("Enter path to wallet data CSV: ")
            try:
                wallet_data = pd.read_csv(wallet_file)
                
                for i, row in wallet_data.iterrows():
                    address = row['Address']
                    print(f"\nAnalyzing wallet: {address}")
                    
                    # Step 1: Predict risk
                    single_wallet = pd.DataFrame([row])
                    risk_result = detector.predict_risk(single_wallet)
                    
                    print(f"Risk Score: {risk_result['risk_score']:.4f}")
                    print(f"Risk Label: {'HIGH RISK' if risk_result['risk_label'] == 1 else 'LOW RISK'}")
                    
                    # Step 2: Check for illegal transactions if suspicious
                    if risk_result['risk_label'] == 1:
                        print("\nChecking for illegal transactions...")
                        illegal_check = detector.check_illegal_transactions(single_wallet)
                        
                        if illegal_check['has_suspicious_activity']:
                            print("Suspicious activity detected!")
                            print("Suspicious indicators:")
                            for indicator in illegal_check['suspicious_indicators']:
                                print(f"- {indicator}")
                            
                            # Step 3a: Track IP and alert authorities
                            detector.track_ip_address(address)
                            detector.alert_authorities(address, risk_result['risk_score'], illegal_check)
                            
                            # Generate investigative report
                            print("\nGenerating detailed report...")
                            detector.generate_report(single_wallet, risk_result)
                        else:
                            print("No specific illegal transactions detected, but wallet remains high risk.")
                            # Step 3b: Generate standard report
                            detector.generate_report(single_wallet, risk_result)
                    else:
                        # Low risk wallet
                        print("Wallet appears to be low risk.")
                        detector.generate_report(single_wallet, risk_result)
                    
                    # Log activity
                    detector.log_activity(address, risk_result['risk_score'], risk_result['prediction_time'])
                    
                    # Ask for verification to improve model
                    if input("\nDo you have verified information about this wallet? (y/n): ").lower() == 'y':
                        actual_label = int(input("Enter actual label (0 for legitimate, 1 for fraudulent): "))
                        detector.add_verified_data(single_wallet, actual_label)
                
            except Exception as e:
                print(f"Error analyzing wallet: {str(e)}")
        
        elif choice == '2':
            # Retrain model with verified data
            if len(detector.verified_data) > 0:
                training_data = input("Enter path to original training data (default: trained_dataset.csv): ") or "trained_dataset.csv"
                detector.retrain_with_verified_data(training_data)
            else:
                print("No verified data available for retraining.")
        
        elif choice == '3':
            print("Exiting program.")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")