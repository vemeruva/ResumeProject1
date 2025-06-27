import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import random
import time

# Set page configuration
st.set_page_config(layout="wide", page_title="INTRIA Anomaly Detection", page_icon="🚨")

# --- Utility Functions (Cached for performance) ---

@st.cache_resource(show_spinner="Training Machine Learning Models...")
def train_anomaly_detection_models(X_train, y_train):
    """
    Trains Isolation Forest for unsupervised anomaly detection
    and Random Forest for supervised fraud detection.
    """
    st.write("Training Isolation Forest...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    iso_forest = IsolationForest(random_state=42, contamination=0.01) # Assuming 1% anomalies
    iso_forest.fit(X_scaled)
    st.success("Isolation Forest trained!")

    rf_classifier = None
    if y_train.sum() > 0: # Check if there are any fraud samples
        st.write("Training Random Forest Classifier...")
        rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')
        rf_classifier.fit(X_train, y_train)
        st.success("Random Forest Classifier trained!")
    else:
        st.warning("Not enough fraud samples (y_train has no '1's) to train Random Forest. Skipping.")

    return iso_forest, rf_classifier, scaler

@st.cache_data(show_spinner="Generating synthetic transaction data...")
def generate_synthetic_transaction_data(num_transactions=100000, num_anomalies=500):
    """
    Generates synthetic transaction data with embedded anomalies for demonstration.
    Includes normal transactions, duplicates, outlier amounts, and high velocity.
    """
    transactions = []
    start_time = datetime.now() - timedelta(days=30) # Data for the last 30 days

    account_ids = [f'ACC{i:05d}' for i in range(1, 1001)] # 1000 unique accounts
    locations = ['New York', 'London', 'Paris', 'Tokyo', 'Sydney', 'Mumbai', 'Dubai']
    transaction_types = ['purchase', 'transfer', 'withdrawal', 'deposit']

    # Generate normal transactions
    for i in range(num_transactions):
        tx_id = f'TXN{i:07d}'
        timestamp = start_time + timedelta(minutes=random.randint(0, 30*24*60))
        sender = random.choice(account_ids)
        receiver = random.choice([acc for acc in account_ids if acc != sender])
        amount = round(random.uniform(10.0, 5000.0), 2)
        tx_type = random.choice(transaction_types)
        location = random.choice(locations)
        is_fraud = 0

        transactions.append([tx_id, timestamp, sender, receiver, amount, tx_type, location, is_fraud])

    df = pd.DataFrame(transactions, columns=[
        'transaction_id', 'timestamp', 'sender_account', 'receiver_account',
        'amount', 'transaction_type', 'location', 'is_fraud'
    ])

    # Introduce anomalies
    np.random.seed(42)

    # 1. Duplicate Payments
    for _ in range(int(num_anomalies * 0.3)):
        idx = np.random.randint(0, num_transactions)
        original_tx = df.iloc[idx].copy()
        original_tx['transaction_id'] = f'TXN_DUP_{_}{original_tx["transaction_id"]}'
        original_tx['timestamp'] = original_tx['timestamp'] + timedelta(seconds=random.randint(1, 300))
        original_tx['is_fraud'] = 1
        transactions.append(original_tx.tolist())

    # 2. Outlier Amounts
    for _ in range(int(num_anomalies * 0.4)):
        idx = np.random.randint(0, num_transactions)
        original_tx = df.iloc[idx].copy()
        original_tx['transaction_id'] = f'TXN_OUTLIER_{_}{original_tx["transaction_id"]}'
        if random.random() < 0.5:
            original_tx['amount'] = round(original_tx['amount'] * random.uniform(50, 200), 2)
        else:
            original_tx['amount'] = round(original_tx['amount'] * random.uniform(0.001, 0.1), 2)
        original_tx['is_fraud'] = 1
        transactions.append(original_tx.tolist())

    # 3. High Velocity
    attacker_accounts = random.sample(account_ids, 5)
    for attacker_account in attacker_accounts:
        recent_time = datetime.now() - timedelta(hours=random.randint(1, 24))
        for _ in range(random.randint(20, 50)):
            tx_id = f'TXN_VEL_{attacker_account}_{_}'
            timestamp = recent_time + timedelta(seconds=random.randint(0, 3600))
            receiver = random.choice([acc for acc in account_ids if acc != attacker_account])
            amount = round(random.uniform(50.0, 500.0), 2)
            tx_type = random.choice(['purchase', 'transfer'])
            location = random.choice(locations)
            is_fraud = 1
            transactions.append([tx_id, timestamp, attacker_account, receiver, amount, tx_type, location, is_fraud])

    df = pd.DataFrame(transactions, columns=[
        'transaction_id', 'timestamp', 'sender_account', 'receiver_account',
        'amount', 'transaction_type', 'location', 'is_fraud'
    ]).sort_values(by='timestamp').reset_index(drop=True)

    st.success(f"Generated {len(df)} transactions, with {df['is_fraud'].sum()} marked as fraud (anomalies).")
    return df

@st.cache_data(show_spinner="Engineering features...")
def engineer_features(df):
    """
    Engineers relevant features from raw transaction data using optimized Pandas operations.
    Directly assigns rolling results to avoid merge issues with duplicate timestamps.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by='timestamp').reset_index(drop=True)

    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_of_month'] = df['timestamp'].dt.day

    df_temp_indexed = df.set_index('timestamp')

    st.write("Calculating sender velocity features...")
    df['sender_velocity_1h'] = df_temp_indexed.groupby('sender_account').rolling('1h', closed='left')['transaction_id'].count().values
    df['sender_velocity_24h'] = df_temp_indexed.groupby('sender_account').rolling('24h', closed='left')['transaction_id'].count().values

    st.write("Calculating receiver velocity features...")
    df['receiver_velocity_1h'] = df_temp_indexed.groupby('receiver_account').rolling('1h', closed='left')['transaction_id'].count().values
    df['receiver_velocity_24h'] = df_temp_indexed.groupby('receiver_account').rolling('24h', closed='left')['transaction_id'].count().values

    st.write("Calculating amount deviation features...")
    df['sender_avg_amount_24h'] = df_temp_indexed.groupby('sender_account')['amount'].rolling('24h', closed='left').mean().values
    df['receiver_avg_amount_24h'] = df_temp_indexed.groupby('receiver_account')['amount'].rolling('24h', closed='left').mean().values

    df['sender_avg_amount_24h'] = df['sender_avg_amount_24h'].fillna(df['amount'])
    df['receiver_avg_amount_24h'] = df['receiver_avg_amount_24h'].fillna(df['amount'])

    df['amount_deviation_sender'] = (df['amount'] - df['sender_avg_amount_24h']).fillna(0)
    df['amount_deviation_receiver'] = (df['amount'] - df['receiver_avg_amount_24h']).fillna(0)

    df['amount_to_sender_avg_ratio'] = (df['amount'] / df['sender_avg_amount_24h']).replace([np.inf, -np.inf], 0).fillna(0)
    df['amount_to_receiver_avg_ratio'] = (df['amount'] / df['receiver_avg_amount_24h']).replace([np.inf, -np.inf], 0).fillna(0)

    for col in ['sender_velocity_1h', 'sender_velocity_24h', 'receiver_velocity_1h', 'receiver_velocity_24h',
                'amount_deviation_sender', 'amount_deviation_receiver',
                'amount_to_sender_avg_ratio', 'amount_to_receiver_avg_ratio']:
        df[col] = pd.to_numeric(df[col]).fillna(0)

    st.success("Feature engineering complete!")
    return df

# --- Streamlit UI ---

st.title("🚨 INTRIA Real-Time Anomaly Detection System")
st.markdown("""
    This application demonstrates a real-time anomaly detection system for payment transactions.
    It uses synthetic data, feature engineering, and machine learning models (Isolation Forest and Random Forest)
    to identify fraudulent activities, operational errors, or unusual patterns.
""")

# Define feature columns for models
FEATURE_COLUMNS = [
    'amount', 'hour_of_day', 'day_of_week', 'day_of_month',
    'sender_velocity_1h', 'sender_velocity_24h', 'receiver_velocity_1h', 'receiver_velocity_24h',
    'amount_deviation_sender', 'amount_deviation_receiver',
    'amount_to_sender_avg_ratio', 'amount_to_receiver_avg_ratio'
]

# Initialize session state variables
if 'historical_df' not in st.session_state:
    st.session_state.historical_df = None
if 'historical_df_features' not in st.session_state:
    st.session_state.historical_df_features = None
if 'iso_forest_model' not in st.session_state:
    st.session_state.iso_forest_model = None
if 'rf_classifier_model' not in st.session_state:
    st.session_state.rf_classifier_model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'live_transactions_data' not in st.session_state:
    st.session_state.live_transactions_data = None
if 'processed_batches_count' not in st.session_state:
    st.session_state.processed_batches_count = 0
if 'all_anomalies_detected' not in st.session_state:
    st.session_state.all_anomalies_detected = pd.DataFrame()
if 'total_transactions_processed' not in st.session_state:
    st.session_state.total_transactions_processed = 0

# --- Sidebar for Controls ---
st.sidebar.header("System Controls")

# Data Generation and Model Training
st.sidebar.subheader("Data & Model Setup")
num_historical_transactions = st.sidebar.slider("Number of Historical Transactions", 10000, 200000, 100000, 10000)
num_historical_anomalies = st.sidebar.slider("Number of Historical Anomalies", 100, 1000, 500, 50)

if st.sidebar.button("Generate & Train Models"):
    st.session_state.historical_df = generate_synthetic_transaction_data(
        num_historical_transactions, num_historical_anomalies
    )
    st.session_state.historical_df_features = engineer_features(st.session_state.historical_df.copy())

    X = st.session_state.historical_df_features[FEATURE_COLUMNS].dropna()
    y = st.session_state.historical_df_features['is_fraud'].loc[X.index] # Align y with X's index

    # Handle case where y might be all one class for stratification
    if len(y.unique()) > 1:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    st.session_state.iso_forest_model, st.session_state.rf_classifier_model, st.session_state.scaler = \
        train_anomaly_detection_models(X_train, y_train)

    st.sidebar.success("System Ready: Models Trained!")

# Real-time Simulation Controls
if st.session_state.iso_forest_model:
    st.sidebar.subheader("Real-Time Simulation")
    num_live_transactions = st.sidebar.slider("Transactions per Batch", 10, 200, 50, 10)
    num_live_anomalies_per_batch = st.sidebar.slider("Anomalies per Batch", 0, 10, 2, 1)

    if st.sidebar.button("Generate New Live Stream Data"):
        st.session_state.live_transactions_data = generate_synthetic_transaction_data(
            num_transactions=2000, # Generate a larger pool of live data once
            num_anomalies=50
        )
        st.session_state.processed_batches_count = 0
        st.session_state.all_anomalies_detected = pd.DataFrame()
        st.session_state.total_transactions_processed = 0
        st.sidebar.info("New live stream data generated. Click 'Process Next Batch' to start.")

    if st.sidebar.button("Process Next Batch"):
        if st.session_state.live_transactions_data is not None and \
           st.session_state.processed_batches_count * num_live_transactions < len(st.session_state.live_transactions_data):

            start_idx = st.session_state.processed_batches_count * num_live_transactions
            end_idx = start_idx + num_live_transactions
            chunk_df = st.session_state.live_transactions_data.iloc[start_idx:end_idx].copy()

            if not chunk_df.empty:
                with st.spinner(f"Processing Batch {st.session_state.processed_batches_count + 1}..."):
                    # Scoring logic adapted from original script
                    processed_chunk = engineer_features(chunk_df.copy()) # Engineer features for the chunk

                    # Select only the features used for training
                    X_new = processed_chunk[FEATURE_COLUMNS].dropna()
                    # Filter processed_chunk to align with X_new's index after dropna
                    processed_chunk_aligned = processed_chunk.loc[X_new.index]

                    X_new_scaled = st.session_state.scaler.transform(X_new)
                    processed_chunk_aligned['anomaly_score_iso_forest'] = st.session_state.iso_forest_model.decision_function(X_new_scaled)
                    processed_chunk_aligned['anomaly_score_iso_forest_normalized'] = (
                        1 - (processed_chunk_aligned['anomaly_score_iso_forest'] - processed_chunk_aligned['anomaly_score_iso_forest'].min()) /
                        (processed_chunk_aligned['anomaly_score_iso_forest'].max() - processed_chunk_aligned['anomaly_score_iso_forest'].min())
                    )

                    if st.session_state.rf_classifier_model:
                        processed_chunk_aligned['fraud_probability_rf'] = st.session_state.rf_classifier_model.predict_proba(X_new)[:, 1]
                    else:
                        processed_chunk_aligned['fraud_probability_rf'] = 0.0

                    # Define thresholds for alerting
                    ISO_FOREST_ANOMALY_THRESHOLD = 0.7
                    RF_FRAUD_PROBABILITY_THRESHOLD = 0.7

                    potential_anomalies = processed_chunk_aligned[
                        (processed_chunk_aligned['anomaly_score_iso_forest_normalized'] > ISO_FOREST_ANOMALY_THRESHOLD) |
                        (processed_chunk_aligned['fraud_probability_rf'] > RF_FRAUD_PROBABILITY_THRESHOLD)
                    ].copy()

                    if not potential_anomalies.empty:
                        st.session_state.all_anomalies_detected = pd.concat([st.session_state.all_anomalies_detected, potential_anomalies])
                        st.sidebar.error(f"🚨 Anomalies Detected in Batch {st.session_state.processed_batches_count + 1}!")
                    else:
                        st.sidebar.info(f"No significant anomalies detected in Batch {st.session_state.processed_batches_count + 1}.")

                    st.session_state.processed_batches_count += 1
                    st.session_state.total_transactions_processed += len(chunk_df)

                    st.session_state.last_processed_chunk = processed_chunk_aligned # Store for display

            else:
                st.sidebar.warning("No more live transactions to process in the current stream. Generate new data.")
        else:
            st.sidebar.warning("Please generate historical data and train models first, then generate new live stream data.")

# --- Main Content Area ---
st.header("Dashboard Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric(label="Total Transactions Processed (Simulation)", value=f"{st.session_state.total_transactions_processed:,}")
with col2:
    st.metric(label="Total Anomalies Detected", value=f"{len(st.session_state.all_anomalies_detected):,}")
with col3:
    if st.session_state.historical_df_features is not None:
        actual_anomalies = st.session_state.historical_df_features['is_fraud'].sum()
        st.metric(label="Historical Fraud (Synthetic)", value=f"{actual_anomalies:,}")
    else:
        st.metric(label="Historical Fraud (Synthetic)", value="N/A")

st.markdown("---")

st.subheader("Last Processed Batch Transactions")
if 'last_processed_chunk' in st.session_state and not st.session_state.last_processed_chunk.empty:
    def highlight_anomalies(s):
        if s['anomaly_score_iso_forest_normalized'] > ISO_FOREST_ANOMALY_THRESHOLD or \
           s['fraud_probability_rf'] > RF_FRAUD_PROBABILITY_THRESHOLD:
            return ['background-color: #ffe0e0'] * len(s) # Light red background for anomalies
        return [''] * len(s)

    st.dataframe(
        st.session_state.last_processed_chunk[[
            'transaction_id', 'timestamp', 'sender_account', 'receiver_account', 'amount',
            'transaction_type', 'anomaly_score_iso_forest_normalized', 'fraud_probability_rf', 'is_fraud'
        ]].style.apply(highlight_anomalies, axis=1),
        use_container_width=True
    )
else:
    st.info("No batch processed yet. Generate data and process a batch to see transactions here.")

st.subheader("All Detected Anomalies")
if not st.session_state.all_anomalies_detected.empty:
    st.dataframe(
        st.session_state.all_anomalies_detected[[
            'transaction_id', 'timestamp', 'sender_account', 'receiver_account', 'amount',
            'transaction_type', 'anomaly_score_iso_forest_normalized', 'fraud_probability_rf', 'is_fraud'
        ]].drop_duplicates(subset=['transaction_id']).sort_values(by='timestamp', ascending=False),
        use_container_width=True
    )
else:
    st.info("No anomalies detected yet. Process more batches.")

st.markdown("---")
st.subheader("Model Performance (on Historical Test Set)")
if st.session_state.rf_classifier_model and st.session_state.historical_df_features is not None:
    # Re-evaluate on a test set (simplified for UI, not proper cross-validation)
    X = st.session_state.historical_df_features[FEATURE_COLUMNS].dropna()
    y = st.session_state.historical_df_features['is_fraud'].loc[X.index]

    if len(y.unique()) > 1:
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_pred_rf = st.session_state.rf_classifier_model.predict(X_test)
        y_proba_rf = st.session_state.rf_classifier_model.predict_proba(X_test)[:, 1]

        st.write("Random Forest Classification Report:")
        st.code(classification_report(y_test, y_pred_rf, zero_division=0))
        st.write(f"Random Forest ROC AUC Score: **{roc_auc_score(y_test, y_proba_rf):.4f}**")
    else:
        st.warning("Cannot evaluate Random Forest performance: historical data contains only one class for 'is_fraud'.")
else:
    st.info("Train models to see performance metrics.")

st.markdown("---")
st.caption("Developed for INTRIA Payment Transaction Anomaly Detection.")
