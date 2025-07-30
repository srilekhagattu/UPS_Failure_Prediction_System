
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.stats import skew, kurtosis


def run_failure_detection():
    # Load dataset
    df = pd.read_csv("dashboard/Ups_DataSet.csv")

    # Preprocessing
    numeric_cols = ['battery_voltage', 'input_voltage', 'output_voltage',
                    'input_frequency', 'max_output_current_percentage', 'temp']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=numeric_cols, inplace=True)

    if 'last_modified' in df.columns:
        df['last_modified'] = pd.to_datetime(
            df['last_modified'], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['last_modified']).reset_index(drop=True)
        df['time_diff_sec'] = df['last_modified'].diff().dt.total_seconds()
        df['time_gap_flag'] = (df['time_diff_sec'].fillna(
            0) > 2 * df['time_diff_sec'].median()).astype(int)
    else:
        df['time_gap_flag'] = 0

    # Sliding window feature extraction
    def extract_window_features(window):
        stats = []
        for col in numeric_cols:
            values = window[col].values
            std = np.std(values)
            skew_val, kurt_val = (
                0, 0) if std < 1e-6 else (skew(values), kurtosis(values))
            stats.extend([
                np.mean(values), std, np.min(values), np.max(values),
                skew_val, kurt_val, np.median(values)
            ])
        return stats

    window_size = 10
    X_stats, row_indices = [], []
    for i in range(len(df) - window_size + 1):
        window = df.iloc[i:i + window_size]
        X_stats.append(extract_window_features(window))
        row_indices.append(i + window_size - 1)

    X_scaled = StandardScaler().fit_transform(X_stats)

    # KMeans
    kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    kmeans_distances = [np.linalg.norm(
        X_scaled[i] - centroids[kmeans_labels[i]]) for i in range(len(X_scaled))]
    kmeans_threshold = np.percentile(kmeans_distances, 75)

    # Isolation Forest
    iso = IsolationForest(contamination=0.25, random_state=42)
    iso_labels = iso.fit_predict(X_scaled)
    iso_scores = iso.decision_function(X_scaled)
    iso_threshold = np.percentile(iso_scores, 5)

    # Failure classification
    status_cols = ['utility_no_failure', 'battery_not_low', 'bypass_inactive',
                   'ups_no_failure', 'ups_online', 'no_test',
                   'shutdown_inactive', 'beeper_on']
    for col in status_cols:
        df[col] = np.nan

    def classify_failure_with_history(index, df, window=10):
        def update_status(vals):
            for k, v in zip(status_cols, vals):
                df.at[index, k] = v

        if df.at[index, 'anomaly_label'] != -1:
            update_status([1]*8)
            return "Normal"

        start_idx = max(0, index - window)
        history = df.iloc[start_idx:index]
        curr = df.iloc[index]
        if len(history) < window:
            return "Unknown Anomaly"

        bat_dynamic_thresh = history['battery_voltage'].median() - 0.5

        if index > 0:
            prev = df.iloc[index - 1]
            if (
                prev['temp'] >= 25 and curr['temp'] > prev['temp'] and
                180 <= curr['input_voltage'] <= 260 and
                210 <= curr['output_voltage'] <= 240 and
                11 <= curr['battery_voltage'] <= 13 and
                curr['max_output_current_percentage'] < 80
            ):
                update_status([1, 1, 0, 0, 1, 1, 1, 0])
                return "Fan Failure"

        if (
            history['input_frequency'].apply(lambda x: x < 47 or x > 53).mean() > 0.6 and
            (curr['input_frequency'] < 47 or curr['input_frequency'] > 53) and
            curr['battery_voltage'] < 10.5
        ):
            update_status([0, 0, 0, 1, 1, 1, 1, 1])
            return "Input Frequency Fault"

        if (
            history['input_voltage'].apply(lambda x: x < 180 or x > 260).mean() > 0.6 and
            (curr['input_voltage'] < 180 or curr['input_voltage'] > 260) and
            curr['battery_voltage'] < 10.5
        ):
            update_status([0, 0, 0, 1, 1, 1, 1, 1])
            return "Input Voltage Instability"

        if (
            (history['max_output_current_percentage'] > 80).mean() > 0.6 and
            all(history['temp'].diff().dropna() >= 0) and
            curr['max_output_current_percentage'] > 80 and
            curr['temp'] > 35
        ):
            update_status([1, 1, 0, 0, 1, 1, 1, 0])
            return "Overload (Rising Temp)"

        if (
            ((history['output_voltage'] >= 210) & (history['output_voltage'] <= 240)).mean() > 0.6 and
            (curr['output_voltage'] < 210 or curr['output_voltage'] > 240) and
            (210 <= curr['input_voltage'] <= 240)
        ):
            update_status([1, 1, 0, 0, 1, 1, 1, 1])
            return "Output Voltage Issue"

        if (
            (history['battery_voltage'] < 11).mean() > 0.6 and
            curr['battery_voltage'] < 10.5 and
            curr['battery_voltage'] < bat_dynamic_thresh and
            all(history['battery_voltage'].diff().dropna() <= 0) and
            (history['input_voltage'].between(210, 240).mean() > 0.6) and
            (history['max_output_current_percentage'] < 80).mean() > 0.6
        ):
            update_status([1, 0, 0, 1, 1, 1, 1, 1])
            return "Battery Degradation"

        return "Unknown Anomaly"

    # Apply results
    df['kmeans_anomaly'] = np.nan
    df['kmeans_score'] = np.nan
    df['failure_kmeans'] = pd.Series(dtype='object')
    df['iso_anomaly'] = np.nan
    df['iso_score'] = np.nan
    df['failure_iso'] = pd.Series(dtype='object')

    for i, idx in enumerate(row_indices):
        df.at[idx, 'kmeans_anomaly'] = - \
            1 if kmeans_distances[i] > kmeans_threshold else 1
        df.at[idx, 'kmeans_score'] = kmeans_distances[i]
        df.at[idx, 'anomaly_label'] = df.at[idx, 'kmeans_anomaly']
        df.at[idx, 'failure_kmeans'] = classify_failure_with_history(idx, df)

    for i, idx in enumerate(row_indices):
        df.at[idx, 'iso_anomaly'] = iso_labels[i]
        df.at[idx, 'iso_score'] = iso_scores[i]
        df.at[idx, 'anomaly_label'] = df.at[idx, 'iso_anomaly']
        df.at[idx, 'failure_iso'] = classify_failure_with_history(idx, df)

    # Save results to dashboard folder
    df_cleaned = df.iloc[window_size - 1:].reset_index(drop=True)
    df_cleaned[['last_modified'] + numeric_cols + ['kmeans_anomaly', 'kmeans_score',
                                                   'failure_kmeans']].to_csv("dashboard/ups_kmeans_results.csv", index=False)
    df_cleaned[['last_modified'] + numeric_cols + ['iso_anomaly', 'iso_score',
                                                   'failure_iso']].to_csv("dashboard/ups_isolation_forest_results.csv", index=False)

    print("✅ Saved:")
    print("- dashboard/ups_kmeans_results.csv")
    print("- dashboard/ups_isolation_forest_results.csv")
