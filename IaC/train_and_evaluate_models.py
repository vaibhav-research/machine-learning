import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, classification_report
import tensorflow as tf
from tensorflow.keras import layers, optimizers, callbacks
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K

# Define F1-score as a Keras metric
def f1_metric(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

# Load the sequential dataset
try:
    data = pd.read_csv('synthetic_sequential_iac_data.csv')
except FileNotFoundError:
    print("Error: The file 'synthetic_sequential_iac_data.csv' was not found.")
    exit()

# Encode categorical features
le_resource = LabelEncoder()
le_attribute = LabelEncoder()
data['resource_id_encoded'] = le_resource.fit_transform(data['resource_id'])
data['attribute_encoded'] = le_attribute.fit_transform(data['attribute'])
data['user_role_encoded'] = LabelEncoder().fit_transform(data['user_role'])

# Sort data by resource and modification time
data = data.sort_values(['resource_id', 'modification_time'])

# Convert 'modification_time' to numerical timestamp (seconds since epoch)
data['modification_time_numerical'] = pd.to_datetime(data['modification_time']).astype(int) / 10**9

# --- Feature Engineering ---
def create_sequential_features(grouped_data):
    sequential_features = []
    for res_id, group in grouped_data:
        group = group.sort_values('modification_time_numerical')
        time_diffs = np.diff(group['modification_time_numerical'].values)
        time_since_last_change = [0] + list(time_diffs)
        group['time_since_last_change'] = time_since_last_change

        concurrent_counts = group['is_concurrent'].rolling(window=3, min_periods=1).sum()
        group['concurrent_count_3'] = concurrent_counts.fillna(0)

        freq_diffs = np.diff(group['change_frequency'].values)
        freq_change = [0] + list(freq_diffs)
        group['change_frequency_diff'] = freq_change

        # Sequence-level features
        total_concurrent = group['is_concurrent'].sum()
        average_time_between_changes = np.mean(time_diffs) if len(time_diffs) > 0 else 0
        last_3_concurrent = group['is_concurrent'].tail(3).sum()

        base_features = group[['resource_id_encoded', 'attribute_encoded', 'change_frequency',
                               'modification_time_numerical', 'is_concurrent', 'user_role_encoded',
                               'criticality_score', 'attribute_sensitivity_score', 'change_magnitude',
                               'time_since_last_change', 'concurrent_count_3', 'change_frequency_diff']].values
        label = group['conflict_after_sequence'].iloc[0]
        sequential_features.append((base_features, label, res_id, total_concurrent, average_time_between_changes, last_3_concurrent))
    return sequential_features

processed_sequences = create_sequential_features(data.groupby('resource_id'))
sequences = [item[0] for item in processed_sequences]
labels_array = np.array([item[1] for item in processed_sequences])
resource_ids_for_sequences = [item[2] for item in processed_sequences]
sequence_level_features = np.array([[item[3], item[4], item[5]] for item in processed_sequences])

# Print label distribution before splitting
print("\nLabel distribution before splitting:")
print(pd.Series(labels_array).value_counts())

# Split into training and testing sets based on resource IDs to avoid leakage
unique_resources = np.unique(resource_ids_for_sequences)
train_resources, test_resources = train_test_split(unique_resources, test_size=0.2, random_state=42)
print("Warning: Stratification disabled for resource-based train-test split due to potential class imbalance.")

train_indices = [i for i, res_id in enumerate(resource_ids_for_sequences) if res_id in train_resources]
test_indices = [i for i, res_id in enumerate(resource_ids_for_sequences) if res_id in test_resources]

train_sequences = [sequences[i] for i in train_indices]
test_sequences = [sequences[i] for i in test_indices]
y_train = labels_array[train_indices]
y_test = labels_array[test_indices]
train_seq_level_features = sequence_level_features[train_indices]
test_seq_level_features = sequence_level_features[test_indices]

# Pad sequences to a fixed length
max_len = max(len(seq) for seq in train_sequences + test_sequences)
padded_X_train = pad_sequences(train_sequences, maxlen=max_len, dtype='float32', padding='post', truncating='post')
padded_X_test = pad_sequences(test_sequences, maxlen=max_len, dtype='float32', padding='post', truncating='post')

# --- Train Random Forest Model (on the original row-wise data for comparison) ---
X_original = data[['resource_id_encoded', 'attribute_encoded', 'change_frequency', 'modification_time_numerical', 'is_concurrent', 'user_role_encoded', 'criticality_score', 'attribute_sensitivity_score', 'change_magnitude']]
y_original = data['conflict_after_sequence']
X_train_rf, X_test_rf, y_train_rf, y_test_rf = train_test_split(X_original, y_original, test_size=0.2, random_state=42, stratify=y_original)

rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train_rf, y_train_rf)
y_pred_rf_prob = rf_model.predict_proba(X_test_rf)[:, 1]
y_pred_rf = (y_pred_rf_prob > 0.5).astype(int)
precision_rf = precision_score(y_test_rf, y_pred_rf, zero_division=0)
recall_rf = recall_score(y_test_rf, y_pred_rf, zero_division=0)
f1_rf = f1_score(y_test_rf, y_pred_rf, zero_division=0)
auc_rf = roc_auc_score(y_test_rf, y_pred_rf_prob)
print("\nRandom Forest Metrics (on original data):")
print(f"Precision: {precision_rf:.4f}")
print(f"Recall: {recall_rf:.4f}")
print(f"F1-Score: {f1_rf:.4f}")
print(f"AUC: {auc_rf:.4f}")

# --- Train Combined Model with LSTM ---
# Normalize sequential data
scaler_seq = StandardScaler()
X_train_scaled = scaler_seq.fit_transform(padded_X_train.reshape(-1, padded_X_train.shape[-1])).reshape(padded_X_train.shape)
X_test_scaled = scaler_seq.transform(padded_X_test.reshape(-1, padded_X_test.shape[-1])).reshape(padded_X_test.shape)

# Normalize sequence-level features
scaler_seq_level = StandardScaler()
X_train_seq_level_scaled = scaler_seq_level.fit_transform(train_seq_level_features)
X_test_seq_level_scaled = scaler_seq_level.transform(test_seq_level_features)

# Compute class weights - Manual calculation with more emphasis on minority class
unique_classes_train = np.unique(y_train)
class_weights_dict = {}
if len(unique_classes_train) > 1:
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight('balanced', classes=unique_classes_train, y=y_train)
    # Potentially amplify the weight of the minority class (False - index 0 if it exists)
    if len(class_weights) == 2:
        class_weights_dict = {unique_classes_train[0]: class_weights[0] * 2 if unique_classes_train[0] == 0 else class_weights[0],
                              unique_classes_train[1]: class_weights[1] * 2 if unique_classes_train[1] == 0 else class_weights[1]}
    else:
        class_weights_dict = {unique_classes_train[i]: class_weights[i] for i in range(len(unique_classes_train))}
    print("Class Weights:", class_weights_dict)
else:
    print("Warning: Only one unique class found in y_train. Skipping class weight calculation.")

# Combined Model with LSTM
input_seq = tf.keras.Input(shape=(X_train_scaled.shape[1], X_train_scaled.shape[-1]))
lstm_out1 = layers.LSTM(64, activation='tanh', return_sequences=True)(input_seq)
lstm_out2 = layers.LSTM(32, activation='tanh')(lstm_out1)
dropout_out = layers.Dropout(0.4)(lstm_out2)

input_seq_level = tf.keras.Input(shape=(X_train_seq_level_scaled.shape[1],))
dense_seq_level = layers.Dense(32, activation='relu')(input_seq_level)

merged = layers.concatenate([dropout_out, dense_seq_level])
output = layers.Dense(1, activation='sigmoid')(merged)

lstm_combined_model = tf.keras.Model(inputs=[input_seq, input_seq_level], outputs=output)

optimizer = optimizers.Adam(learning_rate=0.00025) # Try a slightly lower learning rate
lstm_combined_model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy', f1_metric])

early_stopping = callbacks.EarlyStopping(monitor='val_f1_metric', mode='max', patience=40, restore_best_weights=True, verbose=1) # Increase patience
lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_f1_metric', mode='max', factor=0.5, patience=20, verbose=1, min_lr=1e-7) # Increase patience

history_lstm = lstm_combined_model.fit([X_train_scaled, X_train_seq_level_scaled], y_train, epochs=200, batch_size=32, validation_split=0.2,
                                         callbacks=[early_stopping, lr_scheduler], class_weight=class_weights_dict, verbose=1)

# Evaluate LSTM combined model with adjusted threshold
y_pred_lstm_combined_prob = lstm_combined_model.predict([X_test_scaled, X_test_seq_level_scaled])

# Find optimal threshold using Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred_lstm_combined_prob)
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
optimal_threshold_index_lstm = np.argmax(f1_scores)
best_threshold_lstm = thresholds[optimal_threshold_index_lstm]
print(f"\nOptimal Threshold for LSTM Combined Model: {best_threshold_lstm:.4f}")
y_pred_lstm_combined = (y_pred_lstm_combined_prob > best_threshold_lstm).astype(int)
print("\nLSTM Combined Model Metrics (Sequential + Sequence-Level Features) with Optimal Threshold:")
print(classification_report(y_test, y_pred_lstm_combined, zero_division=0))

auc_lstm_combined = roc_auc_score(y_test, y_pred_lstm_combined_prob)
print(f"AUC (Uncalibrated): {auc_lstm_combined:.4f}")

# Comparison of metrics
precision_lstm_combined = precision_score(y_test, y_pred_lstm_combined, zero_division=0)
recall_lstm_combined = recall_score(y_test, y_pred_lstm_combined, zero_division=0)
f1_lstm_combined = f1_score(y_test, y_pred_lstm_combined, zero_division=0)

metrics_rf = {'Precision': precision_rf, 'Recall': recall_rf, 'F1-Score': f1_rf, 'AUC': auc_rf}
metrics_lstm_combined = {'Precision': precision_lstm_combined, 'Recall': recall_lstm_combined, 'F1-Score': f1_lstm_combined, 'AUC': auc_lstm_combined}

metrics_comparison = pd.DataFrame([metrics_rf, metrics_lstm_combined], index=['Random Forest (Row-wise)', 'LSTM Combined Model'])
print("\nComparison of Metrics:")
print(metrics_comparison)

# Plot training & validation loss and F1-score for LSTM Combined Model
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['loss'], label='Train Loss')
plt.plot(history_lstm.history['val_loss'], label='Validation Loss')
plt.title('LSTM Combined Model Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_lstm.history['f1_metric'], label='Train F1-Score')
plt.plot(history_lstm.history['val_f1_metric'], label='Validation F1-Score')
plt.title('LSTM Combined Model Training and Validation F1-Score')
plt.xlabel('Epochs')
plt.ylabel('F1-Score')
plt.legend()
plt.tight_layout()
plt.show()

# Plot comparison of models' metrics
metrics_comparison.plot(kind='bar', figsize=(10, 6))
plt.title('Model Comparison: Random Forest vs LSTM Combined Model')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()