# Load the data
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump


# Example: loading one trial data, you would loop or function this for all trials
def load_trial_data(file_path):
    df = pd.read_csv(file_path)
    features = {}
    for axis in ['x', 'y', 'z']:
        features[f'{axis}_mean'] = df[axis].mean()
        features[f'{axis}_std'] = df[axis].std()
        features[f'{axis}_max'] = df[axis].max()
        features[f'{axis}_min'] = df[axis].min()
        features[f'{axis}_kurtosis'] = df[axis].kurtosis()
        features[f'{axis}_skew'] = df[axis].skew()
        # Add more features as needed
    return features


# Assuming you have a structure for storing files such that you can loop through them
gestures = ['point', 'raise-hand', 'dab', 'hair-swipe', 'rps']
trial_data = []

for gesture in gestures:
    for trial_num in range(0, 10):  # Assuming 10 trials per gesture
        file_path = f'data/Johnny/{gesture}_Johnny_{trial_num}_data.csv'
        features = load_trial_data(file_path)
        features['gesture'] = gesture
        trial_data.append(features)

# Convert to DataFrame
data = pd.DataFrame(trial_data)

# Prepare features and labels
X = data.drop('gesture', axis=1)
y = data['gesture']

# Setup K-Folds
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Model selection
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Training and validation
accuracy_scores = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    accuracy_scores.append(accuracy)
    print("Fold accuracy:", accuracy)

print("Average accuracy:", np.mean(accuracy_scores))

# Final model training
model.fit(X, y)

# Save model to file to be loaded later
dump(model, 'gesture_classifier.joblib')
