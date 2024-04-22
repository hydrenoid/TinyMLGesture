# Load the data
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns


# Function that takes in a file path for a trial and returns the calculated features
def load_trial_data(file_path):
    df = pd.read_csv(file_path)
    features = {}
    for axis in ['x', 'y', 'z', 'mag']:
        features[f'{axis}_mean'] = df[axis].mean()
        features[f'{axis}_std'] = df[axis].std()
        features[f'{axis}_max'] = df[axis].max()
        features[f'{axis}_min'] = df[axis].min()
        features[f'{axis}_kurtosis'] = df[axis].kurtosis()
        features[f'{axis}_skew'] = df[axis].skew()
        # Add more features as needed
    return features


# ------------------------------------------------------------------
# Lets look at some of the data
# Load data for each trial and plot them overlaying
gestures = ['point', 'raise-hand', 'dab', 'hair-swipe', 'rps']
for gesture in gestures:
    num_trials = 10  # Adjust based on how many trials you have for this gesture
    trials_data = [pd.read_csv(f'data/Johnny/{gesture}_Johnny_{i}_data.csv') for i in range(0, num_trials)]

    # Initialize a plot
    plt.figure(figsize=(14, 8))

    # Plot each trial
    for i, df in enumerate(trials_data, start=1):
        plt.subplot(3, 1, 1)
        plt.plot(df['x'], label=f'Trial {i}', alpha=0.6)  # alpha for transparency
        plt.title(f'X-axis Readings for {gesture}')
        plt.ylabel('X-axis')

        plt.subplot(3, 1, 2)
        plt.plot(df['y'], label=f'Trial {i}', alpha=0.6)
        plt.title(f'Y-axis Readings for {gesture}')
        plt.ylabel('Y-axis')

        plt.subplot(3, 1, 3)
        plt.plot(df['z'], label=f'Trial {i}', alpha=0.6)
        plt.title(f'Z-axis Readings for {gesture}')
        plt.ylabel('Z-axis')
        plt.xlabel('Time (samples)')

    # Add legends to each subplot
    plt.subplot(3, 1, 1)
    plt.legend(loc='upper right')

    plt.subplot(3, 1, 2)
    plt.legend(loc='upper right')

    plt.subplot(3, 1, 3)
    plt.legend(loc='upper right')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot
    plt.show()
# -----------------------------------------------------


gestures = ['point', 'raise-hand', 'dab', 'hair-swipe', 'rps']
trial_data = []

# Load in all of the calculated features for each trial and gesture and put in table
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

# Visualize the features within the dataset and how they affect the model
feature_importances = model.feature_importances_

# Create a pandas series to view the feature importances for better visualization
features = pd.Series(feature_importances, index=X.columns).sort_values(ascending=False)

# Plotting
plt.figure(figsize=(10, 6))
sns.barplot(x=features, y=features.index)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()
