import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
from utils.feature_extraction import extract_features

# Load voice samples
X = []
y = []
target_speaker_dir = 'voice_samples/target_speaker/'
other_speakers_dir = 'voice_samples/other_speakers/'

for file in os.listdir(target_speaker_dir):
    features = extract_features(os.path.join(target_speaker_dir, file))
    X.append(features)
    y.append(1)

for file in os.listdir(other_speakers_dir):
    features = extract_features(os.path.join(other_speakers_dir, file))
    X.append(features)
    y.append(0)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Split the dataset into training and testing sets (training 80% and testing 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# StratifiedKFold for Cross-Validation
# Provides stratified K-Folds cross-validator
# Define StratifiedKFold with n_splits=2
cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

# Perform hyperparameter tuning through cross-validation using GridSearchCV
grid = GridSearchCV(SVC(probability=True), param_grid, refit=True, verbose=2, cv=cv)
grid.fit(X_train, y_train)

# Print the best parameters
print(f'Best parameters found: {grid.best_params_}')

# Evaluate the model
y_pred = grid.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Save the best model
joblib.dump(grid.best_estimator_, 'speaker_recognition_model.pkl')
