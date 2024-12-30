# Load data
train_data, train_labels = load_ecog_data('HCTSA_train_ch10.mat')
test_data, test_labels = load_ecog_data('HCTSA_validate1_ch10.mat')

# Preprocess data
fs = 1000  # Sampling frequency
train_data_preprocessed = preprocess_data(train_data, fs)
test_data_preprocessed = preprocess_data(test_data, fs)

# Extract features
train_features = compute_features(train_data_preprocessed, fs)
test_features = compute_features(test_data_preprocessed, fs)

# Train and evaluate the classifier
classifier = NearestMedianClassifier()
classifier.fit(train_features, train_labels.flatten())

train_accuracy = classifier.score(train_features, train_labels.flatten())
test_accuracy = classifier.score(test_features, test_labels.flatten())

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Validation Accuracy: {test_accuracy * 100:.2f}%")
