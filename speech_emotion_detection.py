import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Function to extract MFCC features from an audio file
def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_processed = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
    return mfccs_processed

# Function to load data from a specified directory
def load_data(data_directory):
    features = []
    labels = []
    for file_path in glob.glob(os.path.join(data_directory, "*.wav")):
        file_name = os.path.basename(file_path)
        try:
            parts = file_name.split("-")
            label = int(parts[2])  # Adjust based on your dataset's naming format
        except Exception as e:
            print(f"Filename {file_name} not in expected format: {e}")
            continue
        data = extract_features(file_path)
        if data is not None:
            features.append(data)
            labels.append(label)
    return np.array(features), np.array(labels)

def main():
    data_directory = "data"  # Change this if your dataset is in a different folder
    print("Loading data...")
    X, y = load_data(data_directory)
    
    if len(X) == 0:
        print("No data found. Check your data folder and file formats.")
        return

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training the classifier...")
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)
    
    print("Evaluating the classifier...")
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    
    model_filename = "speech_emotion_model.pkl"
    joblib.dump(model, model_filename)
    print(f"Trained model saved as {model_filename}")

if __name__ == "__main__":
    main()
