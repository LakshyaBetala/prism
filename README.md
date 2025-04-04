# Speech-to-Voice Health Analysis Model

## Overview
This project trains a **deep learning model** to classify speech samples as **healthy** or **unhealthy** using **MFCC-based feature extraction** and a **CNN-LSTM** architecture.

## Dataset
- **Format:** `.wav`
- **Classes:** `Healthy`, `Unhealthy`
- **Directory Structure:**
  ```
  dataset/
  ├── Healthy/
  │   ├── sample1.wav
  │   ├── sample2.wav
  ├── Unhealthy/
      ├── sample1.wav
      ├── sample2.wav
  ```

## Model Pipeline
1. **Load Dataset:** Reads `.wav` files from specified directories.
2. **Feature Extraction:** Computes MFCCs from speech signals.
3. **Preprocessing:** Normalizes and converts labels to categorical.
4. **Training:** CNN-LSTM architecture for feature learning.
5. **Evaluation:** Splits data into training & test sets.
6. **Model Saving:** Saves trained model as `speech_health_model.h5`.

## Model Architecture
- **Conv1D Layers** → Extracts local speech features
- **MaxPooling** → Reduces dimensionality
- **LSTM Layer** → Captures temporal dependencies
- **Dense Layers** → Classifies healthy vs unhealthy

## Training Configuration
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Batch Size:** 16
- **Epochs:** 30

## Requirements
- Python 3.8+
- TensorFlow 2.x
- Librosa (for audio processing)
- NumPy, Scikit-learn, Matplotlib

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run training script:
   ```bash
   python train_speech_health_model.py
   ```
3. Model will be saved as `speech_health_model.h5`.

## Future Enhancements
- Improve dataset diversity.
- Implement real-time inference.
- Fine-tune using transformer-based models like Wav2Vec.
