# Audio Classification CNN Project

A deep learning project that uses a ResNet-based Convolutional Neural Network to classify environmental sounds from the ESC-50 dataset. The project includes training infrastructure using Modal for cloud computing, a FastAPI inference service, and a Next.js web interface for visualization.

## 🎯 Project Overview

This project implements an audio classification system that can identify 50 different environmental sound categories using a custom ResNet architecture optimized for audio spectrograms. The system processes audio files by converting them to mel-spectrograms and classifying them using deep learning.

### Key Features

- **Custom ResNet Architecture**: Modified ResNet with residual blocks for audio classification
- **ESC-50 Dataset**: Training on 50 environmental sound categories
- **Cloud Training**: Modal-based distributed training with GPU support
- **Real-time Inference**: FastAPI service for audio classification
- **Web Interface**: Next.js frontend for interactive audio analysis
- **Feature Visualization**: Real-time visualization of neural network activations

## 📊 Dataset Information

**ESC-50 Dataset** (Environmental Sound Classification)

- **Source**: [https://github.com/karolpiczak/ESC-50](https://github.com/karolpiczak/ESC-50)
- **Classes**: 50 environmental sound categories
- **Samples**: 2,000 audio recordings (40 per class)
- **Duration**: 5 seconds per clip
- **Format**: 44.1 kHz WAV files

### Sound Categories Include:

- **Animals**: dogs, cats, birds, insects, frogs
- **Natural soundscapes**: rain, wind, fire, crackling fire
- **Human sounds**: crying, coughing, footsteps, clapping
- **Interior/domestic sounds**: clocks, vacuum cleaner, door slam
- **Exterior/urban sounds**: car horns, trains, helicopters, sirens

## 🏗️ Model Architecture

### ResNet-Based Audio CNN

The model uses a custom ResNet architecture specifically designed for audio classification:

```
Input: Mel-Spectrogram (1 x 128 x Variable)
├── Initial Conv Block (1→64 channels)
│   ├── Conv2d(7x7, stride=2) + BatchNorm + ReLU + MaxPool
├── Layer 1: 3 x ResidualBlock(64→64)
├── Layer 2: 4 x ResidualBlock(64→128, first stride=2)
├── Layer 3: 6 x ResidualBlock(128→256, first stride=2)
├── Layer 4: 3 x ResidualBlock(256→512, first stride=2)
├── Global Average Pooling
├── Dropout(0.5)
└── Linear(512→50 classes)
```

### ResidualBlock Architecture

```
Input
├── Conv2d(3x3) → BatchNorm → ReLU
├── Conv2d(3x3) → BatchNorm
├── Skip Connection (1x1 conv if dimensions change)
└── Add → ReLU → Output
```

**Key Design Decisions:**

- **Mel-Spectrograms**: Audio converted to 128-band mel-spectrograms
- **Residual Connections**: Enables training deep networks without vanishing gradients
- **Data Augmentation**: Frequency/time masking, mixup for robustness
- **Adaptive Pooling**: Handles variable-length audio inputs

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Node.js 16+ (for web interface)
- CUDA-compatible GPU (recommended)
- Modal account (for cloud training)

### Local Setup

1. **Clone the Repository**

```bash
git clone https://github.com/hangsheng0625/Audio-CNN.git
cd Audio-CNN
```

2. **Set Up Python Environment**

```bash
# Create virtual environment
python -m venv .venv

# Activate environment (Windows)
.venv\Scripts\activate

# Activate environment (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Set Up Web Interface**

```bash
cd audio-cnn
npm install
```

### Cloud Training with Modal

1. **Install Modal**

```bash
pip install modal
modal setup
```

2. **Deploy Training**

```bash
modal run train.py
```

3. **Monitor Training**

- TensorBoard logs are saved to `tensorboard_logs/`
- Model checkpoints saved to Modal volume

### Running Inference

1. **Start FastAPI Service**

```bash
modal run main.py
```

2. **Start Web Interface**

```bash
cd audio-cnn
npm run dev
```

3. **Access Application**

- Web Interface: http://localhost:3000
- API Documentation: Available through Modal deployment

## 📁 Project Structure

```
Audio-CNN/
├── model.py              # ResNet architecture definition
├── train.py              # Modal-based training script
├── main.py               # FastAPI inference service
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
├── tensorboard_logs/    # Training logs and metrics
└── audio-cnn/           # Next.js web interface
    ├── src/
    │   ├── app/         # Next.js app router
    │   ├── components/  # React components
    │   │   ├── Waveform.tsx      # Audio waveform display
    │   │   ├── FeatureMap.tsx    # CNN feature visualization
    │   │   └── ColorScale.tsx    # Visualization utilities
    │   └── lib/         # Utility functions
    ├── package.json     # Node.js dependencies
    └── next.config.js   # Next.js configuration
```

## 🔧 Technical Details

### Audio Preprocessing

- **Sample Rate**: 22,050 Hz
- **Mel-Spectrogram**: 128 mel bands
- **Window**: 1024 samples (46ms)
- **Hop Length**: 512 samples (23ms)
- **Frequency Range**: 0-11,025 Hz

### Training Configuration

- **Optimizer**: Adam with OneCycleLR scheduler
- **Batch Size**: 32 (adjustable based on GPU memory)
- **Epochs**: 50-100 (with early stopping)
- **Loss Function**: CrossEntropyLoss with label smoothing
- **Data Augmentation**: SpecAugment, Mixup

### Model Performance

- **Target Accuracy**: 85%+ on ESC-50 test set
- **Inference Speed**: <100ms per audio clip
- **Model Size**: ~23M parameters

## 🌐 Web Interface Features

The Next.js web application provides:

- **Audio Upload**: Drag-and-drop audio file upload
- **Real-time Classification**: Instant prediction results
- **Waveform Visualization**: Interactive audio waveform display
- **Feature Map Visualization**: CNN layer activation visualization
- **Confidence Scores**: Probability distribution across all 50 classes

## 📈 Training Monitoring

### TensorBoard Integration

```bash
tensorboard --logdir tensorboard_logs
```

**Tracked Metrics:**

- Training/Validation Loss
- Training/Validation Accuracy
- Learning Rate Schedule
- Feature Map Activations

## 🙏 Acknowledgments

- **ESC-50 Dataset**: Karol J. Piczak for the environmental sound dataset
- **Modal**: Cloud infrastructure for distributed training
- **PyTorch**: Deep learning framework
- **librosa**: Audio processing library
- **Next.js**: React framework for web interface
