# Disaster Image Classification CNN

## Convolutional Neural Network for Natural Disaster Image Classification

### 📌 Overview
This project implements a deep learning solution for classifying images of natural disasters using Convolutional Neural Networks. The model is designed to recognize and categorize different types of disaster imagery, which can be valuable for emergency response and disaster management systems.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Project Goals
- Build and train a robust CNN model for accurate disaster image classification
- Demonstrate CNN effectiveness for visual recognition in disaster management contexts
- Provide both local and cloud-based implementations for flexibility
- Serve as an educational resource for CNN applications in real-world scenarios

## 🛠️ Technologies
| Component | Technology | Purpose |
|-----------|------------|---------|
| **Core Language** | Python 3.x | Main programming language |
| **Deep Learning** | TensorFlow/Keras | Model building and training |
| **Data Processing** | NumPy, Pandas | Numerical operations and data manipulation |
| **Visualization** | Matplotlib, Seaborn | Results plotting and data visualization |
| **Development** | Jupyter Notebook | Interactive development environment |
| **Cloud Platform** | Google Colab | GPU-accelerated training |

## 📁 Repository Structure
```
disaster-classification-cnn/
│
├── notebooks/
│   ├── disaster_classification.ipynb           # Main notebook for local execution
│   └── disaster_classification_colab.ipynb     # Colab-optimized notebook
│
├── data/
│   └── disaster_dataset_example/               # Example dataset structure
│       ├── fire/
│       ├── flood/
│       ├── landslide/
│       ├── smoke/
│       └── normal/
│
├── src/ (optional - for future expansion)
│   ├── __init__.py
│   ├── data_preprocessing.py
│   └── model.py
│
├── requirements.txt          # Python dependencies
├── LICENSE                   # MIT License
└── README.md                 # This file
```

## ⚡ Quick Start

### Option 1: Local Installation
```bash
# Clone repository
git clone https://github.com/Nidzoki/disaster-classification-cnn.git
cd disaster-classification-cnn

# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

### Option 2: Google Colab
1. Upload `notebooks/disaster_classification_colab.ipynb` to Google Colab
2. Follow the notebook instructions for mounting Google Drive/Kaggle
3. Run cells sequentially

## 📊 Dataset Preparation

### Using Kaggle Dataset
The project uses the "Disaster Damage 5-Class" dataset from Kaggle:

1. **Get Kaggle API credentials:**
   - Go to https://www.kaggle.com/account
   - Create API token (download `kaggle.json`)

2. **Setup Kaggle CLI:**
```bash
# Install Kaggle CLI
pip install kaggle

# Configure API
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

3. **Download dataset:**
```bash
kaggle datasets download -d sarthaktandulje/disaster-damage-5class -p ./data
unzip ./data/disaster-damage-5class.zip -d ./data/raw_dataset
```

### Expected Dataset Structure
Organize your dataset as follows:
```
data/
└── raw_dataset/
    ├── fire/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── flood/
    ├── landslide/
    ├── smoke/
    └── normal/
```

## 🚀 Model Training

### Key Features
- **Data Augmentation**: Rotation, zoom, flip transformations
- **CNN Architecture**: Custom Conv2D layers with dropout regularization
- **Transfer Learning**: Option to use pre-trained models
- **Evaluation**: Accuracy, precision, recall, and confusion matrix

### Training Parameters
- Image Size: 224×224 pixels
- Batch Size: 32
- Epochs: 50 (with early stopping)
- Optimizer: Adam
- Loss: Categorical Crossentropy

## 📈 Results & Performance
The model achieves competitive accuracy on disaster image classification. Performance metrics include:
- Training/Validation accuracy plots
- Confusion matrix visualization
- Per-class precision and recall scores

## 🔧 Advanced Configuration

### GPU Acceleration
```python
# Enable mixed precision for faster training
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')
```

### Hyperparameter Tuning
Modify these parameters in the notebook:
```python
config = {
    'img_height': 224,
    'img_width': 224,
    'batch_size': 32,
    'epochs': 50,
    'learning_rate': 0.001,
    'dropout_rate': 0.5
}
```

## 🤝 Contributing
Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 style guide
- Add comments and docstrings
- Update documentation as needed
- Test changes before submitting PR

## 🐛 Troubleshooting

### Common Issues
1. **Out of Memory Error**: Reduce batch size or image dimensions
2. **Slow Training**: Enable GPU acceleration or use Colab
3. **Dataset Loading Issues**: Verify folder structure matches expected format
4. **Dependency Conflicts**: Use virtual environment and exact versions from requirements.txt

### Getting Help
Open an issue with:
- OS and Python version
- Error traceback
- Steps to reproduce
- Expected vs actual behavior

## 📚 References
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials/images/cnn)
- [Kaggle Dataset](https://www.kaggle.com/datasets/sarthaktandulje/disaster-damage-5class)
- [CNN for Image Classification](https://arxiv.org/abs/1512.03385)

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
- Kaggle community for datasets and inspiration
- TensorFlow/Keras developers
- Contributors and users of this repository

## 📬 Contact
Maintainer: [Nidzoki](https://github.com/Nidzoki)

---

**Note**: This project is intended for educational and research purposes. Models should be properly validated before deployment in real-world disaster management systems.