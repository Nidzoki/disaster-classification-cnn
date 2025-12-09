# disaster-classification-cnn

## Natural Disaster Image Classification using Convolutional Neural Networks (CNN)

The goal of this project is to develop a model based on a **Convolutional Neural Network (CNN)** for automatic classification of images depicting various types of natural disasters.

***

### 📋 Table of Contents

1.  [About the Project](#-about-the-project)
2.  [Project Goal](#-project-goal)
3.  [Technologies](#-technologies)
4.  [Repository Structure](#-repository-structure)
5.  [Installation and Running](#-installation-and-running)
6.  [Dataset](#-dataset)
7.  [License](#-license)

***

### 💡 About the Project

This project implements a deep learning solution for classifying images that belong to different categories of natural disasters (e.g., earthquakes, floods, fires). Using the Convolutional Neural Network architecture, the model learns to extract key visual features from images to accurately identify the type of disaster.

The project is primarily developed in **Jupyter Notebooks**, which allows for interactive experimentation with data preprocessing, model building, and evaluation.

### 🎯 Project Goal

The main objectives of the project are:

* **Build and train** a robust CNN model capable of classifying natural disaster images with high accuracy.
* **Demonstrate the effectiveness** of CNN in visual recognition in the context of disaster management.
* Provide **two notebook versions** – one for local running and one optimized for Google Colab.

***

### 🛠️ Technologies

The project is built using the following key tools and libraries (the full list is detailed in the `requirements.txt` file):

| Component | Technology/Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.x | Core programming language. |
| **Deep Learning** | **TensorFlow** / **Keras** | For building, training, and evaluating the CNN model. |
| **Numerical Ops** | NumPy | Essential library for numerical computations. |
| **Data Analysis** | Pandas | For managing and manipulating data (if needed). |
| **Visualization** | Matplotlib | For displaying accuracy, loss, and example images. |
| **Environment** | Jupyter Notebook / Google Colab | Interactive development environment. |

***

### 📁 Repository Structure

| File/Directory | Description |
| :--- | :--- |
| `disaster_classification.ipynb` | **Main notebook** containing the entire workflow: data loading, preprocessing, CNN architecture definition, training, and model evaluation. |
| `disaster_classification_colab.ipynb` | Notebook adapted for the **Google Colab** environment, including specific commands for loading data from Drive or Kaggle. |
| `disaster_dataset_example/` | Directory showing the **expected dataset structure** (e.g., subdirectories for each disaster class). |
| `requirements.txt` | List of all necessary **Python packages** required to run the project. |
| `LICENSE` | File containing the terms of the **MIT license**. |

***

### ⚙️ Installation and Running

Follow these steps to set up the project on your local machine:

#### 1. Clone the Repository

```bash
git clone [https://github.com/Nidzoki/disaster-classification-cnn.git](https://github.com/Nidzoki/disaster-classification-cnn.git)
cd disaster-classification-cnn
```
#### 2. Create Virtual Enviroment (Recommended)
```bash
python -m venv venv
# Activate environment (Linux/macOS)
source venv/bin/activate
# Activate environment (Windows)
.\venv\Scripts\activate
```

#### 3. Install Required Libraries
```bash
pip install -r requirements.txt
```

#### 4. Run the Project
- Local (Jupyter): Start the Jupyter notebook server and open `disaster_classification.ipynb`

```bash
jupyter notebook
```

- Google Colab: Upload and run the disaster_classification_colab.ipynb notebook directly in the Google Colab environment, paying attention to the steps for mounting Google Drive to access the dataset.
___

 ### 📊 Dataset
The model in this repository was trained and evaluated using the "Disaster Damage 5-Class" dataset, available on Kaggle.
- Source: Kaggle -> [Disaster Damage 5-Class](https://www.kaggle.com/datasets/sarthaktandulje/disaster-damage-5class)
- Classes: The dataset is structured into 5 classes of disaster damage.
- Important: You must download this dataset and place the organized images into a directory structure that matches the required format, replacing the placeholder folder (disaster_dataset_example/).

Example Structure:
```
(Project Root)
└── disaster-classification-cnn/
    └── (ACTUAL DATASET FOLDER NAME, e.g., 'disaster_data')/
        ├── fire/
        │   ├── fire_001.jpg
        │   └── ...
        └── flood/
        │   └── flood_001.jpg
        |   └── ...
        └── normal/
        │   └── normal_001.jpg
        |   └── ...
        └── landslide/
        │   └── landslide_001.jpg
        |   └── ...
        └── smoke/
            └── smoke_001.jpg
            └── ...

```
___

### 📝 License
This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

