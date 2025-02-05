# Plant Disease Detection  System

My objective is to aid in the early detection of plant diseases, allowing you to take prompt action to protect your crops and ensure a healthy production.

## How It Works

1. **Upload Image**: Navigate to the https://plant-diseaseapp.streamlit.app/ and either upload an image of a plant.
2. **Analysis**: Our deep learning model processes the image to detect signs of diseases based on trained algorithms.
3. **Results**: Once the analysis is complete, you’ll see the predicted disease along with its symptoms and recommended treatment.

## About the Model

The model is based on **VGG-16** architecture, a widely used pre-trained model for image recognition tasks. We leverage transfer learning to fine-tune the VGG-16 model for plant disease recognition.

## About the Dataset

This dataset consists of approximately 77,000 RGB images of healthy and diseased crop leaves, categorized into 33 different classes. The total dataset is divided into an 80/20 training and validation split, preserving the directory structure.

- **Training Set**: 60,930 images
- **Validation Set**: 17,572 images
- **Test Set**: 33 images

## Requirements

To run the project locally, ensure you have the following Python packages installed:

- `tensorflow` >= 2.0
- `keras`
- `numpy`
- `matplotlib`
- `pandas`
- `scikit-learn`

You can install these dependencies via:

```bash
pip install -r requirements.txt
