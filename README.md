# ✍️ Handwritten Digit Recognition using CNN

This project implements a deep learning model to recognize handwritten digits using a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset**. The model achieves high accuracy in classifying digits from 0 to 9 based on pixel values.

---

## 🧠 Overview

- **Dataset**: [MNIST](http://yann.lecun.com/exdb/mnist/)
- **Model**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow & Keras
- **Accuracy**: > 98% on test set
- **Input Format**: 28x28 grayscale images

---

## 🖼 Sample Digits

The MNIST dataset contains 60,000 training and 10,000 test grayscale images of handwritten digits.

![MNIST Digits](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

---

## 📂 Project Structure

```bash
handwritten-digit-recognition/
│
├── digit_recognition_cnn.ipynb         # Main notebook (Google Colab/Jupyter)
├── model/
│   ├── digit_model.h5                  # Trained CNN model
│
├── images/
│   └── test_digit.png                  # (Optional) Custom test image
│
├── utils/
│   └── preprocess.py                   # (Optional) Image preprocessing utilities
│
└── README.md                           # Project documentation
```

---

## 🚀 How to Run

### ✅ Step 1: Install Dependencies

```bash
pip install tensorflow keras numpy matplotlib opencv-python
```

### ✅ Step 2: Open the Notebook

Open `digit_recognition_cnn.ipynb` using Jupyter Notebook or Google Colab.

### ✅ Step 3: Run All Cells

The notebook includes:
- Data loading and preprocessing
- CNN model building
- Training and evaluation
- Predictions on custom test images

---

## 📊 Model Architecture

- **Input**: 28x28x1 grayscale image
- **Conv2D**: 32 filters, kernel size 3x3, ReLU
- **MaxPooling2D**: 2x2
- **Conv2D**: 64 filters, kernel size 3x3, ReLU
- **MaxPooling2D**: 2x2
- **Flatten**
- **Dense**: 128 units, ReLU
- **Output**: Dense(10), softmax

---

## 📈 Performance

- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98%
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam

---

## 🔍 Predict Custom Image

To test with your own digit image (e.g., `test_digit.png`):
1. Upload to `/images/`
2. Use `cv2.imread()` and preprocessing (grayscale, resize, normalize)
3. Call model prediction

```python
img = cv2.imread('images/test_digit.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (28, 28))
img = img.reshape(1, 28, 28, 1) / 255.0
prediction = model.predict(img)
print("Predicted Digit:", np.argmax(prediction))
```


## 📌 Dataset Reference

- [MNIST Handwritten Digits Dataset](http://yann.lecun.com/exdb/mnist/)
- Available via `tensorflow.keras.datasets.mnist`

---

## 👤 Author

**Adilshakhan Pathan**  
B.Tech in CSE (AI) | Machine Learning & Computer Vision Enthusiast  
🔗 [LinkedIn](https://www.linkedin.com/in/yourprofile) • 🐙 [GitHub](https://github.com/yourusername)

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
