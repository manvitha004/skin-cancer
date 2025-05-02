# 🧠 canNet - Custom CNN for Binary Image Classification

**canNet** is a custom-built Convolutional Neural Network (CNN) model created using TensorFlow and Keras to classify medical images into **malignant** and **benign** categories. This deep learning solution avoids transfer learning, focusing instead on a fully custom architecture tailored to the problem domain.

🔬 Ideal for binary classification tasks in medical imaging, research, or educational settings.

## 🚀 Features

- 📊 **Binary Image Classification**: Classifies images as **malignant** or **benign**.
- 🎛️ **Custom CNN Architecture**: No pre-trained models, built from scratch.
- 🧪 **Advanced Data Augmentation**: Improves generalization with brightness, zoom, shear, flips, and rotation.
- 🔁 **Early Stopping**: Stops training when the model starts overfitting.
- 📉 **Comprehensive Evaluation**: Includes accuracy, confusion matrix, and classification report.
- 🧠 **Fine-Tuned Training**: Uses Adam optimizer and categorical crossentropy for multi-class compatibility.

📦 Requirements

pip install tensorflow numpy scikit-learn


🧠 Future Improvements
✅ Add validation split for real-time monitoring

🔍 Use Grad-CAM for visualizing learned features

🧪 Deploy as a web app with Streamlit or Flask

📦 Export the model to TensorFlow Lite or ONNX for deployment

🤝 Contributing
Pull requests and issues are welcome! Please open an issue first to discuss what you would like to change or improve.
