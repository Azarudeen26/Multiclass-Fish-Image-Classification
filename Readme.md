# 🐠 Multiclass Fish Image Classification  
**Deep Learning for Automated Marine Species Identification**  
**Developed by:** AzaruDeen  

---

## 🌟 Executive Summary

This project presents a **production-grade deep learning system** designed for **multi-class fish species classification**, combining **data engineering**, **transfer learning**, and **interactive deployment** using **TensorFlow, Keras, and Streamlit**.  

It demonstrates **end-to-end AI expertise** — from dataset preprocessing and model training to deployment through an advanced Streamlit web interface.  
The solution is scalable, interpretable, and optimized for **real-time marine species recognition**.

---

## 🎯 Core Objectives

- Build a **robust image classifier** for identifying multiple fish species from real-world photographs.  
- Compare and analyze multiple **CNN architectures and transfer learning models**.  
- Optimize accuracy and minimize overfitting using **augmentation, dropout, and batch normalization**.  
- Deploy a **Streamlit-based web app** enabling real-time inference and visualization.  
- Create a **modular and extensible workflow** adaptable for future AI applications.  

---

## project folder stucture

Multiclass_Fish_Image_Classification/
│
├── data/                                      # Folder containing all image datasets
│   ├── train/                                 # Training images, organized by class folders
│   │   ├── animal fish/
│   │   ├── animal fish bass/
│   │   ├── fish sea_food black_sea_sprat/
│   │   ├── fish sea_food gilt_head_bream/
│   │   ├── fish sea_food hourse_mackerel/
│   │   ├── fish sea_food red_mullet/
│   │   ├── fish sea_food red_sea_bream/
│   │   ├── fish sea_food sea_bass/
│   │   ├── fish sea_food shrimp/
│   │   ├── fish sea_food striped_red_mullet/
│   │   └── fish sea_food trout/
│   │
│   ├── validation/                            # Validation images, organized by same class folders
│   │   ├── animal fish/
│   │   ├── animal fish bass/
│   │   └── ... (other classes)
│   │
│   └── test/                                  # Test images, organized by same class folders
│       ├── animal fish/
│       ├── animal fish bass/
│       └── ... (other classes)
│
├── tf_env/                                    # Virtual environment folder
├── Custom CNN/                                # Folder containing your custom CNN model files
│
├── data.ipynb                                 # Jupyter Notebook for data exploration & preprocessing
├── VGG16_best.keras                           # Best VGG16 model checkpoint
├── VGG16_final.keras                          # Final VGG16 model
├── VGG16_final.h5                             # Final VGG16 model in H5 format
├── ResNet50_best.keras                        # Best ResNet50 model checkpoint
├── ResNet50_final.keras                       # Final ResNet50 model
├── ResNet50_final.h5                          # Final ResNet50 model in H5 format
├── MobileNet_best.keras                        # Best MobileNet model checkpoint
├── MobileNet_final.keras                       # Final MobileNet model
├── MobileNet_final.h5                          # Final MobileNet model in H5 format
├── InceptionV3_best.keras                      # Best InceptionV3 model checkpoint
├── InceptionV3_final.keras                     # Final InceptionV3 model
├── InceptionV3_final.h5                        # Final InceptionV3 model in H5 format
├── EfficientNetV2B0_best.keras                 # Best EfficientNetV2B0 model checkpoint
├── EfficientNetV2B0_final.keras                # Final EfficientNetV2B0 model
├── EfficientNetV2B0_final.h5                   # Final EfficientNetV2B0 model in H5 format
├── PREDICT.PY                                  # Script for predicting fish class from an image
├── Class_Label.json                            # JSON file mapping class indices to fish names
├── app.py                                      # Streamlit app for interactive multi-model prediction
├── Readme.md                                   # Project documentation and insights
└── requirements.txt                            # Python dependencies for project

---

## 🚀 Technical Highlights & Value Additions

| **Category** | **What You Implemented** | **Professional Value** |
|--------------|--------------------------|--------------------------|
| **Data Engineering** | Used `ImageDataGenerator` for train/validation/test splits with real-time augmentation | Ensures reproducibility, scalability, and robustness |
| **Modeling Expertise** | Trained 6 CNN architectures — Custom CNN, VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetB0 | Demonstrates mastery of deep learning architectures and comparative analysis |
| **Transfer Learning** | Leveraged pre-trained ImageNet weights and fine-tuned final layers | Accelerated convergence and improved feature extraction |
| **Optimization & Regularization** | Applied dropout, batch normalization, and augmentation | Improved generalization and reduced overfitting |
| **Model Evaluation** | Used metrics like accuracy, loss curves, and confusion matrices | Enabled precise model comparison and interpretability |
| **Deployment** | Built Streamlit dashboard for multi-model prediction with dynamic visual analytics | Showcases full-stack ML deployment capability |
| **Automation** | Integrated JSON-based label mapping and reusable prediction functions | Highlights engineering discipline and modularity |
| **Result Analysis** | Compared model performance, selecting **MobileNet** as the best model | Data-driven decision-making with practical insight |

---

## 🧠 Deep Insights & Learnings

### 1️⃣ Dataset Understanding & Preprocessing
- Balanced data representation using **augmentation** techniques (rotation, zoom, flip).  
- Normalized pixel values (`rescale=1/255`) ensuring consistent input distribution.  
- Enhanced dataset diversity to simulate real-world underwater conditions such as lighting and angle variations.  

### 2️⃣ Model Development Insights
- **Custom CNN** established a strong baseline (Train Acc: 82%, Test Acc: 86.48%).  
- Transition to **transfer learning models** significantly boosted performance and reduced training time.  
- Applied **layer freezing** and **fine-tuning** for better generalization across architectures.  

| **Model** | **Validation Accuracy (%)** | **Test Accuracy (%)** | **Remarks** |
|------------|-----------------------------|------------------------|--------------|
| Custom CNN | 82.00 | 86.48 | Strong baseline with good generalization |
| VGG16 | 92.75 | 95.42 | High accuracy but computationally heavy |
| ResNet50 | 46.09 | 56.82 | Underperformed due to gradient instability |
| **MobileNet** | **95.97** | **97.52** | ✅ Best performing model – fast, lightweight, and accurate |
| InceptionV3 | 95.81 | 97.21 | Excellent performance, slightly heavier |
| EfficientNetB0 | 17.65 | 16.32 | Struggled due to overfitting and data mismatch |

---

### 3️⃣ Model Evaluation & Interpretability
- Visualized **training/validation accuracy & loss curves** for all models.  
- Analyzed confusion matrices to identify visually similar fish classes.  
- Implemented **Top-3 prediction charts** in the app for deeper interpretability.  

### 4️⃣ Comparative Analysis & Decision
After thorough comparison across **accuracy, inference speed, and scalability**,  
**MobileNet** was selected as the **production-ready model**:  
- ✅ 97.52% test accuracy  
- ✅ Low latency  
- ✅ Lightweight & deployable on mobile or edge devices  

---

## 💻 Streamlit Application Layer

The deployed **Streamlit dashboard** integrates model inference, comparison, and visualization:

### 🔹 Key Features:
- Multi-model selection and parallel inference.  
- Interactive bar charts showing **Top-3 predictions** with probabilities.  
- Real-time batch image upload and result table.  
- Downloadable CSV of all prediction results.  
- Dynamic color-coded confidence visualization using **Altair**.  

This provides both **technical depth** for analysts and **ease of use** for non-technical users.

---

## 📈 Business / Research Impact

| **Domain** | **Practical Application** |
|-------------|----------------------------|
| 🌊 **Marine Research** | Automated fish species identification for biodiversity studies |
| 🧾 **Seafood Industry** | Sorting and quality control in seafood processing lines |
| 📱 **Edge AI Systems** | Deploy lightweight models like MobileNet for mobile or embedded cameras |
| 💡 **Scalable AI Framework** | Reusable template for future datasets (e.g., fruits, plants, or medical images) |

---

## 💬 Key Technical Takeaways

- Mastery over **end-to-end deep learning workflows**.  
- Proficiency in **TensorFlow, Keras, and Transfer Learning**.  
- Designed **multi-model analytical pipelines** with real-time inference.  
- Built modular, production-ready code for easy maintenance.  
- Delivered a **Streamlit app** combining functionality, aesthetics, and explainability.  

---

## 🛠 Technical Stack

Language: Python 3

Deep Learning: TensorFlow, Keras, CNNs, Transfer Learning (VGG16, ResNet50, MobileNet, InceptionV3, EfficientNetV2B0)

Data & Preprocessing: NumPy, Pandas, Pillow, ImageDataGenerator

Visualization: Matplotlib, Seaborn, Altair

Deployment: Streamlit, JSON, CSV export

Environment: Virtualenv (tf_env), requirements.txt

---

📘 **Author:** AzaruDeen  
🔗 *For AI Research, Industry Collaboration, and Computer Vision Projects*
