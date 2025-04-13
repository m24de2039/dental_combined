# 🦷 Multi Task Semantic Image Segmentation for Dental Radiograph
Dental radiography plays a pivotal role in the diagnosis and treatment planning of various oral pathologies. Timely and accurate identification of dental diseases through radiographic analysis is essential for optimal patient outcomes. However, manual interpretation of dental radiographs is often time-consuming and subject to inter-observer variability. This project proposes the development of an automated computer vision system leveraging deep learning techniques to facilitate the objective

# Models Used
# Dental Disease Classification Approach and Techniques
To automate the classification of dental diseases from radiographic images, we designed and trained a Convolutional Neural Network (CNN) model tailored for multi-class image classification. The model distinguishes among seven classes of dental pathologies: Pulpitis, Bony Impaction, Improper Restoration with Chronic Apical Periodontitis, Chronic Apical Periodontitis with Vertical Bone Loss, Embedded Tooth, Dental Caries, and Periodontitis.

# Input
Upload a dental image to predict the type of dental condition using a trained PyTorch CNN model.

# Output
![Alt text](images/imageclass.png)

# Dental Image Segmentation Approach and Techniques
To achieve accurate segmentation of dental structures and lesions from radiographic images, we implemented a deep learning-based semantic segmentation pipeline using a custom U-Net architecture. This encoder-decoder model is well-suited for pixel-wise classification and medical imaging tasks.

# Input
Upload a dental image to predict the type of dental condition using a trained PyTorch CNN model.

# Output
![Alt text](images/imageseg1.png)
![Alt text](images/imageseg2.png)

## Setup (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
