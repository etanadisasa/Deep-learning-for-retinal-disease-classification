# DEEP LEARNING FOR RETINAL DISEASE CLASSIFICATION
**Submitted by: Etana Disasa**



This project develops convolutional neural network (CNN) models to identify retinal diseases from fundus images. Leveraging publicly available datasets, the goal is to create an AI-assisted tool that can aid in early detection and improve clinical decision-making.

---

## What This Project Does

- Trains CNN models on three datasets: RFMiD, APTOS, and a combined set.
- Evaluates and compares model performances using accuracy and loss metrics.
- Finds that the APTOS-trained model achieves the highest accuracy.
- Provides visualization of training progress through learning curves.
- Offers a user-friendly Streamlit app for testing retinal images with the trained models.

---

## Technical Details

The project uses Python and TensorFlow to build and train CNN models tailored for multi-class classification of retinal diseases. It includes scripts for dataset-specific training, utilities for preprocessing, and tools to visualize model learning and performance.

Model artifacts and training histories are saved to enable reproducibility. The Streamlit application allows for interactive image inference, making it easier to explore model outputs without deep technical knowledge.

---

## Datasets

This project uses two publicly available retinal image datasets:

- **RFMiD (Retinal Fundus Multi-Disease Image Dataset)**  
  Download from Kaggle: [https://www.kaggle.com/datasets/andrewmvd/retinal-fundus-images-for-multi-disease-classification](https://www.kaggle.com/datasets/andrewmvd/retinal-fundus-images-for-multi-disease-classification)

- **APTOS (Asia Pacific Tele-Ophthalmology Society Diabetic Retinopathy Detection Dataset)**  
  Download from Kaggle: [https://www.kaggle.com/c/aptos2019-blindness-detection](https://www.kaggle.com/c/aptos2019-blindness-detection)

Please note that you need to create a Kaggle account to access and download these datasets. After downloading, place the data files in the appropriate folders as required by the training scripts.

---

## How to Use

1. Set up your Python environment (Python 3.8+) and install dependencies:

   pip install -r requirements.txt

2. Train models using:

   python Code/01_RFMiDTraining.py

   python Code/02_APTOSTraining.py

   python Code/03_CombinedTraining.py

3. Review evaluation plots located in the Code/Plots directory.

4. Launch the Streamlit app for inference:

   streamlit run Code/app.py

5. You can also see saved models in the Models folders. For each model, there will be training 

history saved int he History folder. 

Note: For a detailed explanation of the project, methodology, and results, please see the [Final Report (PDF)](Report/Etana%20Disasa%20Final%20Report.pdf).

---

## Summary of Model Results

| Model     | Performance     | Notes                                  |
|-----------|-----------------|---------------------------------------|
| RFMiD     | Moderate        | Dataset-specific training only.       |
| APTOS     | Best Accuracy   | Outperformed other models consistently.|
| Combined  | Lower Accuracy  | Combining datasets didn’t improve results.|

---

## References

- Gulshan, Varun, et al. *Development and validation of a deep learning algorithm for detection of diabetic retinopathy in retinal fundus photographs*. JAMA, 2016. [https://doi.org/10.1001/jama.2016.17216](https://doi.org/10.1001/jama.2016.17216)

- Asia Pacific Tele-Ophthalmology Society (APTOS). *Diabetic Retinopathy Detection Dataset*, 2019. [Kaggle APTOS2019](https://www.kaggle.com/c/aptos2019-blindness-detection) (Accessed 2025)

- Retinal Fundus Multi-Disease Image Dataset (RFMiD). *Retinal Fundus Multi-Disease Image Dataset*, 2020. [Kaggle RFMiD](https://www.kaggle.com/datasets/andrewmvd/retinal-fundus-images-for-multi-disease-classification) (Accessed 2025)

- Takahashi, Hidetoshi, et al. *Applying artificial intelligence to disease staging: Deep learning for diabetic retinopathy*. Ophthalmology Retina, 2017. [https://doi.org/10.1016/j.oret.2017.03.009](https://doi.org/10.1016/j.oret.2017.03.009)

- Rajalakshmi, R., et al. *Automated diabetic retinopathy screening and monitoring using digital fundus images*. Journal of Diabetes Science and Technology, 2018. [https://doi.org/10.1177/1932296817747773](https://doi.org/10.1177/1932296817747773)

- Lin, Chung-Yuan, et al. *Multi-task learning for retinal disease classification with improved accuracy*. Computers in Biology and Medicine, 2020. [https://doi.org/10.1016/j.compbiomed.2020.103952](https://doi.org/10.1016/j.compbiomed.2020.103952)

---

## Gratitude

This project was made possible with the insightful mentorship of Dr. Ghulam Mujtaba. His guidance and encouragement were instrumental in navigating the technical and conceptual challenges involved in retinal disease classification. I am deeply thankful for his support, which greatly enriched this research experience.

---
