# Detecting Flies in Videos to Study Aging

## Dataset Information

12 Normal Categories:
BAS (Basophil), EBO (Erythroblast), EOS (Eosinophil), LYT (Lymphocyte), MMZ (Metamyelocyte), MON (Monocyte), MYB (Myelocyte), NGB (Band neutrophil), NGS (Segmented neutrophil), PEB (Proerythroblast), PLM (Plasma cell), PMO (Promyelocyte)

6 Abnormal Categories:
ABE (Abnormal eosinophil), BLA (Blast), FGC (Faggott cell), HAC (Hairy cell), KSC (Smudge cell), LYI (Immature lymphocyte)

3 "Other" Categories:
ART (Artefact), NIF (Not identifiable), OTH (Other cell)

## Presentation/Information
We will use the following criteria to grade your presentations *(note: this applies only to the presentation, not the written report)*:

| Category | Points | Evaluation Criteria | Our Response |
| :--- | :---: | :--- | :--- |
| **1. Problem Definition & Medical Relevance** | 4 | Clearly defines the problem and explains its clinical or biomedical significance. Demonstrates understanding of how AI/ML could address the problem and its potential impact on healthcare or biomedical research. | In this project, we will use various image detection methods to predict and detect cell abnormalities from images of bone marrow biopsies cells, with evaluation using F1-score, accuracy, precision, and recall. The dataset that we will be using is the “Bone Marrow Cell Classification” dataset from the Cancer Imaging Archive Matek et al., 2021, which contains 945 subjects, each labeled into one of 21 classes. We will transform these 21 classes into seven: 6 abnormal classes and 1 normal class. Our goal is for our model to robustly predict what type of abnormality a cell has, if any. Being able to accurately classify cells as cancerous or not with machine learning, can expedite cancer detecting, saving lives by beginning treatments earlier.|
| **2. Technical Approach & Innovation** | 5 | Clearly explains the AI/ML methodology and justifies major technical decisions. Demonstrates meaningful technical effort or innovation, such as implementing a nontrivial algorithm, adapting an existing method, or investigating a novel application. | Our project aims to automate this process by utilizing the ImageNet ResNet50 model’s embeddings paired with a kNN (He et al., 2015), training our own Convolutional Neural Network (CNN), and our own Vision Transformer (ViT) to identify cell abnormalities based on visual characteristics (Dosovitskiy et al., 2021). The input will be images of individual bone marrow cells and the output will be its predicted classification: Normal or abnormal, where abnormal cells will be further placed into 6 subcategories. Therefore, the models will perform a multi-classification task with c = 7. We will study how well different models are able to distinguish between the classes. Previous research by Matek et al., 2021, has shown that CNNs can accurately differentiate bone marrow cell morphologies. We plan to compare other multiple classification methods to see which performs best.|
| **3. Results & Evaluation** | 4 | Presents meaningful experimental results using appropriate metrics and visualizations. Demonstrates an understanding of model performance, including relevant comparisons or baselines where feasible. | |
| **4. Critical Thinking & Demonstrated Understanding** | 5 | Demonstrates a strong understanding of the underlying AI/ML concepts and implementation. Explains why the approach works (or fails), acknowledges limitations, and identifies reasonable improvements. Goes beyond simply presenting code and model outputs—i.e., presents a good discussion. | |
| **5. Presentation Quality & Organization** | 2 | Presentation is clear, well-organized, visually effective, and delivered within the 5-minute limit. Prioritizes essential information and the big picture vs. implementation details. | |
| **Total** | **20** | | |
