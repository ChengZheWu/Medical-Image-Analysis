# Final Project: Lung Nodule Classification

In this project, I referred to this paper [Lung Nodule Classification using Deep Local-Global Networks](https://arxiv.org/abs/1904.10126) and the official implementation from this [GitHub](https://github.com/mundher/local-global). Then I improved the performance by building a new model.

## Dataset
I used the same data as the above paper and it had already been preprocessed. The data was extracted from LIDC-IDRI whcih was a public dataset for lung cancer analysis. 

## Model Architecture

## Result

Model         | AUC       | Accuracy  | Precision | Recall    | Remark
:-------------|----------:|----------:|----------:|----------:|:----------
DANet         |95.17      |88.81      |90.39      |85.71      |My implementation           
Local-Global  |94.75      |87.16      |88.17      |84.48      |           
BasicResNet   |90.00      |79.51      |73.87      |88.42      |           
AllAtn        |94.45      |87.16      |87.41      |85.47      |           
AllAtnBig     |94.89      |87.40      |89.87      |83.00      | 
ResNet50      |93.33      |85.28      |85.39      |83.50      |Transfer Learning 
ResNet18      |92.75      |85.16      |87.84      |80.05      |Transfer Learning
DenseNet121   |92.71      |85.28      |87.27      |81.03      |Transfer Learning       
