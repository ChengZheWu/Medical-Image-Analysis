# Final Project: Lung Nodule Classification

In this project, I referred to this paper [Lung Nodule Classification using Deep Local-Global Networks](https://arxiv.org/abs/1904.10126) and the official implementation from this [GitHub Repository](https://github.com/mundher/local-global). Then I improved the performance by building a new model.

## Dataset
I used the same data as the above paper and it had already been preprocessed. The data was extracted from LIDC-IDRI whcih was a public dataset for lung cancer analysis. 

## Model Architecture
I fused the concepts of densenet and self-attention which was called non-local block in the above paper. I called the model DenseAttentionNet(DANet).  
![DANet](https://github.com/ChengZheWu/Medical-Image-Analysis/blob/main/images/DANet.png)  
Fig.1 DANet, where Attention denoted self-attention , GAP denotes global average pooling, and FC Layer denotes fully connected layer.  
  
![Dense Block](https://github.com/ChengZheWu/Medical-Image-Analysis/blob/main/images/dense%20block.png)  
Fig.2 Dense Block.  
  
![Self-Attention](https://github.com/ChengZheWu/Medical-Image-Analysis/blob/main/images/self-attention.png)  
Fig.3 Self-Attention.

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

## Enviroment
 - Operation Ststem: CentOS7  
 - GPU: Nvidia Tesla V100 * 1

## Reference
[Lung Nodule Classification using Deep Local-Global Networks](https://arxiv.org/abs/1904.10126) [[GitHub]](https://github.com/mundher/local-global)  
[Self-Attention Generative Adversarial Networks](https://arxiv.org/abs/1805.08318) [[GitHub]](https://github.com/heykeetae/Self-Attention-GAN)  
[Densely Connected Convolutional Networks](https://arxiv.org/abs/1608.06993)  

