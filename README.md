# Project Goal 



The goal of this project is to implement and analyze the U-Net architecture for pixel-wise image segmentation using the Oxford-IIIT Pet Dataset.



Specifically, this project investigates how architectural modifications affect segmentation performance on a relatively small dataset.



### Objective



The primary objective is to segment cats and dogs from background pixels using a convolutional encoder–decoder architecture (U-Net).



In addition to training a baseline U-Net model, this project evaluates the impact of:



* Batch Normalization (nn.BatchNorm2d)



* Dropout (nn.Dropout2d)



* Scheduler



### Model Architecture



The model is based on the U-Net architecture:

* Encoder (Contracting path)
* Bottleneck
* Decoder (Expansive path)
* Skip connections



#### Results without Batchnorm, DropOut, and Scheduler:

U-Net without BatchNorm2d, DropOut, and Scheduler

Train Loss: 0.4771, Train IoU: 0.5724 

Val Loss: 0.4802, Val IoU: 0.5722





#### U-Net after adding BatchNowm2d to the DoubleConv section 

Train Loss: 0.1973, Train IoU: 0.8248 

Val Loss: 0.3366, Val IoU: 0.6800



#### Results after adding DropOut p = 0.3 

Train Loss: 0.1970, Train IoU: 0.8250 

Val Loss: 0.2715, Val IoU: 0.7378



#### Results after adding scheduler

scheduler = torch.optim.lr\_scheduler.ReduceLROnPlateau( optimizer, mode='max', # we observe IoU patience=3, # how many epoch we need to wait without improvement factor=0.5, # decrease lr by 2 verbose=True ) 

Train Loss: 0.1870, Train IoU: 0.8343 

Val Loss: 0.2212, Val IoU: 0.8056





#### Conclusion / Results



This project investigated the impact of architectural modifications and regularization on U-Net segmentation performance using the Oxford-IIIT Pet Dataset.



I trained a baseline U-Net, then sequentially added BatchNorm, Dropout, and a learning rate scheduler, comparing their effects on training and validation metrics.



#### Key Observations



Batch Normalization improved training stability slightly, but validation IoU was still limited (0.68), indicating some overfitting.



Dropout (p=0.3) helped reduce overfitting: validation IoU improved to 0.7378 while maintaining training IoU.



Learning Rate Scheduler further enhanced convergence and generalization, giving the best results:



Train IoU: 0.8343



Validation IoU: 0.8056



#### Insights



Architectural regularization (BatchNorm + Dropout) is crucial for small datasets to prevent overfitting.



Adaptive learning rates using ReduceLROnPlateau significantly improved validation performance.



The sequential experimentation approach clearly shows the effect of each modification.



#### Next Steps / Future Work



Apply data augmentations such as color jitter, rotation, and elastic transforms.



Test the pipeline on medical image segmentation datasets to evaluate generalization to real-world applications.



#### Project Structure

src/

&nbsp; model.py

&nbsp; dataset.py

&nbsp; train.py

configs/

README.md

requirements.txt







