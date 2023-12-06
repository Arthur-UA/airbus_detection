# Semantic segmentation model
The idea of this project is to solve the airbus detection problem ([Kaggle link](https://www.kaggle.com/c/airbus-ship-detection/overview)) using the Semantic segmentation model, with U-NET architecture and tensorflow. The model will be evaluated using the Dice score metric.

## Dataset overview
During the EDA process it was found out that the dataset consists of jpg images with the same size - 768x768 and the bigger part of it has zero or just one ship.
Also it was figured out that most of the pictures are blue and green.

## My approach to solve this problem
To create a model, I used the U-NET architecture with less total number of filters because of the software limitations. As a result I got a model with the following counts of filters - Encoder (8 -> 16 -> 32 -> 64) - Bridge 128 - Decoder (64 -> 32 -> 16 -> 8) instead of original - Encoder (64 -> 128 -> 256 -> 512) - Bridge 1024 - Decoder (512 -> 256 -> 128 -> 64). The last layer was the 1x1 Convolution output of which would be the following shape (height, width, num_classes), (768, 768, 1) in this case. This reduction allowed to decrease the number of model parameters from 20+ millions to ~480k. Of course, preserving the primary structure would significantly improve the final results, but since the training took place on the CPU, this can significantly shorten the training time. The number of training images was also decreased to 3500. 
There also were some additional hyperparameters for a model like:
- Optimizer: Adam.
- Initial learning rate: 0.01
- Batch size: 2
- Epoch: 20
- Loss function: dice loss (custom)
- Cost function: dice score (custom)
- Gaussian noise: 0.1

## Results
After training, we end up with a dice score of about 0.61, which is pretty good, including the size of the used dataset, which was significantly decreased. Train/test loss plots showed that the training could be stopped after the first 5 epochs, where it got the minimum loss values. From the generated masks we can see that due to the little model size it can detect edges of ships pretty well, but it fails to correctly classify the interior part of the ship.

## Recommendations for further tuning and training
Here are my recommendations for some further training in order to deploy and use the model in the future:
- Transfer learning from CPU to GPU to have the ability to increase the number of model parameters and dataset size.
- As this model is more like baseline, it would be good to tune some of the hyperparameters like LR or try different optimizers, batch sizes, number of epochs.

## Instruction to run and test the model:
- Install all packages from the requirements.txt through pip
- Run the app.py file
- Open the localhost in any browser
- Use the proposed GUI for testing the existing images or download the new ones
