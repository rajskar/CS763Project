# Title
Implementation of "Temporal Recurrent Networks for Online Action Detection" by Mingze Xu, Mingfei Gao, Yi-Ting Chen, Larry S. Davis and David J. Crandall. 

# Astract
Real time systems, especially surveillance, require identifying actions at the earliest. But most of the work on temporal action detection is formulated in offline workflow. To make the action detection online and at the earliest, not only the past evidence and present state are sufficient, but also the future anticipation is necessary. Under this assumption, this paper presents a novel framework called Temporal Recurrent Networks (TRNs) to model temporal context of a video frame by simultaneously performing online detection and anticipation of immediate future. I implemented TRN Cell Module and other required modules from scratch using pytorch framework and python language. I used Thumos'15 Dataset for training the model.

# Dataset
Dataset used for training the model is THUMOS'15 dataset. Since online detection needs untrimmed videos, only 20 out of 101 classes can be used, since the dataset provides untrimmed videos for only the following 20 classes.

7 BaseballPitch, 9 BasketballDunk, 12 Billiards, 21 CleanAndJerk, 22 CliffDiving, 23 CricketBowling, 24 CricketShot,
26 Diving, 31 FrisbeeCatch, 33 GolfSwing, 36 HammerThrow, 40 HighJump, 45 JavelinThrow, 51 LongJump, 68 PoleVault,
79 Shotput, 85 SoccerPenalty, 92 TennisSwing, 93 ThrowDiscus, 97 VolleyballSpiking. Number indicates the index of the class.

## Feature Extraction
I used two architectures as feature extractors, 1. FC6 layer of VGG16 architecture pretrained on imagenet and 2. Two stream Model, with resnet200 for rgb and BN_Inception for optical flow features. I have made a framework for extracting features from these networks. Also I have used Lucas-kanade algorithm for extracting optical flow features, which are inputs to BN_Inception module.

## Annotations
After this step, we have features of each image in all the videos. But the annotations provided were not at all straightforward. Annotations only provide the start time and end time of action in the long untrimmed video. Labelling each frame of every video  required for Classification task was in itself a challenging task and not discussed in the paper.

## Padding
Not all videos are of same duration. so frames sequences extracted were not of same length. To use them in Batches, I padded each batch with the zeros/ blank frames and assigned background class to the frame.

# Implementation

## Code Details
I implemented Temporal Recurrent Cell which is the basic building for the proposed Network. It consists of decoder RNN Cell, Future Gate and a SpatioTemporal RNN Cell from scratch. I made a network using this TRN Cell which takes input as sequences (video sequences). I implemented Loss function required for the TRN Network. Loss is weighted sum of losses obtained in anticipated action and current action and cross entropy is used as loss criterion. Optimiser used is Adam. I used pytorch framework for implementing the model in python environment.

![alt text](https://github.com/rajskar/CS763Project/blob/master/Block%20Diagram.png?raw=true "Block Diagram")

All the modules are in src folder.
TRNCell.py is the basic module, criterion.py implements the required loss functions. ModelParam.py lists the trainable parameters in the network.feature_extractor.py extracts all the features for rgb and optical features.

### Trainable Parameters
         TRNCell.DecRNNCell.weight_ih tensor(344064)
         TRNCell.DecRNNCell.weight_hh tensor(67108864)
         TRNCell.DecRNNCell.bias_ih tensor(16384)
         TRNCell.DecRNNCell.bias_hh tensor(16384)
         TRNCell.LinLayerR0.weight tensor(16777216)
         TRNCell.LinLayerR0.bias tensor(4096)
         TRNCell.LinLayerRp.weight tensor(86016)
         TRNCell.LinLayerRp.bias tensor(21)
         TRNCell.LinLayerRr.weight tensor(441)
         TRNCell.LinLayerRr.bias tensor(21)
         TRNCell.LinLayerS0.weight tensor(16777216)
         TRNCell.LinLayerS0.bias tensor(4096)
         TRNCell.StaRNNCell.weight_ih tensor(134217728)
         TRNCell.StaRNNCell.weight_hh tensor(67108864)
         TRNCell.StaRNNCell.bias_ih tensor(16384)
         TRNCell.StaRNNCell.bias_hh tensor(16384)
         TRNCell.LinLayerSn.weight tensor(86016)
         TRNCell.LinLayerSn.bias tensor(21)
Total trainable parameters tensor(302580216)

## Hardware
Hardware Platform consisted of a Geforce GTX 1060 Graphic card, intel i3 processor cpu with 4GB RAM. 

# Code dependencies
pytorch-gpu  
numpy  
and basic python libraries  

# Detailed instructions for running the code, 
## for default run
cd src
python main4.py 
## Args it can take are
D_inpseq, D_future, D_in, D_hid, D_out, BatchSize, alpha lr, path to model

This will run the model and save the final model as Model.pth in same folder
Download the (not very well) trained Model from   


# References
https://arxiv.org/pdf/1811.07391.pdf  
https://arxiv.org/abs/1707.04818: RED: Reinforced Encoder-Decoder Networks for Action Anticipation   
http://pytorch.org
