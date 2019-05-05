# Title
Implementation of "Temporal Recurrent Networks for Online Action Detection" by Mingze Xu, Mingfei Gao, Yi-Ting Chen, Larry S. Davis and David J. Crandall. 
Link: https://arxiv.org/pdf/1811.07391.pdf

# Astract
Real time systems, especially surveillance, require identifying actions at the earliest. But most of the work on temporal action detection is formulated in offline workflow. To make the action detection online and at the earliest, not only the past evidence and present state are sufficient, but also the future anticipation is necessary. Under this assumption, this paper presents a novel framework called Temporal Recurrent Networks (TRNs) to model temporal context of a video frame by simultaneously performing online detection and anticipation of immediate future. Authors evaluated this approach on three online action detection datasets, HDD, TVSeries and THUMOS’14. The results show that TRN significantly outperforms the state-of-the-art.
3. Datasets being used to test/train, if at all: THUMOS’14 Validation Dataset for training set and the TestSet for testing.
4. Final deliverables (Plan A, Plan B, and optionally Plan C)
	Plan A:
We shall try to implement the idea described in the paper for online action detection and train the model on THUMOS’14 dataset. As described in paper, the training set of the dataset cannot be used for training TRNs as the training dataset consists of trimmed videos. So validation set which contains untrimmed videos of 20 classes will be used for training the model and test set of will be used for evaluation. This approach is also adopted by the authors.
1.	Implementation of TRN Cells described in paper
2.	Model trained on THUMOS’14 Dataset as described in paper
We will choose one of PyTorch or Keras frameworks for the development.
Plan B: 
We shall try to implement the same on TV Human Interaction Dataset, which contains only 4 action classes, in case of any problems/delays in downloading datasets.
5. An estimate of mid-term deliverable (by second week of March)
1.	Get the datasets ready
2.	Understand the RNNs and TRNs 
3.	Implement TRN module 


# Abstract
1. A brief abstract of your project including the problem statement and solution approach. If the project has cool visual results, you may provide one image or GIF of the results. See this for reference.

2. A list of code dependencies.

3. Detailed instructions for running the code, preferably, command instructions that may reproduce the declared results. If your code requires a model that can't be provided on GitHub, store it somewhere else and provide a download link.

4. Results: If numerical, mention them in tabular format. If visual, display. If you've done a great project, this is the area to show it!

5. Additional details, discussions, etc.

6. References.
