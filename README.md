# Understanding the Ranking Loss for Recommendation with Sparse User Feedback
This repository contains the open-source code for xxx

## Main Idea
Recent advancements suggest combining BCE loss with ranking loss has shown substantial performance improvement in many industrial deployments.
However, the efficacy of this combination loss is not fully understood.
We identify a novel challenge in CTR prediction with BCE loss: gradient vanishing in negative samples. 
We then propose a novel perspective on the effectiveness of ranking loss in CTR prediction, that it **leads to larger gradients on negative samples and hence mitigates their optimization issue, resulting in better classification ability**.

We conducted a comparative analysis of both ranking and classification performance, gradient examination, and loss landscape observation using the code from this repository.

# Getting Start
## Data Preparation
Please download the criteo_x1 data from official source into `/data/criteo` folder. Kindly be informed that the usage of this dataset is restricted to academic research purposes exclusively and it must not be utilized for any commercial or illegal activities.


