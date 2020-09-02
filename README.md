<<<<<<< HEAD
## Domain-Adaptation Project
An implementation of the article ["Domain-Adversarial Training of Neural Networks" (2016)](https://arxiv.org/pdf/1505.07818.pdf) by Ganin et al. This article deals with the ["Domain Adptation"](https://en.wikipedia.org/wiki/Domain_adaptation) problem which happens often in machine learning.

The goal of domain adaptation is to transfer the knowledge of a model to a different but related data distribution. The model is trained on a source dataset and applied to a target dataset (usually unlabeled).

The suggested neural network in the article consists of two classifiers at the training stage: label classification and a domain classification. Each classifier has its own loss (classification loss and the domain confusion loss accordingly). 

That way we achieve in one go two tasks for the network:
1. Classifying the data by supervised labels and 
2. Becoming invariant to the domain differences and “forgetting” the features which represent the difference between the domains

[[IMAGE OF THE NETWORK]]


The proposed architecture includes a deep feature extractor (green) and a deep
label predictor (blue), which together form a standard feed-forward architecture.
Unsupervised domain adaptation is achieved by adding a domain classifier (red)
connected to the feature extractor via a gradient reversal layer that multiplies
the gradient by a certain negative constant during the backpropagation-based
training. Other than that change, the training proceeds standardly and minimizes the label
prediction loss (for source examples) and the domain classification loss (for all
samples). 
Gradient reversal ensures that the feature distributions over the two
domains are made similar (as indistinguishable as possible for the domain classifier), thus resulting in the domain-invariant features.

We implemented this architecture and tested it as a variation of the classic task of recognition of seven emotional states (neutral, joy, sadness, surprise, anger, fear, disgust) based on facial expressions. The two domains are “Women” and “Men” and the domain-adaptation challenge is training the network on one sex and applying it on the other. 
=======
# classificationProject
>>>>>>> add README
