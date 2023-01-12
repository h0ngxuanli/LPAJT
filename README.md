# Locality-Preserved Adaptive Joint Transfer (LPAJT) with Resting-State fMRI for Versatile Cross-Site Alzheimer’s Disease Diagnosis





## Overview

*"LPAJT minimizes the marginal and conditional distribution divergence of selected source samples and target samples, and enforces intra-class compactness to tackle the feature distortion problem caused by MMD."*



- ### Motivation

To boost diagnostic performance of models, local hospital expects to extend their small dataset by the rich-labeled dataset from large research institutions. However, this cross-site extension always suffers three major difficulties in real-world applications:

1) the inter-site heterogeneity will cause serious degradation of model performance, or even mismatch. 
2) only limited labeled data are available in the small dataset due to expensive labeling costs over medical data. 
3) the categories of subjects collected by the local hospital are usually a subset of those in the large research institutions. 

<img src="experiments/motivation.png" width=35% height=35%>
