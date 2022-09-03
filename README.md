# Locality-Preserved Adaptive Joint Transfer (LPAJT) with Resting-State fMRI for Versatile Cross-Site Alzheimer’s Disease Diagnosis





## Overview

*"LPAJT minimizes the marginal and conditional distribution divergence of selected source samples and target samples, and enforces intra-class compactness to tackle the feature distortion problem caused by MMD."*



- ### Motivation

To boost diagnostic performance of models, local hospital expects to extend their small dataset by the rich-labeled dataset from large research institutions. However, this cross-site extension always suffers three major difficulties in real-world applications:

1) the inter-site heterogeneity will cause serious degradation of model performance, or even mismatch. 
2) only limited labeled data are available in the small dataset due to expensive labeling costs over medical data. 
3) the categories of subjects collected by the local hospital are usually a subset of those in the large research institutions. 

<img src="experiments/motivation.png" width=35% height=35%>

## Methodology

- ####  Landmark-based Global Alignment
  
<div align = 'center'><img width="507" alt="global" src="https://user-images.githubusercontent.com/68325219/188080105-f982b35a-2090-4b56-9f6f-612cb2b71d5a.png"></div>
<div align = "justify"> 
where $\mathbf{A}\in \mathbb{R}^{k \times d}$ is the projection matrix to be optimized, $\alpha=\left[ \alpha^{1}; \ldots; \alpha^{c} \right] \in \mathbb{R}^{n_{s}}, \alpha^{c} = \left[\alpha_{1}^{c}  ; \cdots ; \alpha_{n^c_s}^{c} \right] \in \mathbb{R}^{n^c_s}$ and $\alpha_{i}$ denotes the confidence of source sample $\mathbf{x}_{s}^{i}$ selected as the landmark.</div>


<div align = "justify"> 
To learn $\alpha$, we firstly apply SVM-based classifiers to estimate the class posterior probability of the projected target domain data, and the probability label matrix is denoted as $\mathbf{P}_{t}=\left[\mathbf{p}_t^{1}; \ldots; \mathbf{p}_t^{n_t}\right] \in \mathbb{R}^{ n_t \times C_s}$. $\mathbf{p}_t^i=\left[\mathbf{p}_t^{i(1)}, \ldots, \mathbf{p}_t ^ {i\left(C_s\right)}  \right] \in \mathbb{R}^{C_s}$ is the probabilistic label of
target samples $\mathbf{x}_t^i$ and $\mathbf{p}_t^{i(c)}$ denotes the probability that $\mathbf{x}_t^i$ belongs to the $c$-th class.  We enable overall estimation across all source classes to search the relevant classes by computing the class level weights $\overline{\mathbf{p}}_t$. Then, all the same-class samples in $\alpha^c$ are assigned with the class weight $\overline{\mathbf{p}}_t^c$.  </div>



- ####  Local Alignment via Weighted Semantic Loss
<div align = 'center'> <img width="397" alt="local" src="https://user-images.githubusercontent.com/68325219/188078468-8aebc8c2-3bb5-440d-b6f9-2bfd8907ee5d.png"> </div>

<div align = "justify"> 
where $\mu_s^c$ denotes the $c$-th class center of source domain, i.e., $\mu_s^c=\frac{1}{n_s^c} \sum_{\mathbf{x}_j \in \mathcal{D}_s^c} \mathbf{x}_j$, and $\mathcal{D}_s^c$ is source domain belonging to the $c$-th category. $n_s^c$ is the total number of source domain samples in the specific $c$-th category $\mathcal{D}_s^c$.</div>

- ####  Locality Preserving Loss
<div align = 'center'>  <img width="343" alt="preserve" src="https://user-images.githubusercontent.com/68325219/188078702-b2e72371-82f0-4bd8-9fb8-a612fe26c479.png"></div>

<div align = "justify"> 
where $\mu_{c}=\frac{1}{n^{c}} \sum_{\mathbf{x}_{j} \in \mathcal{D}_{s}^{c} \cup \mathcal{D}_{t}^{c}} \mathbf{x}_{j}$ is the domain-irrelevant class center.</div>

- ####   Reformulation and Optimization
<div align = "justify"> 
To simplify the optimization, we introduce domain alignment matrices $\mathbf{M_g}$, $\mathbf{M_l}$, $\mathbf{M_p}$ and rewrite the above equations into the following equivalent form:</div>

<div align = 'center'> <img width="255" alt="Mg" src="https://user-images.githubusercontent.com/68325219/188078741-704ededf-96f7-4ec8-af92-dc83c6e1fd38.png"> </div>
<div align = 'center'> <img width="445" alt="Ml" src="https://user-images.githubusercontent.com/68325219/188078757-fc0824b4-d8dd-49bb-8a8e-c604777eac3c.png"> </div>
<div align = 'center'> <img width="413" alt="Mp" src="https://user-images.githubusercontent.com/68325219/188078888-f071e129-7593-4341-8318-ff602b2d2ead.png">
 </div>
 
 <div align = "justify"> 
where the total sample matrix $\mathbf{X}= \left[\mathbf{X_s}, \mathbf{X_t}\right] \in \mathbb{R}^{d \times\left(n_s+n_t\right)}$ is defined for convenience, and $\mathbf{x}_i$ and $\mathbf{x}_j$ are the $i$-th and $j$-th columns of $\mathbf{X}$ respectively. $\operatorname{tr}(\cdot)$ is the trace of a matrix. </div>

The global alignment matrix $\mathbf{M_g}$, local alignment matrix $\mathbf{M_l}$, locality preserving matrix $\mathbf{M_p}$ are expressed as:
  
<div align = 'center'> <img width="464" alt="Mg" src="https://user-images.githubusercontent.com/68325219/188079453-acd78783-38f4-4e54-ada7-32be42faf4cd.png"></div>
<div align = 'center'> <img width="276" alt="Ml" src="https://user-images.githubusercontent.com/68325219/188079505-332514e9-59ea-45cf-8fe6-f1ddae5b126b.png"></div>
<div align = 'center'> <img width="272" alt="Mp" src="https://user-images.githubusercontent.com/68325219/188079564-92b65a9a-93da-4dcd-adde-218b1c202424.png"> </div>

 where $\mathbf{M_g},\mathbf{M_l},\mathbf{M_p} \in \mathbb{R}^{\left(n_s+n_t\right) \times\left(n_s+n_t\right)}$, $\mathbf{I}$ is an identity matrix, and $\mathbf{Y_{s t}} = \mathbf{Y_s} \left(\mathbf{Y_s}^{\top} \mathbf{Y_s}\right)^{-1} \mathbf{P_t}^{\top}, \mathbf{Y_c}=\mathbf{Y}\left(\mathbf{Y}^{\top} \mathbf{Y}\right)^{-1} \mathbf{Y}^{\top},$ where
$\mathbf{Y}=\left[\mathbf{Y_s} ; \mathbf{P_t}\right]$.

By integrating the above three domain alignment matrices, we have the overall objective function:

<div align = 'center'> <img width="494" alt="goal" src="https://user-images.githubusercontent.com/68325219/188079806-c8ff8962-89cf-4ad7-b2e4-26be3589d1d1.png"> </div>

Since it is a constrained convex optimization problem, we introduce Lagrangian multiplier and derive the Lagrange function as follows:
 <div align = 'center'> <img width="572" alt="Laplacian" src="https://user-images.githubusercontent.com/68325219/188079869-ba2c5ca2-f1dd-4759-a0eb-93b4ebfd16f8.png"> </div>
 
 where  $\mathbf{\Phi}$  is a diagonal matrix with Lagrangian multipliers defined as $\mathbf{\Phi}=\operatorname{diag}\left(\phi_1, \ldots, \phi_d\right)$ , and $\left(\phi_1, \cdots, \phi_d\right)$ are the $d$ smallest eigenvalues of the generalized eigendecomposition problem with $\frac{\partial \mathcal{L}(\mathbf{A}, \mathbf{\Phi})}{\partial \mathbf{A}}=0$:
 
 <div align = 'center'> <img width="527" alt="eigen" src="https://user-images.githubusercontent.com/68325219/188079877-fe238826-fbc0-4f62-8869-6eb367081e52.png"> </div>
 
 Finally, the optimal projection matrix $\mathbf{A}$ is composed of the corresponding $d$ eigenvectors derived by solving the above equation.
 
## Experiments


- ### Experiment Results
Performance of five different methods and module combinations under $\mathcal{C}_{t} =$ MCI+AD, NC+AD, NC+MCI scenarios. 

<img src="experiments/results.png" width=100% height=100%>

- ### Convergence Performance

We verify the convergence performance of LPAJT by inspecting the convergence performance of joint MMD distance, ACC, AUC and OTR(Outlier Transfer Rate).

<img src="experiments/convergence.png" width=100% height=100%>

- ### Landmark Confidence

We plot the class-level weights $\alpha$  for the source domain estimated from the unlabeled target samples.  For all scenarios, the shared classes across two sites get highest weights among all classes and the outlier classes weights is significantly smaller. 

<img src="experiments/selected.png" width=100% height=100%>

- ### Embedding Similarity

The similarity matrices of the original data and the subspace embeddings $\mathbf{Z} = \mathbf{A}^{\top} \mathbf{X}$ over seven transfer scenarios.

<img src="experiments/sim.png" width=100% height=100%>

- ### Parameter Sensitivity
To validate that LPAJT can achieve promising domain adaptation performance and encourage positive transfer effect in a wide range of parameter values,
we vary a parameter and the rest are fixed. 

<img src="experiments/para sensitivity.png" width=100% height=100%>
