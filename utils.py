#####################################################################################################################
#                                                    utils                                                          #
#####################################################################################################################

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import sklearn
import numpy as np

def binary_metric(y_true,y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    ACC = (tp+tn)/(tp+tn+fp+fn)
    SEN = tp/(tp+fn)
    SPE = tn/(tn+fp)
    BAC = (SEN+SPE)/2
    PPV = tp/(tp+fp)
    NPV = tn/(tn+fn)
    fpr, tpr, _ = roc_curve(y_true,y_pred)
    AUC = auc(fpr, tpr)  
    
    return AUC,ACC,SEN,SPE,BAC,PPV,NPV  

# get resultes under different scenarios
def get_metric(y_true,y_pred,selected_class):
    Yt_bin = y_true.copy()
    pred_bin = y_pred.copy()
    
    
    idx = pred_bin == selected_class[0]
    OTR = len(pred_bin[idx])/len(pred_bin)
    
    idx = ~idx
    pred_bin = pred_bin[idx]
    Yt_bin = Yt_bin[idx]
    
    if len(np.unique(pred_bin)) == 0:
        return [0]*8
    
    elif len(np.unique(pred_bin)) == 1:
        
        pred_bin[pred_bin >= 1] = 1
        pred_bin[pred_bin <1 ] = 0
    else: 
        pred_bin[pred_bin == pred_bin.min()] = 0
        pred_bin[pred_bin == pred_bin.max()] = 1
        
    Yt_bin[Yt_bin == Yt_bin.min()] = 0
    Yt_bin[Yt_bin == Yt_bin.max()] = 1
        
    AUC, ACC, SEN,SPE,BAC,PPV,NPV = binary_metric(Yt_bin,pred_bin)

    return AUC, ACC, SEN, SPE, BAC, PPV, NPV, OTR



# one-hot encode source and target label
def one_hot(Ys):
    C = len(np.unique(Ys))
    Ys = np.eye(C)[Ys]
    return Ys

# drop certain class samples to create scenarios
def imbalance(X,Y,classes):
    if len(classes) == 0:
        return X,Y
    for i in classes:
        idx = np.where(Y!=i)[0]
        X = X[idx,:]
        Y = Y[idx]
    return X,Y

# kernel function
def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(
                np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(
                np.asarray(X1).T, None, gamma)
    return K

# compute MMD with true target label
def compute_MMD(X,ys,yt,beta):
    Ys = np.eye(3)[ys]
    Pt = np.eye(3)[yt]
    ns = len(ys)
    nt = len(yt)
    W = np.zeros((ns,1))
    C = np.unique(yt)
    for c in C:
        idx = ys == c
        W[idx] = 1
    
    # global divergence measurement
    e0 = np.vstack((1 / np.sum(W) * W, -1 / nt * np.ones((nt, 1))))
    Mg = e0.dot(e0.T)

    # local divergence measurement
    Yst = Ys.dot(np.linalg.inv(Ys.T.dot(Ys))).dot(Pt.T)
    Ml = np.block([ 
                    [Yst.dot(Yst.T),            -Yst],
                    [-Yst.T        , np.identity(nt)]
                                                            ])
    M = Mg+beta*Ml
    M /= np.linalg.norm(M, 'fro')
    MMD = np.trace((X).dot(M).dot(X.T))
    return MMD

# np.unique in orginial order
def np_unranked_unique(nparray):
    n_unique = len(np.unique(nparray))
    ranked_unique = np.zeros([n_unique])
    i = 0
    for x in nparray:
        if x not in ranked_unique:
            ranked_unique[i] = x
            i += 1
    return ranked_unique


