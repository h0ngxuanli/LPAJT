#####################################################################################################################
#                                          LPAJT / LSC / TCA / JDA / SVM / KNN                                      #
#####################################################################################################################


import numpy as np
import scipy
import sklearn
from sklearn import svm
from utils import get_metric, one_hot, kernel, compute_MMD, np_unranked_unique
from sklearn.neighbors import KNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from SimilarityGraph import graph


def LPAJT_TF(Xs,ys,Xt,yt,beta,gamma,lamda,dim,selected_class, kernel_type, soft, seed, T = 20):

    list_acc = []
    list_metric = []
    list_mmd = []
    list_otr = []

    Ys = one_hot(ys)

    X = np.hstack((Xs.T, Xt.T))
    X /= np.linalg.norm(X, axis=0)
    m, n = X.shape
    ns, nt = len(Xs), len(Xt)
    H = np.eye(n) - 1 / n * np.ones((n, n))
    
    # SVM output probabilistic label
    SVM = svm.SVC(C = soft,kernel = 'linear',random_state=seed)
    clf = CalibratedClassifierCV(SVM) 
    clf.fit(Xs, ys)
    Pt = clf.predict_proba(Xt) 

    # kernel function
    K = kernel(kernel_type, X, None, gamma=1)
    ori_mmd = compute_MMD(K,ys,yt,beta)
    list_mmd.append(ori_mmd)
    print('Original MMD {:.4f}'.format(ori_mmd))
    
    for t in range(T):

        ########## get W #############
        pt = Pt.sum(0)
        W = np.zeros((ns,1))
        C = len(np.unique(ys))
        pt/= pt.sum()
   
        for c in range(C):
            idx = ys == c
            W[idx] = pt[c]
        ########## get Mg ############

        e0 = np.vstack((1 / np.sum(W) * W, -1 / nt * np.ones((nt, 1))))
        Mg = e0.dot(e0.T)
        
        ########## get Ml ############
        Yst = Ys.dot(np.linalg.inv(Ys.T.dot(Ys))).dot(Pt.T)
        Ml = np.block([ 
                        [Yst.dot(Yst.T),            -Yst],
                        [-Yst.T        , np.identity(nt)]
                                                            ])
        ########## get Mp ############
        
        Y = np.vstack((Ys,Pt))
        Yc = Y.dot(np.linalg.inv(Y.T.dot(Y))).dot(Y.T)
        Mp = (np.identity(len(Yc)) - Yc).dot((np.identity(len(Yc)) - Yc).T)

        ########## get latent representation ############


        M =  Mg + beta*Ml + gamma*Mp
        M /= np.linalg.norm(M, 'fro')

        K = kernel(kernel_type, X, None, gamma=1)
        n_eye = m if kernel_type == 'primal' else n  

        # get optimal projection matrix with eigenvalue decomposition
        a, b = np.linalg.multi_dot(
            [K, M, K.T]) + lamda * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)

        # optimal projection matrix
        A = np.real(V[:, ind[:dim]])

        #latent representation
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)

        MMD = compute_MMD(Z,ys,yt,beta)

        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T   


        SVM = svm.SVC(C = soft ,kernel = kernel_type,random_state=seed)
        clf = CalibratedClassifierCV(SVM) 
        clf.fit(Xs_new, ys)
        Pt = clf.predict_proba(Xt_new)
        Pt = clf.predict_proba(Xt_new)
        pred = np.array(np.argmax(Pt, axis=1)).ravel()
        Y_tar_pseudo = pred.ravel()


        
        ACC = sklearn.metrics.accuracy_score(yt, Y_tar_pseudo)
        
        if len(selected_class) == 1:
            AUC, ACC, SEN, SPE, BAC, PPV, NPV, OTR = get_metric(yt, Y_tar_pseudo,selected_class)
            list_metric.append([AUC, ACC, SEN, SPE, BAC, PPV, NPV, OTR])
            list_otr.append(OTR)
            list_acc.append(AUC)
            print('LPAJT Iteration [{}/{}]: AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, OTR: {:.4f}, MMD: {:.4f}'.format(t+1, T, AUC, ACC, SEN, OTR, MMD))
        else:
            list_acc.append(ACC)
            print('LPAJT Iteration [{}/{}]: ACC: {:.4f}, MMD: {:.4f}'.format(t+1, T, ACC, MMD))
        
        list_mmd.append(MMD)            
    
    list_acc = np.array(list_acc)
    list_acc[np.isnan(list_acc)] = 0
    idx = np.where(list_acc==np.max(list_acc))[0][0]
 
    
    if len(selected_class) == 1:
        max_acc = list_metric[idx]
    else:
        max_acc = list_acc[idx]
    return list_acc, max_acc, list_otr, list_mmd, [Xs, ys, Xt, yt, Xs_new, Xt_new], list(np_unranked_unique(W))

def LSC_TF(Xs,ys,Xt,yt,lamb,delta,k,threshold,dim,selected_class,kernel_type, soft, seed, T = 10):
    class parameters:
       def __init__(self):
           pass
    X = np.hstack((Xs.T, Xt.T))
    X /= np.linalg.norm(X, axis=0)
    m, n = X.shape
    ns, nt = len(Xs), len(Xt)
    e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
    C = len(np.unique(ys))
    H = np.eye(n) - 1 / n * np.ones((n, n))


    list_acc = []
    list_metric = []
    list_otr = []
    M = 0
    Y_tar_pseudo = None
    
    # conditonal and marginal MMD
    for t in range(T):
        N = 0
        M0 = e * e.T * C
        if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
            for c in range(C):
                e = np.zeros((n, 1))
                tt = ys == c
                e[np.where(tt == True)] = 1 / len(ys[np.where(ys == c)])
                yy = Y_tar_pseudo == c
                ind = np.where(yy == True)
                inds = [item + ns for item in ind]
                if len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)]) ==0:
                    e[tuple(inds)] = 0
                else:
                    e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                N = N + np.dot(e, e.T)
                
        M = M0 + N
        M = M / np.linalg.norm(M, 'fro')
        K = kernel(kernel_type, X, None, gamma= 1)
        n_eye = m if kernel_type == 'primal' else n
        a, b = np.linalg.multi_dot([K, M, K.T]) + lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = np.real(V[:, ind[:dim]])
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

        SVM = svm.SVC(C = soft ,kernel = 'linear',random_state=seed)
        clf = CalibratedClassifierCV(SVM) 
        clf.fit(Xs_new,ys)   
        
        # estimating the class posterior probability of the transformed target-domain data Zt .
        prob = clf.predict_proba(Xt_new)
        for i in range(len(prob)):
            row = prob[i,:]
            if (row <= threshold).all():
                row[row != row.max()] = -1
                row[row == row.max()] = 2*row.max()-1

            else:
                row[row <= threshold] = -1
                row[row > threshold] = 2*row[row > threshold] - 1    
            prob[i,:] = row    
            
        Y0 = prob

        # label propagation
        options = parameters()
        options.Metric = 'Euclidean'
        options.NeighborMode = 'KNN'
        options.WeightMode = 'HeatKernel'
        options.k = k
        options.t = delta
        E =  graph(Xt_new,options)
        E =  np.array(E)
        
        #each target-domain instance observes the structural information
        Dw = np.diag(np.sqrt(1/np.sum(E,0)))
        
        S = Dw.dot(E).dot(Dw)
        
        alpha = 0
        
        # propagating the label information in the target site
        Pt = (np.eye(nt)-alpha).dot(np.linalg.inv(np.eye(nt)-alpha*S)).dot(Y0)

        pred = np.array(np.argmax(Pt, axis=1)).ravel()

        Y_tar_pseudo = pred.ravel()

        ACC = sklearn.metrics.accuracy_score(yt,Y_tar_pseudo)      
        if len(selected_class) == 1:
            AUC, ACC, SEN, SPE, BAC, PPV, NPV, OTR = get_metric(yt, Y_tar_pseudo, selected_class)
            list_metric.append([AUC, ACC, SEN, SPE, BAC, PPV, NPV, OTR])
            list_otr.append(OTR)
            list_acc.append(AUC)
            print('LSC Iteration [{}/{}]: AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, OTR: {:.4f}'.format(t + 1, T, AUC, ACC, SEN, OTR))
        else:
            list_acc.append(ACC)
            print('LSC Iteration [{}/{}]: ACC: {:.4f}'.format(t + 1, T, ACC))
    list_acc = np.array(list_acc)
    list_acc[np.isnan(list_acc)] = 0        
    idx = np.where(list_acc == np.max(list_acc))[0][0]
    if len(selected_class) == 1:
        max_acc = list_metric[idx]
    else:
        max_acc = list_acc[idx]
    return list_acc, max_acc, list_otr, [Xs, ys, Xt, yt, Xs_new, Xt_new]



def TCA_TF(Xs, ys, Xt, yt, lamb, dim, selected_class, kernel_type, soft):

    X = np.hstack((Xs.T, Xt.T))
    X /= np.linalg.norm(X, axis=0)
    m, n = X.shape
    ns, nt = len(Xs), len(Xt)
    e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
    M = e * e.T 
    M = M / np.linalg.norm(M, 'fro')
    H = np.eye(n) - 1 / n * np.ones((n, n))

    K = kernel(kernel_type, X, None, gamma=1)
    n_eye = m if kernel_type == 'primal' else n        

    a, b = np.linalg.multi_dot([K, M, K.T]) + lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
    w, V = scipy.linalg.eig(a, b)
    ind = np.argsort(w)
    A = np.real(V[:, ind[:dim]])
    Z = np.dot(A.T, K)
    Z /= np.linalg.norm(Z, axis=0)
    Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T        
    clf = svm.SVC(C = soft, kernel  = kernel_type)
    clf.fit(Xs_new, ys.ravel())
    Y_tar_pseudo = clf.predict(Xt_new)
    ACC = sklearn.metrics.accuracy_score(yt, Y_tar_pseudo)
    if len(selected_class) == 1:
        AUC, ACC, SEN, SPE, BAC, PPV, NPV, OTR = get_metric(yt, Y_tar_pseudo, selected_class)
        result = [AUC, ACC, SEN, SPE, BAC, PPV, NPV, OTR]

        print('TCA : AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, OTR: {:.4f}'.format(AUC, ACC, SEN, OTR))
    else:
        result = ACC
        print('TCA : ACC: {:.4f}'.format(ACC))
        
    return result, [Xs, ys, Xt, yt, Xs_new, Xt_new]

def JDA_TF(Xs,ys,Xt,yt,lamb,dim, selected_class,kernel_type, soft, seed, T = 20):

    list_acc = []
    list_metric = []
    list_otr = []


    X = np.hstack((Xs.T, Xt.T))
    X /= np.linalg.norm(X, axis=0)
    m, n = X.shape
    ns, nt = len(Xs), len(Xt)
    e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
    C = len(np.unique(ys))
    H = np.eye(n) - 1 / n * np.ones((n, n))

    M = 0
    Y_tar_pseudo = None
    for t in range(T):
        N = 0
        M0 = e * e.T * C
        if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
            for c in range(C):
                e = np.zeros((n, 1))
                tt = ys == c
                e[np.where(tt == True)] = 1 / len(ys[np.where(ys == c)])
                yy = Y_tar_pseudo == c
                ind = np.where(yy == True)
                inds = [item + ns for item in ind]
                if len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)]) ==0:
                    e[tuple(inds)] = 0
                else:
                    e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                N = N + np.dot(e, e.T)
        M = M0 + N
        M = M / np.linalg.norm(M, 'fro')
        K = kernel(kernel_type, X, None, gamma= 1)

        n_eye = m if kernel_type == 'primal' else n      
        a, b = np.linalg.multi_dot([K, M, K.T]) + lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = np.real(V[:, ind[:dim]])
        Z = np.dot(A.T, K)
        Z /= np.linalg.norm(Z, axis=0)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        clf = svm.SVC(C = soft,kernel  = kernel_type ,random_state= seed)
        clf.fit(Xs_new, ys.ravel())
        Y_tar_pseudo = clf.predict(Xt_new)
        ACC = sklearn.metrics.accuracy_score(yt, Y_tar_pseudo)
        
        if len(selected_class) == 1:
            AUC, ACC, SEN, SPE, BAC, PPV, NPV, OTR = get_metric(yt, Y_tar_pseudo,selected_class)
            list_metric.append([AUC, ACC, SEN, SPE, BAC, PPV, NPV, OTR])
            list_otr.append(OTR)
            list_acc.append(AUC)
            print('JDA Iteration [{}/{}]: AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, OTR: {:.4f}'.format(t+1, T, AUC, ACC, SEN, OTR))
        else:
            list_acc.append(ACC)
            print(
                'JDA Iteration [{}/{}]: ACC: {:.4f}'.format(t+1, T, ACC))
              
    
    list_acc = np.array(list_acc)
    list_acc[np.isnan(list_acc)] = 0
    idx = np.where(list_acc==np.max(list_acc))[0][0]
    
    if len(selected_class) == 1:
        max_acc = list_metric[idx]
    else:
        max_acc = list_acc[idx]
    return list_acc, max_acc, list_otr, [Xs, ys, Xt, yt, Xs_new, Xt_new]

def KNN_TF(Xs,ys,Xt,yt,k, selected_class):  
    clf = KNeighborsClassifier(n_neighbors= k)
    clf.fit(Xs,ys)
    Y_tar_pseudo = clf.predict(Xt)
    
    ACC = sklearn.metrics.accuracy_score(yt, Y_tar_pseudo)
    
    if len(selected_class) == 1:
        AUC, ACC, SEN, SPE, BAC, PPV, NPV, OTR = get_metric(yt, Y_tar_pseudo, selected_class)
        result = [AUC, ACC, SEN, SPE, BAC, PPV, NPV, OTR]
        print('KNN [ K--> {} ]: AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, OTR: {:.4f}'.format(k, AUC, ACC, SEN, OTR))
    else:
        result = ACC
        print('KNN [ K--> {} ]: ACC: {:.4f}'.format(k, ACC))
    return result


def SVM_TF(Xs,ys,Xt,yt, C, selected_class, kernel_type):   

    clf = svm.SVC(C = C,kernel  = kernel_type)
    clf.fit(Xs,ys)
    Y_tar_pseudo = clf.predict(Xt)

    ACC = sklearn.metrics.accuracy_score(yt, Y_tar_pseudo)
    
    if len(selected_class) == 1:
        AUC, ACC, SEN, SPE, BAC, PPV, NPV, OTR = get_metric(yt, Y_tar_pseudo, selected_class)
        result = [AUC, ACC, SEN, SPE, BAC, PPV, NPV, OTR]
        print('SVM [ C--> {} ]: AUC: {:.4f}, ACC: {:.4f}, SEN: {:.4f}, OTR: {:.4f}'.format(C, AUC, ACC, SEN, OTR))
    else:
        result = ACC
        print('SVM [ C--> {} ]: ACC: {:.4f}'.format(C, ACC))
    return result






