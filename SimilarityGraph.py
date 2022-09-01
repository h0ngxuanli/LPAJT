#####################################################################################################################
#                                        consturct affinity matrix for LSC                                         #
#####################################################################################################################

import numpy as np
from sklearn.metrics import pairwise_distances
from numpy.matlib import repmat
import scipy.sparse as sps
import matplotlib.pyplot as plt

def graph(fea,options):
    """
    	Usage:
    	W = graph(fea,options)
    
    	fea: Rows of vectors of data points. Each row is x_i
      options: Struct value in Matlab. The fields in options that can be set:
              Metric -  Choices are:
                  'Euclidean' - Will use the Euclidean distance of two data 
                                points to evaluate the "closeness" between 
                                them. [Default One]
                  'Cosine'    - Will use the cosine value of two vectors
                                to evaluate the "closeness" between them.
                                A popular similarity measure used in
                                Information Retrieval.
                     
              NeighborMode -  Indicates how to construct the graph. Choices
                              are: [Default 'KNN']
                   'KNN'            -  k = 0
                                          Complete graph
                                       k > 0
                                         Put an edge between two nodes if and
                                         only if they are among the k nearst
                                         neighbors of each other. You are
                                         required to provide the parameter k in
                                         the options. Default k=5.
                  'Supervised'      -  k = 0
                                          Put an edge between two nodes if and
                                          only if they belong to same class. 
                                       k > 0
                                          Put an edge between two nodes if
                                          they belong to same class and they
                                          are among the k nearst neighbors of
                                          each other. 
                                       Default: k=0
                                      You are required to provide the label
                                      information gnd in the options.
                                                 
              WeightMode   -  Indicates how to assign weights for each edge
                              in the graph. Choices are:
                  'Binary'       - 0-1 weighting. Every edge receiveds weight
                                   of 1. [Default One]
                  'HeatKernel'   - If nodes i and j are connected, put weight
                                   W_ij = exp(-norm(x_i - x_j)/2t^2). This
                                   weight mode can only be used under
                                   'Euclidean' metric and you are required to
                                   provide the parameter t.
                  'Cosine'       - If nodes i and j are connected, put weight
                                   cosine(x_i,x_j). Can only be used under
                                   'Cosine' metric.
                  
               k         -   The parameter needed under 'KNN' NeighborMode.
                             Default will be 5.
               gnd       -   The parameter needed under 'Supervised'
                             NeighborMode.  Colunm vector of the label
                             information for each data point.
               bLDA      -   0 or 1. Only effective under 'Supervised'
                             NeighborMode. If 1, the graph will be constructed
                             to make LPP exactly same as LDA. Default will be
                             0. 
               t         -   The parameter needed under 'HeatKernel'
                             WeightMode. Default will be 1
            bNormalized  -   0 or 1. Only effective under 'Cosine' metric.
                             Indicates whether the fea are already be
                             normalized to 1. Default will be 0
         bSelfConnected  -   0 or 1. Indicates whether W(i,i) == 1. Default 1
                             if 'Supervised' NeighborMode & bLDA == 1,
                             bSelfConnected will always be 1. Default 1.
    """
    
    #######################################
    
    if "Metric" not in dir(options):
        options.Metric = "Cosine"
    
    if 'bNormalized' not in dir(options):
        options.bNormalized = 0
    
    ########################################
    
    if 'NeighborMode' not in dir(options):
        options.NeighborMode = 'KNN'
        
    ########################################
    
    if options.NeighborMode == "KNN":
        if "k" not in dir(options):
            options.k = 5
    elif options.NeighborMode == 'Supervised':
        if "bLDA" not in dir(options):
            options.bLDA = 0
        if options.bLDA:
            options.bSelfConnected = 1
        if "k" not in dir(options):
            options.k = 0
        
        if "gnd" not in dir(options):
            raise ValueError('Label(gnd) should be provided under ''Supervised'' NeighborMode!')
    
        if fea.shape[0] != len(options.gnd):
            raise ValueError('gnd doesn''t match with fea!')
    else:
        raise ValueError('NeighborMode does not exist!')
    
    
    ######################################################
    
    if 'WeightMode' not in dir(options):
        options.WeightMode = 'Binary'
    
    
    bBinary = 0
    if options.WeightMode == "Binary":
        bBinary = 1
    elif options.WeightMode == "HeatKernel":
        if options.Metric != 'Euclidean':
            raise ValueError(" 'HeatKernel' WeightMode should be used under 'Euclidean' Metric!")
        #options.Metric = 'Euclidean'
        if "t" not in dir(options):
            options.t = 1
    elif options.WeightMode == "Cosine":
        if options.Metric != 'Cosine':
            raise ValueError(" 'Cosine' WeightMode should be used under 'Cosine' Metric!")
        if 'bNormalized' not in dir(options):
            options.bNormalized = 0
    else:
        raise ValueError('WeightMode does not exist!')
    
    ##########################################################
    
    
    if 'bSelfConnected' not in dir(options):
        options.bSelfConnected = 1
    
    
    if "gnd" in dir(options):
        nSmp = len(options.gnd)
    else:
        nSmp = fea.shape[0]
    
    
    maxM = 62500000
    
    
    BlockSize = int(np.floor(maxM/(nSmp*3)))
    
    
    if options.NeighborMode == 'Supervised':
        Label = np.unique(options.gnd)
        nLabel  = len(Label)
        if options.bLDA:
            G = np.zeros((nSmp,nSmp))
            for idx in range(nLabel):
                classIdx = np.where(options.gnd==Label(idx))[0]
                G[np.ix_(classIdx, classIdx)] = 1/len(classIdx)
            W = G#sps.csr_matrix(G)
        if options.WeightMode == "Binary":
            if options.k >0:
                G = np.zeros((nSmp*(options.k+1),3))
                idNow = 0
                for i in range(nLabel):
                    classIdx = np.where(options.gnd == Label[i])[0]
                    D = pairwise_distances(fea[classIdx,:])**2
                    idx = np.argsort(D, axis=1)
                    idx = idx[:,:options.k+1]
                    nSmpClass = len(classIdx)*(options.k+1)
                    G[idNow:nSmpClass+idNow,0] = repmat(classIdx.reshape(1,-1).T,options.k+1,1).reshape(-1)#
                    G[idNow:nSmpClass+idNow,1] = classIdx[idx.flatten('F').reshape(1,-1).T].reshape(-1)
                    G[idNow:nSmpClass+idNow,2] = 1
                G = sps.csr_matrix((G[:,2], (G[:,0], G[:,1])), shape=(nSmp,nSmp),dtype= np.int32)
                G = G.todense()
                G = np.maximum(G,G.T)
            else:
                G = np.zeros((nSmp,nSmp))
                for i in range(nLabel):
                    classIdx = np.where(options.gnd == Label[i])[0]
                    G[np.ix_(classIdx, classIdx)] = 1
            
            if not options.bSelfConnected:
                for i in range(G.shape[0]):
                    G[i,i] = 0
            #W = sps.csr_matrix(np.maximum(G,G.T))
            W = G#np.maximum(G,G.T)
        if options.WeightMode == "HeatKernel":
            if options.k >0:
                G = np.zeros((nSmp*(options.k+1),3))
                idNow = 0
                for i in range(nLabel):
                    classIdx = np.where(options.gnd == Label[i])[0]
                    D = pairwise_distances(fea[classIdx,:])**2
                    idx = np.argsort(D, axis=1)
                    idx = idx[:,:options.k+1]
                    dump = np.sort(D)
                    dump = dump[:,:options.k+1]
                    dump = np.exp(-dump/(2*options.t**2))
                    
                    nSmpClass = len(classIdx)*(options.k+1)
                    G[idNow:nSmpClass+idNow,0] = repmat(classIdx.reshape(1,-1).T,options.k+1,1).reshape(-1)#
                    G[idNow:nSmpClass+idNow,1] = classIdx[idx.flatten('F').reshape(1,-1).T].reshape(-1)
                    G[idNow:nSmpClass+idNow,2] = dump.flatten('F')
                    idNow = idNow + nSmpClass
                G = sps.csr_matrix((G[:,2], (G[:,0], G[:,1])), shape=(nSmp,nSmp),dtype= np.int32)
                G = G.todense()
            else:
                G = np.zeros((nSmp,nSmp))
                for i in range(nLabel):
                    classIdx = np.where(options.gnd == Label[i])[0]
                    
                    D = pairwise_distances(fea[classIdx,:])**2
                    D = np.exp(-D/(2*options.t**2))
                    G[np.ix_(classIdx, classIdx)] = D
            if not options.bSelfConnected:
                for i in range(G.shape[0]):
                    G[i,i] = 0
            #W = sps.csr_matrix(np.maximum(G,G.T))
            W = np.maximum(G,G.T)
        if options.WeightMode == "Cosine":
            if not options.bNormalized:
                nSmp,nFea = fea.shape
                feaNorm = np.sum(fea**2,0)**0.5
                for i in range(nSmp):
                    fea[i,:] = fea[i,:] / max(1e-12,feaNorm[i])
            if options.k >0:
                G = np.zeros((nSmp*(options.k+1),3))
                idNow = 0
                for i in range(nLabel):
                    classIdx = np.where(options.gnd == Label[i])[0]
                    D = fea[classIdx,:].dot(fea[classIdx,:].T)
                    idx = np.argsort(-D, axis=1)
                    idx = idx[:,:options.k+1]
                    dump = np.sort(-D)
                    dump = -dump[:,:options.k+1]
                    nSmpClass = len(classIdx)*(options.k+1)
                    G[idNow:nSmpClass+idNow,0] = repmat(classIdx.reshape(1,-1).T,options.k+1,1).reshape(-1)#
                    G[idNow:nSmpClass+idNow,1] = classIdx[idx.flatten('F').reshape(1,-1).T].reshape(-1)
                    G[idNow:nSmpClass+idNow,2] = dump.flatten('F')
                    idNow = idNow + nSmpClass
                G = sps.csr_matrix((G[:,2], (G[:,0], G[:,1])), shape=(nSmp,nSmp),dtype= np.int32)
                G = G.todense()
            else:
                G = np.zeros((nSmp,nSmp))
                for i in range(nLabel):
                    classIdx = np.where(options.gnd == Label[i])[0]
                    D = fea[classIdx,:].dot(fea[classIdx,:].T)
                    G[np.ix_(classIdx, classIdx)] = D
                    
            if not options.bSelfConnected:
                for i in range(G.shape[0]):
                    G[i,i] = 0
            #W = sps.csr_matrix(np.maximum(G,G.T))
            W = np.maximum(G,G.T)
        else:
            raise ValueError('WeightMode does not exist!')
        return W
                  
    
                
    if options.NeighborMode == 'KNN' and options.k > 0:
        if options.Metric == "Euclidean":
            G = np.zeros((nSmp*(options.k+1),3))

            for i in range(int(np.ceil(nSmp/BlockSize))):
                if i == int(np.ceil(nSmp/BlockSize))-1:
                    smpIdx = np.arange(i*BlockSize,nSmp)
                    
                    D = pairwise_distances(fea[smpIdx,:],fea)**2
                    idx = np.argsort(D, axis=1)
                    idx = idx[:,:options.k+1]
                    dump = np.sort(D)
                    dump = dump[:,:options.k+1]
                    if not bBinary:
                        dump = np.exp(-dump/(2*options.t**2))

                    G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),0] = repmat(smpIdx.reshape(1,-1).T,options.k+1,1).reshape(-1)#
                    G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),1] = idx.flatten('F').reshape(1,-1)
        
                    if not bBinary:
                        G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),2] = dump.flatten('F')
                    else:
                        G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),2] = 1
                else:
                    smpIdx = np.arange(i*BlockSize,(i+1)*BlockSize)
                    D = pairwise_distances(fea[smpIdx,:],fea)**2
                    idx = np.argsort(D, axis=1)
                    idx = idx[:,:options.k+1]
                    dump = np.sort(D)
                    dump = dump[:,:options.k+1]
                    if not bBinary:
                        dump = np.exp(-dump/(2*options.t**2))
             
                    G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),0] = repmat(smpIdx.reshape(1,-1).T,options.k+1,1).reshape(-1)#
                    G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),1] = idx.flatten('F').reshape(1,-1)
        
                    if not bBinary:
                        G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),2] = dump.flatten('F')
                    else:
                        G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),2] = 1  
            # for i in G[:,0][:10]:
            #     print(i)
            # np.set_printoptions(threshold=np.inf)
            # print(np.sum(G[:,1])+2802)

            W = sps.csr_matrix((G[:,2], (G[:,0], G[:,1])), shape=(nSmp,nSmp))#,dtype = np.int32)
            W = W.todense()
            #print(W[0,:])
        else:

            if not options.bNormalized:
                nSmp,nFea = fea.shape
                feaNorm = np.sum(fea**2,1)**0.5
                #print(feaNorm.shape)
                for i in range(nSmp):
                    fea[i,:] = fea[i,:] / max(1e-12,feaNorm[i])
    
            G = np.zeros((nSmp*(options.k+1),3))
            for i in range(int(np.ceil(nSmp/BlockSize))):
                if i == int(np.ceil(nSmp/BlockSize))-1:
                    smpIdx = np.arange(i*BlockSize,nSmp)
                    D = fea[smpIdx,:].dot(fea[smpIdx,:].T)
                    idx = np.argsort(-D, axis=1)
                    idx = idx[:,:options.k+1]
                    dump = np.sort(-D)
                    dump = -dump[:,:options.k+1]
                    
             
                    G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),0] = repmat(smpIdx.reshape(1,-1).T,options.k+1,1).reshape(-1)#
                    G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),1] = idx.flatten('F').reshape(1,-1)
                    G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),2] = dump.flatten('F')
                else:
                    smpIdx = np.arange(i*BlockSize+1,(i+1)*BlockSize+1)
                    D = fea[smpIdx,:].dot(fea[smpIdx,:].T)
                    idx = np.argsort(-D, axis=1)
                    idx = idx[:,:options.k+1]
                    dump = np.sort(-D)
                    dump = -dump[:,:options.k+1]
    
                    G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),0] = repmat(smpIdx.reshape(1,-1).T,options.k+1,1).reshape(-1)#
                    G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),1] = idx.flatten('F').reshape(1,-1)
                    G[i*BlockSize*(options.k+1) : nSmp*(options.k+1),2] = dump.flatten('F')
    
            #W = sps.csr_matrix(np.maximum(G,G.T))
            W = sps.csr_matrix((G[:,2], (G[:,0], G[:,1])), shape=(nSmp,nSmp))
            W = W.todense()
        if options.WeightMode == 'Binary':
            #W = W.todense()
            W[np.where(W>0)] = 1
        # if 'bSemiSupervised' in dir(options) and options.bSemiSupervised:
        #     tmpgnd = options.gnd[options.SemiSplit]
        #     Label = np.unique(tmpgnd)
        #     nLabel = len(Label)
        #     G = np.zeros()
        # if isfield(options,'bSemiSupervised') && options.bSemiSupervised
        #     tmpgnd = options.gnd(options.semiSplit);
            
        #     Label = unique(tmpgnd);
        #     nLabel = length(Label);
        #     G = zeros(sum(options.semiSplit),sum(options.semiSplit));
        #     for idx=1:nLabel
        #         classIdx = tmpgnd==Label(idx);
        #         G(classIdx,classIdx) = 1;
        #     end
        #     Wsup = sparse(G);
        #     if ~isfield(options,'SameCategoryWeight')
        #         options.SameCategoryWeight = 1;
        #     end
        #     W(options.semiSplit,options.semiSplit) = (Wsup>0)*options.SameCategoryWeight;
        # end    
        if not options.bSelfConnected:
            for i in range(W.shape[0]):
                W[i,i] = 0
        W = np.maximum(W,W.T)
        return W            
    if options.Metric == "Euclidean":

        W = pairwise_distances(fea,fea)**2
        W = np.exp(-W/(2*options.t**2))
    else:
        if not options.bNormalized:
            nSmp,nFea = fea.shape
            feaNorm = np.sum(fea**2,0)**0.5
            for i in range(nSmp):
                fea[i,:] = fea[i,:] / max(1e-12,feaNorm[i])
        W = fea.dot(fea.T)
    
    if not options.bSelfConnected:
        for i in range(W.shape[0]):
            W[i,i] = 0
    
    W = np.maximum(W,W.T)
    
    return W
            




