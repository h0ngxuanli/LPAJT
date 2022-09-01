#####################################################################################################################
#                           train model, save optimal results, ablation study                                      #
#####################################################################################################################


import random
from scipy.io import loadmat
from models import LPAJT_TF, LSC_TF, JDA_TF, TCA_TF, SVM_TF, KNN_TF
import numpy as np


random.seed(0)

seed = 666
soft = 1 
kernel_type = "linear"
root_path = ""

from utils import imbalance

# drop out NC, MCI, AD to generate scenarios 
selected = [[0], [1], [2], [0, 1], [0, 2], [1, 2], []]

# two main transfer task on PL_S and PL_G sites
for domain in ["PL_S_PL_G","PL_G_PL_S"]: 
    for i in range(7):
        data = loadmat(root_path  + "data" + "/" + domain + ".mat")
        Xs,ys,Xt,yt = data["Xs"],data["Ys"].flatten()-1,data["Xt"],data["Yt"].flatten()-1 
        selected_class = selected[i]
        Xt,yt = imbalance(Xt,yt,selected_class)

       #####################  Proposed  ##################################
       
        # drop Mg, Ml
        ablation = [0,1,2]

        for j in ablation:
           parameters = [[1e-3,1e-2,1e-1,1,10,100,1000] for i in range(3)]
           if j == 0:
               parameters[0] = [0]
               parameters[1] = [0]
           elif j == 1:
               parameters[1] = [0]
           best_acc = 0
           best_process = None
           best_error = None
           best_mmd = None
           optimal = None

           for beta in parameters[0]:
                for gamma in parameters[1]: 
                    for lamda in parameters[2]:
                        for dim in [10,20,30,40,50,60,70,80,90,100]:
                            list_acc, max_acc, list_otr, list_mmd, features, W = LPAJT_TF(Xs,ys,Xt,yt,beta,gamma,lamda,dim,selected_class, kernel_type, soft, seed, T = 20)
                                
                            if len(selected_class) == 1:
                                
                                if max(list_acc) - min(list_otr) > best_acc:
                                    
                                    best_acc = max(list_acc) - min(list_otr)
                                    best_otr = min(list_otr)
                                    optimal = [beta,gamma,lamda,dim]
                                    dictionary = {'dropped_class': selected_class, 'best_auc':max_acc, 
                                                  'best_otr':best_otr, 'mmd':list_mmd, 'convergence_acc':list_acc, 
                                                  'convergence_otr': list_otr, 'features': features, 'class_weights': W, 'optimal': optimal}
                                    np.save(root_path + 'results'  + '/' + domain +' LPAJT_' + "DroppedItem" + str(j) + '_' + 'DroppedClass' + str(selected_class) + '.npy', dictionary)
                            else:
                                if max_acc > best_acc:
                                    best_acc = max_acc
                                    optimal = [beta,gamma,lamda,dim]
                                    dictionary = {'dropped_class': selected_class, 'best_acc': best_acc, 'mmd': list_mmd, 'convergence_acc': list_acc, 
                                                  'features': features, 'class_weights': W, 'optimal': optimal}
                                    np.save(root_path + 'results'  + '/' + domain +' LPAJT_' + "DroppedItem" + str(j) + '_' + 'DroppedClass' + str(selected_class) + '.npy', dictionary)

        

        #####################  LSC ################################## 
        best_acc = 0
        optimal = None   
        metric = None
        threshold = 0.2
        for k in [3, 5, 7, 9, 11, 13, 15]:
            for delta in [1e-3,1e-2,1e-1,1,10,100,1000]:
                for lamb in [1e-3,1e-2,1e-1,1,10,100,1000]:
                    for dim in [10,20,30,40,50,60,70,80,90,100]:            

                        list_acc, max_acc, list_otr, features= LSC_TF(
                            Xs, ys, Xt, yt, lamb, delta, k, threshold, dim, selected_class, kernel_type, soft, seed)

                        if len(selected_class) == 1:
                            if max(list_acc) - min(list_otr) > best_acc:
                                
                                best_acc = max(list_acc) - min(list_otr)
                                best_otr = min(list_otr)
                                optimal = [lamb,delta,k,threshold,dim]
                                dictionary = {'dropped_class': selected_class, 'best_auc':max_acc, 'best_otr':best_otr, 
                                              'convergence_acc': list_acc, 'convergence_otr': list_otr, 'features': features, 'optimal': optimal}
                                np.save(root_path + 'results' + '/' + domain +' LSC_' + str(selected_class) + '.npy', dictionary)
                        else:
                            if max_acc > best_acc:
                                best_acc = max_acc
                                optimal = [lamb,delta,k,threshold,dim]
                                dictionary = {'dropped_class': selected_class,'best_acc':best_acc,'convergence_acc':list_acc,
                                              'features': features, 'optimal': optimal}
                                np.save(root_path + 'results'  + '/' + domain +' LSC_' + str(selected_class) + '.npy', dictionary)


        #####################  TCA  ##################################
        optimal = None
        best_acc = 0
        for lamb in [1e-3,1e-2,1e-1,1,10,100,1000]:
             for dim in [10,20,30,40,50,60,70,80,90,100]:  
                result, features = TCA_TF(
                    Xs, ys, Xt, yt, lamb, dim, selected_class, kernel_type, soft)

                if len(selected_class) == 1:
                    if result[0] - result[-1] > best_acc:
                        best_acc = result[0] - result[-1]
                        best_otr = result[-1]
                        optimal = [lamb,dim]
                        dictionary = {'dropped_class': selected_class,'best_auc': result, 
                                      'features': features, 'best_otr': best_otr, 'optimal': optimal}
                        np.save(root_path + 'results' + '/' + domain +
                                ' TCA_' + str(selected_class) + '.npy', dictionary)
                else:
                    if result > best_acc:
                        best_acc = result
                        optimal = [lamb,dim]
                        dictionary = {'dropped_class': selected_class,'best_acc':best_acc,
                                      'features': features, 'optimal': optimal}
                        np.save(root_path + 'results'  + '/' + domain +
                                ' TCA_' + str(selected_class) + '.npy', dictionary)
                
          
        #####################  JDA ##################################
        best_acc = 0
        optimal = None   

        for lamb in [1e-3,1e-2,1e-1,1,10,100,1000]:
             for dim in [10,20,30,40,50,60,70,80,90,100]:    

                list_acc, max_acc, list_otr, features = JDA_TF(
                    Xs, ys, Xt, yt, lamb, dim, selected_class, kernel_type, soft, seed)

                if len(selected_class) == 1:
                    if max(list_acc) - min(list_otr) > best_acc:
                        best_acc = max(list_acc) - min(list_otr)
                        best_otr = min(list_otr)
                        optimal = [lamb,dim]
                        dictionary = {'dropped_class': selected_class, 'best_auc':max_acc, 'best_otr':best_otr, 'convergence_acc':list_acc, 
                                      'convergence_otr':list_otr, 'features': features,'optimal':optimal}
                        np.save(root_path + 'results' + '/' + domain +
                                ' JDA_' + str(selected_class) + '.npy', dictionary)
                else:
                    if max_acc > best_acc:
                        best_acc = max_acc
                        optimal = [lamb,dim]
                        dictionary = {'dropped_class': selected_class,'best_acc':best_acc,
                                      'features': features, 'convergence_acc': list_acc, 'optimal': optimal}
                        np.save(root_path + 'results'  + '/' + domain +' JDA_' + str(selected_class) + '.npy', dictionary)



        #####################  KNN ##################################
        best_acc = 0
        optimal = None
        for k in [3, 5, 7, 9, 11, 13,15]:
            result = KNN_TF(Xs,ys,Xt,yt,k,selected_class)
            if len(selected_class) == 1:
                if result[0] - result[-1] > best_acc:
                    best_acc = result[0] - result[-1]
                    best_otr = result[-1]
                    optimal = [k]
                    dictionary = {'dropped_class': selected_class,'best_auc': result, 
                                  'best_otr': best_otr, 'optimal': optimal}
                    np.save(root_path + 'results' + '/' + domain +
                            ' KNN_' + str(selected_class) + '.npy', dictionary)
            else:
                if result > best_acc:
                    best_acc = result
                    optimal = [k]
                    dictionary = {'dropped_class': selected_class,'best_acc':best_acc,'optimal':optimal}
                    np.save(root_path + 'results' + '/' + domain +
                            ' KNN_' + str(selected_class) + '.npy', dictionary)
    
         
 
        #####################  SVM  ##################################
        best_acc = 0
        optimal = None
        for C in [1e-4,1e-3,1e-2,1e-1,1,10,100,1000,10000]:
            result = SVM_TF(Xs, ys, Xt, yt, C, selected_class, kernel_type)

            if len(selected_class) == 1:
                if result[0] - result[-1] > best_acc:
                    best_acc = result[0] - result[-1]
                    best_otr = result[-1]
                    optimal = [C]
                    dictionary = {'dropped_class': selected_class,
                                  'best_auc': result, 'best_otr': best_otr, 'optimal': optimal}
                    np.save(root_path + 'results' + '/' + domain + 
                            ' SVM_' +str(selected_class) + '.npy', dictionary)
            else:
                if result > best_acc:
                    best_acc = result
                    optimal = [C]
                    dictionary = {'dropped_class': selected_class,'best_acc':best_acc,'optimal':optimal}
                    np.save(root_path + 'results'  + '/' + domain +
                            ' SVM_' + str(selected_class) + '.npy', dictionary)
    
