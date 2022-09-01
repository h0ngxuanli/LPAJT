#####################################################################################################################
#                                     Check the parameter sensitivity of LPAJT                                     #
#####################################################################################################################
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from models import LPAJT_TF
from utils import imbalance

def parameter_plot(acc_list1,acc_list2,ntr_list1,ntr_list2,s):
    fig = plt.figure(dpi = 400,figsize = (24,4))
    fig.subplots_adjust(hspace=0.42, wspace=0.42)    

    task = [r"$S \rightarrow G$",r"$G \rightarrow S$"]    
    Ct = r"$\mathcal{C}_{t}$ = "
    
    if s == []:
        scenario = "Identical"
        idx = "a"
    elif s == [0]:
        scenario = "MCI+AD"
        idx = "b"       
    elif s == [1]:
        scenario = "NC+AD"
        idx = "c"    
    elif s == [2]:
        scenario = "NC+MCI"
        idx = "d"    
    elif s == [1,2]:
        scenario = "NC"
        idx = "e"    
    elif s == [0,2]:
        scenario = "MCI"
        idx = "f"    
    elif s == [0,1]:
        scenario = "AD"
        idx = "g"    
 
    
    
    if len(s) == 0 or len(s) == 2:
        metric = r"ACC(%)"
    else:
        metric = r"AUC(%)"   
    for i in range(4):
        ax = fig.add_subplot(1, 4, i+1)
        if i < 3:
            Range = [1e-3,1e-2,1e-1,1,10,100,1000]
        else:
            Range = [10,20,30,40,50,60,70,80,90,100]    
        
        sen1 = np.array(acc_list1[i])
        sen2 = np.array(acc_list2[i])
        ntr1 = np.array(ntr_list1[i])
        ntr2 = np.array(ntr_list2[i])
        x = range(len(Range))
        if len(s) == 1:
            label  = r"AUC(%)"
        else:
            label  = r"ACC(%)"
        ax.plot(x,sen1*100,marker = 'o',ms =12 ,color = 'r', label = label,lw = 2,clip_on=False, markerfacecolor="None", markeredgecolor='r', markeredgewidth=1.5)
        ax.plot(x,sen2*100,marker = '^',ms = 12,color = 'b', label = label,lw = 2,clip_on=False, markerfacecolor="None",markeredgecolor='b', markeredgewidth=1.5)  
        
        plt.grid(which = "both")
            
        ax.set_xticks(x)
        ax.set_xticklabels(Range,fontsize=14)
        ax.set_yticks([0,20,40,60,80,100])
        ax.set_yticklabels([" ",20,40,60,80,100],fontsize=14)
        ax.set_ylabel(metric,fontsize=14)        

        
        if len(s) == 1:
           ax2 = ax.twinx()
           label = r"NTR(%)"
           
           ax2.plot(x,ntr1*100,marker = 'o',ms =12,color = 'lime',label =  label,lw = 2,clip_on=False, markerfacecolor="None",markeredgecolor='lime', markeredgewidth=1.5)
           ax2.plot(x,ntr2*100,marker = '^',ms = 12,color = 'fuchsia',label = label,lw = 2,clip_on=False , markerfacecolor="None",  markeredgecolor='fuchsia', markeredgewidth=1.5)
        
           ax2.set_ylim((0, 40))
           ax2.set_yticks([8,16,24,32,40])
           ax2.set_yticklabels([8,16,24,32,40],fontsize=14)
           ax2.set_ylabel(r"NTR(%)",fontsize=14)
        if i ==0:
            ax.set_xlabel(r'$\beta$',fontsize = 20)
        elif i==1:
            ax.set_xlabel(r'$\gamma$',fontsize = 20)
        elif i==2:
            ax.set_xlabel(r'$\lambda$',fontsize = 20)
        elif i==3:
            ax.set_xlabel(r'$d$',fontsize = 20)
        
        ax.set_xlim((x[0],x[-1]))
        from matplotlib.patches import Patch
    
        #lines_labels.append(ax2.get_legend_handles_labels())
        lines_labels = [ax.get_legend_handles_labels() for ax in [fig.axes[0],fig.axes[-1]]]
        
       
        # if len(s) == 1:
        #     lines_labels.insert(0,ax2.get_legend_handles_labels())
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        
        
        lines.insert(0,Patch(facecolor="white",label = "Task " + task[0] + " :"))
        lines.insert(1,Patch(facecolor="white",label = "Task " + task[1] + " :"))
        labels.insert(0,"Task " + task[0] + " :")
        labels.insert(1,"Task " + task[1] + " :")
        if len(s) != 1:
            lines = [lines[0],lines[2],lines[1],lines[3]]
            labels = [labels[0],labels[2],labels[1],labels[3]]            
        else:
            lines = [lines[0],lines[2],lines[4],lines[1],lines[3],lines[5]]
            labels = [labels[0],labels[2],labels[4],labels[1],labels[3],labels[5]]
        fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.41, 1.3), ncol=6,framealpha = 0,prop={'size': 17})
        
        fig.suptitle("("+ idx + ")"  + " parameter senestivity for " + Ct + scenario, fontsize=26.5, y=-0.09)
    
    plt.savefig(root_path + "experiment" + "/" + str(s) +
                'parameter.png', bbox_inches='tight')


selected = [[], [0], [1], [2], [1, 2], [0, 2], [0, 1]]
root_path = ""

for s in selected:
    all_para_acc1 = []
    all_para_acc2 = []
    all_para_ntr1 = []
    all_para_ntr2 = []

    for domain in ["PL_S_PL_G", "PL_G_PL_S"]:
        data = loadmat(root_path + "data" + "/" + domain + ".mat")
        Xs,ys,Xt,yt = data["Xs"],data["Ys"].flatten()-1,data["Xt"],data["Yt"].flatten()-1 
        selected_class = s
        Xt,yt = imbalance(Xt,yt,selected_class)


        path = root_path + 'results' + '/' + domain + ' LPAJT_' + "DroppedItem" + str(2) + '_' + 'DroppedClass' + str(s) + '.npy'
        parameters = np.load(path, allow_pickle=True)
        optimal = parameters.item()['optimal']

        for i in range(4):
            acc_list1 = []
            acc_list2 = []
            ntr_list1 = []
            ntr_list2 = []
            
            if i < 3:
                Range =  [1e-3,1e-2,1e-1,1,10,100,1000]
            else:
                Range = [10,20,30,40,50,60,70,80,90,100]
            for j in Range:
                
                sub_optimal = optimal.copy()
                
                # change one paramter and fix the rest
                sub_optimal[i+1] = j
                if domain ==  "PL_S_PL_G":
                    list_acc, max_acc, list_otr, list_mmd, features, W=LPAJT_TF(
                    Xs, ys, Xt, yt, *sub_optimal, selected_class = s, kernel_type = "linear", soft = 1, seed = 666, T = 20)
                    
                    acc_list1.append(max(list_acc))
                    if len(s) ==1:
                        ntr_list1.append(min(list_otr))
                else:
                    acc_list2.append(max(list_acc))
                    if len(s) ==1:
                        ntr_list2.append(min(list_otr))
                        
                        
            if len(acc_list1)!=0:
                all_para_acc1.append(acc_list1)
                all_para_ntr1.append(ntr_list1)
            if len(acc_list2)!=0:
                all_para_acc2.append(acc_list2)
                all_para_ntr2.append(ntr_list2) 

    ########################## parameter sentivity ###################################
    parameter_plot(all_para_acc1,all_para_acc2,all_para_ntr1,all_para_ntr2,s)        


