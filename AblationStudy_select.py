#####################################################################################################################
#                            Visualize the selected source site classes weights                                     #
#####################################################################################################################
import numpy as np
import matplotlib.pyplot as plt


    
def class_weights_plot(W):
    
    fig = plt.figure(dpi = 600,figsize = (36,4))
    
    fig.subplots_adjust(hspace=0.22, wspace=0.22)    
    
    task = [r"$S \rightarrow G$",r"$G \rightarrow S$"]  
    Ct = r"$\mathcal{C}_{t}$ = "
    
    selected = [[0],[1],[2],[1,2],[0,2],[0,1]]
    la = list("abcdef")
    
    for i in range(6):
        
        s = selected[i]
        shared = list(set([0,1,2]).difference(set(s)))
        label = np.array(["NC","MCI","AD"])
        ax = fig.add_subplot(1, 6, i+1)
        classes = shared+s  
        weight = W[i*2:(i+1)*2]

        size = 3
        x = np.arange(size)
        task = [r"$S \rightarrow G$",r"$G \rightarrow S$"]

        total_width, n = 0.8, 2
        width = total_width / n
        x = x - (total_width - width) / 2
        if len(s) == 1:
            idx = -1
        else:
            idx = -2
            
        weight0 = np.array(weight[0])[classes]
        weight1 = np.array(weight[1])[classes]

        
        ax.bar(x[:idx]-0.02 , weight0[:idx],  width=width, label=task[0],color = "red",edgecolor ='red',alpha = None)
        ax.bar((x + width+0.02)[:idx], weight1[:idx], width=width, label=task[1],color = "white",hatch="\\",edgecolor ='red',alpha = None)
        
        
        ax.bar(x[idx:] -0.02 , weight0[idx:],  width=width, label=task[0],color = "cornflowerblue",edgecolor ='cornflowerblue',alpha = None)
        ax.bar((x + width)[idx:]+0.02 , weight1[idx:], width=width, label=task[1],color = "white",hatch="\\",edgecolor ='cornflowerblue',alpha = None)         
    
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(label[classes], fontsize = 22)           
    
        ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
        ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0], fontsize = 20)  
        
        from matplotlib.patches import Patch
        lines_labels = [ax.get_legend_handles_labels() for ax in [fig.axes[0],fig.axes[-1]]]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        lines.insert(0,Patch(facecolor="white",label = "Task " + task[0] + " :"))
        lines.insert(1,Patch(facecolor="white",label = "Task " + task[1] + " :"))
        labels.insert(0,"Shared Classes Weight :")
        labels.insert(1,"Outlier Classes Weight :")
        lines = [lines[0],lines[2],lines[3],lines[1],lines[4],lines[5]]
        
        labels = [labels[0],labels[2],labels[3],labels[1],labels[4],labels[5]]
        fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.40, 1.25), ncol=6,framealpha = 0,prop={'size': 25})   
        if s == [0]:
            ax.set_ylabel("Landmark Confidence",{'size':20})
        label_dict = ["MCI+AD","NC+AD","NC+MCI","NC","MCI","AD"]
        ax.set_title("(" + la[i] + ")" + " " + Ct + label_dict[i],y = -0.31, fontsize = 28)
    plt.savefig(root_path + "experiment" + "/" + 'selected.png', bbox_inches='tight')
            

root_path = ""
selected = [[0],[1],[2],[1,2],[0,2],[0,1]]
Weight = []

for s in selected:
    for domain in ["PL_S_PL_G","PL_G_PL_S"]:  
        path = root_path + 'results'  + '/' + domain +' LPAJT_' + "DroppedItem" + str(2) + '_' + 'DroppedClass' + str(s) + '.npy'
        parameters = np.load(path,allow_pickle=True)
        optimal = parameters.item()['optimal']
        W = parameters.item()['class_weights']
        Weight.append(W)
        
        
########################## demonstrate selected source class ##################################################
class_weights_plot(Weight)  






      



            
        

