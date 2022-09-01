#####################################################################################################################
#                       Visualize sample similarities between two sites after adaptation                            #
#####################################################################################################################


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances,cosine_similarity


def similarity_plot(fea_src,fea_tar,Z_src,Z_tar,Ys,Yt,metric = "l2"):    
    
    fig = plt.figure(dpi = 800,figsize = (28,16))
    fig.subplots_adjust(hspace=0.25, wspace=0.25
                        )
    ct = r"$\mathcal{C}_{t}$ = "
    scenario = ["Identical","MCI+AD","NC+AD","NC+MCI","NC","MCI","AD"]
    

    selected = [[],[0],[1],[2],[0,1],[0,1],[0,2]]
    
    for k in [0, 1, 2, 3, 4, 5, 6, 14, 15, 16, 17, 18, 19, 20]:
        
        if k > 6:
            j = k-7
        else:
            j = k
        Xs,ys,Xt,yt,Zs,Zt = fea_src[j],Ys[j],fea_tar[j],Yt[j],Z_src[j],Z_tar[j]


        ax = fig.add_subplot(4, 7, k+1)
        ax1 = fig.add_subplot(4, 7, k+1+7)

        
        if metric == "cosine":
            sim = cosine_similarity(Zt,Zs)
            ori_sim = cosine_similarity(Xt,Xs)
        elif metric == "l2":
            sim = np.exp(-euclidean_distances(Zt,Zs))
            ori_sim = np.exp(-euclidean_distances(Xt,Xs))
    
        Cs = np.unique(ys)
        Ct = np.unique(yt)
        classes = ["NC","MCI","AD"]
        
        
        s_tick_lael = [classes[i] for i in np.unique(ys)]
        t_tick_lael = [classes[i] for i in np.unique(yt)]
    
        s_tick = []
        t_tick = []
        for i in range(len(Cs)):
            s_tick.append(len(ys[ys==Cs[i]]))
        for i in range(len(Ct)):
            t_tick.append(len(yt[yt==Ct[i]]))
            
        s_tick_may = np.cumsum(np.array(s_tick))[:-1]
        t_tick_may = np.cumsum(np.array(t_tick))[:-1]
        

        s_tick_min = np.cumsum([s_tick[0]/2,(s_tick[0]+s_tick[1])/2,(s_tick[2]+s_tick[1])/2])
        
        if len(selected[k%7]) == 1:
            t_tick_min = np.cumsum([t_tick[0]/2,(t_tick[0]+t_tick[1])/2])#,(t_tick[2]+t_tick[1])/2])
            
        elif len(selected[k%7]) == 0:
            t_tick_min = np.cumsum([t_tick[0]/2,(t_tick[0]+t_tick[1])/2,(t_tick[2]+t_tick[1])/2])
        else: 
            t_tick_min = np.cumsum([t_tick[0]/2])#,(t_tick[0]+t_tick[1])/2])
        
        ax.imshow(ori_sim,aspect='auto', cmap = "viridis")#, cmap =   "plasma")#"viridis")
        
        ax.xaxis.set_ticks_position("top")
        ax.set_xticks(s_tick_may)
        ax.set_yticks(t_tick_may)
        
        ax.set_xticks(s_tick_min, minor=True)
        ax.set_yticks(t_tick_min, minor=True)



        ax.set_xticklabels(s_tick_lael,minor=True,fontsize=16)
        ax.set_yticklabels(t_tick_lael,minor=True,fontsize=16)
        
        ax.set_xticklabels("")
        ax.set_yticklabels("")
        
        ax.grid(True,color = 'white',linewidth = 2.5)
        ax.xaxis.set_tick_params(labelsize=20)
        
        ax.xaxis.set_tick_params(width=1,length = 0, which = "minor")
        ax.yaxis.set_tick_params(width=1,length = 0, which = "minor")
        
        ax.xaxis.set_tick_params(width=1,length = 0)
        ax.yaxis.set_tick_params(width=1,length = 0)

        ax1.imshow(sim,aspect='auto', cmap = "viridis")#"plasma")
        
        ax1.xaxis.set_ticks_position("top")
        ax1.set_xticks(s_tick_may)
        ax1.set_yticks(t_tick_may)
        
        
        ax1.set_xticks(s_tick_min, minor=True)
        ax1.set_yticks(t_tick_min, minor=True)
        
        
        ax1.set_xticklabels(s_tick_lael,minor=True,fontsize=16)
        ax1.set_yticklabels(t_tick_lael,minor=True,fontsize=16)
        
        ax1.set_xticklabels("")
        ax1.set_yticklabels("")
        
        ax1.grid(True,color = 'white',linewidth = 2.5)
        ax1.xaxis.set_tick_params(labelsize=20)
        
        ax1.xaxis.set_tick_params(width=1,length = 0, which = "minor")
        ax1.yaxis.set_tick_params(width=1,length = 0, which = "minor")
        
        ax1.xaxis.set_tick_params(width=1,length = 0)
        ax1.yaxis.set_tick_params(width=1,length = 0)
        
#        ax.colorbar()
#        ax1.colorbar()
        
        if k==0:
            ax.set_ylabel( "Original " + r"(S$\rightarrow$G)",fontsize=25)
            ax1.set_ylabel( "Adapted " + r"(S$\rightarrow$G)",fontsize=25)
        if k==14:
            ax.set_ylabel( "Original " + r"(G$\rightarrow$S)",fontsize=25)
            ax1.set_ylabel("Adapted " + r"(G$\rightarrow$S)",fontsize=25)
            
        
        if k >= 14:
            ax1.set_title(ct + scenario[k%7],fontsize=25,y = -0.21)
    plt.savefig(root_path  + "experiment" + "/" + 'similarity.png', bbox_inches='tight')
    


root_path = ""
selected = [[],[0],[1],[2],[1,2],[0,2],[0,1]]

fea_src = []
fea_tar = []
Z_src = []
Z_tar = []
label_src = []
label_tar = []

for domain in ["PL_S_PL_G", "PL_G_PL_S"]:
    for s in selected:    

        path = root_path + 'results'  + '/' + domain +' LPAJT_' + "DroppedItem" + str(2) + '_' + 'DroppedClass' + str(s) + '.npy'
        parameters = np.load(path,allow_pickle=True)
        optimal = parameters.item()['optimal']
     
    ########################## similarity ##################################################
        features = parameters.item()['features']

        Xs, ys, Xt, yt, Xs_new, Xt_new = features
        fea_src.append(Xs)
        fea_tar.append(Xt)
        Z_src.append(Xs_new)
        Z_tar.append(Xt_new)
        label_src.append(ys)
        label_tar.append(yt)

similarity_plot(fea_src,fea_tar,Z_src,Z_tar,label_src,label_tar,metric = "cosine")
           






  



        
    
