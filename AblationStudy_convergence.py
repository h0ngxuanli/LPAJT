#####################################################################################################################
#                                   Visualize the convergence of AUC/ACC MMD OTR                                   #
#####################################################################################################################
import numpy as np
import matplotlib.pyplot as plt



def convergence_plot(convergence, convergence_MMD, convergence_otr, ori):
    fig = plt.figure(dpi=800, figsize=(20, 8))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    task = [r"$S \rightarrow G$", r"$G \rightarrow S$"]

    Ct = r"$\mathcal{C}_{t}$ = "
    scenario = ["Identical", "MCI+AD", "NC+AD", "NC+MCI", "NC", "MCI", "AD"]
    Index = 'abcdefgh'

    for i in range(7):
        ax = fig.add_subplot(2, 4, i+1)
        ax2 = ax.twinx()

        if i > 3 or i == 0:
            metric = r"ACC(%)"
        else:
            metric = r"AUC(%)"

        ms = 8
        mw = 2
        conver = np.array(convergence[i*2:(i+1)*2][0])
        conver1 = np.array(convergence[i*2:(i+1)*2][1])

        conver_mmd = np.array(convergence_MMD[i*2:(i+1)*2][0])
        conver1_mmd = np.array(convergence_MMD[i*2:(i+1)*2][1])
        

        ori_mmd, ori1_mmd = np.array(ori[i*2:(i+1)*2])

        x = np.arange(len(conver))+1
        x1 = np.arange(len(conver)+1)

        ax.set_xticks(np.array([1, 5, 10, 15, 20]))
        ax.set_xlim((0, 20))
        ax.set_xticklabels([1, 5, 10, 15, 20], fontsize=14)

        # conver_mmd = np.hstack((ori_mmd, conver_mmd))
        # conver1_mmd = np.hstack((ori1_mmd, conver1_mmd))
     
        ax.plot(x1, conver_mmd, marker='o', ms=ms, color='lime',
                label=r"$\mathbf{A}^{\top}\mathbf{X}$ MMD", lw=2, clip_on=False, markerfacecolor="None", markeredgecolor='lime', markeredgewidth=mw)
        ax.plot(x1, conver1_mmd, marker='^', ms=ms, color='fuchsia',
                label=r"$\mathbf{A}^{\top}\mathbf{X}$ MMD", lw=2, clip_on=False, markerfacecolor="None", markeredgecolor='fuchsia', markeredgewidth=mw)

        ax2.plot(x, conver*100, marker='o', ms=ms, color='r', label=r"ACC/AUC(%)", lw=2,
                 clip_on=False, markerfacecolor="None", markeredgecolor='r', markeredgewidth=1.5)
        ax2.plot(x, conver1*100, marker='^', ms=ms, color='blue', label=r"ACC/AUC(%)", lw=2,
                 clip_on=False, markerfacecolor="None",  markeredgecolor='b', markeredgewidth=1.5)
        ax.grid(which="both")
        N = len(x)+1
        ax.plot(range(N), [ori_mmd for q in range(N)],
                label='Raw Data MMD', ls='dashed', color='lime', lw=2.5)
        ax.plot(range(N), [ori1_mmd for q in range(
            N)], label='Raw Data MMD', ls='dashed', color='fuchsia', lw=2.5)
        
        
        ylim = np.ceil(max(ori_mmd, ori1_mmd))
        for k in range(1000):

            if ylim % 5 == 0:
                ax.set_ylim((0, ylim))
                ylim = int(ylim//5)
                break
            ylim += 1
        ax.set_yticks(np.array([ylim*i for i in range(6)]))

        ax.set_yticklabels(np.array([ylim*i for i in range(6)]), fontsize=14)

        ax2.set_ylim((0, 100))
        ax2.set_yticks([20, 40, 60, 80, 100])
        ax2.set_yticklabels([20, 40, 60, 80, 100], fontsize=14)
        ax.set_ylabel("Square MMD", fontsize=14)
        ax2.set_ylabel(metric, fontsize=14)

        ax.set_xlabel("Iterations", fontsize=14)
        ax.set_title("(" + Index[i] + ") " + Ct +
                     scenario[i], y=-0.36, fontsize=20)

    ax = fig.add_subplot(2, 4, 8)

    marker = [">", "<", "D", "s", "o", "v"]

    # ["#82B242","#7D2F90","#EDB01D","#D95218","#4ABDEE","#0070BC"]
    color = ["#813491", "#0E79C1", "#EEB52D", "#DE6936", "#50C1EE", "#0A6A6A"]
    #lsty = ["solid","solid","dashdot","dashdot","dotted","dotted"]#(0, (3, 1, 1, 1, 1, 1)),(0, (3, 1, 1, 1, 1, 1))]

    for i in range(3):
       conver_ntr = np.array(convergence_otr[i*2:(i+1)*2][0])
       # '#DE6C3B'  '#006464'
       conver1_ntr = np.array(convergence_otr[i*2:(i+1)*2][1])

       marker1, marker2 = marker[i*2:(i+1)*2]
       color1, color2 = color[i*2:(i+1)*2]
       x = np.arange(len(conver))+1

       ax.plot(x, conver_ntr*100, marker=marker1, ms=ms, color=color1, ls="solid", label=Ct + scenario[i+1] + " " + r"NTR(%)", lw=1.5, markerfacecolor="None",
               markeredgecolor=color1, markeredgewidth=mw, clip_on=False)
       ax.plot(x, conver1_ntr*100, marker=marker2, ms=ms, color=color2, ls="solid", label=Ct + scenario[i+1] + " " + r"NTR(%)", lw=1.5, markerfacecolor="None",
               markeredgecolor=color2, markeredgewidth=mw, clip_on=False)
       ax.set_xticks(np.array([1, 5, 10, 15, 20]))
       ax.set_xlim((1, 20))
       ax.set_xticklabels([1, 5, 10, 15, 20], fontsize=14)
       ax.set_ylim((0, 15))
       ax.set_yticks([3, 6, 9, 12, 15])
       ax.set_yticklabels([3, 6, 9, 12, 15], fontsize=14)
       ax.set_ylabel(r"NTR(%)", fontsize=14)
       ax.grid(which="both")
       ax.set_xlabel("Iterations", fontsize=14)
       ax.set_title("(h) NTR convergence", y=-0.36, fontsize=20)

    lines_labels = [ax.get_legend_handles_labels()
                    for ax in [fig.axes[0], fig.axes[-1]]]
    lines_labels.insert(0, ax2.get_legend_handles_labels())

    from matplotlib.patches import Patch

    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    lines.insert(0, Patch(facecolor="white", label="Task " + task[0] + " :"))
    lines.insert(1, Patch(facecolor="white", label="Task " + task[1] + " :"))
    labels.insert(0, "Task " + task[0] + " :")
    labels.insert(1, "Task " + task[1] + " :")
    # by_label = dict(zip(labels, lines))
    fig.legend(lines, labels, loc='upper center', bbox_to_anchor=(
        0.448, 1.0), ncol=7, framealpha=0, prop={'size': 13})

    plt.savefig(root_path + "experiment" + "/" +
                'convergence.png', bbox_inches='tight')


selected = [[],[0],[1],[2],[1,2],[0,2],[0,1]]

convergence = []
convergence_MMD = []
convergence_OTR = []
ori = []
Weight = []
fea_src = []
fea_tar = []
Z_src = []
Z_tar = []
label_src = []
label_tar = []

root_path = ""
for s in selected:
   for domain in ["PL_S_PL_G", "PL_G_PL_S"]:
       
        path = root_path + 'results'  + '/' + domain +' LPAJT_' + "DroppedItem" + str(2) + '_' + 'DroppedClass' + str(s) + '.npy'

        parameters = np.load(path,allow_pickle=True)
        optimal = parameters.item()['optimal']
        list_acc = parameters.item()['convergence_acc']
        list_mmd = parameters.item()['mmd']
        list_otr = parameters.item()['convergence_otr']

        convergence.append(list_acc)
        if len(s) == 1:
            convergence_OTR.append(list_otr)
        convergence_MMD.append(list_mmd)
        ori_mmd = list_mmd[0]

        ori.append(ori_mmd)

########################## convergence ##################################################
convergence_plot(convergence,convergence_MMD,convergence_OTR,ori)
