#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import os
import sklearn.metrics as skmetrics

from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

legend_size = 10

path_save_eval = '/home/kiril/github/AGN_article_final_data/inform'

path_classifire = '/home/kiril/github/ML_with_AGN/ML/code/results'
#path_classifire = '/home/kiril/github/AGN_article_final_data/inform/results'
name_classifire = os.listdir(path_classifire)

#fuzzy_options = ['normal']
#fuzzy_options = ['normal', 'fuzzy_err', 'fuzzy_dist']
fuzzy_options = ['fuzzy_err', 'fuzzy_dist']


def eval(y,y_pred,n):
	count = 0
	TP, FP, TN, FN = 0,0,0,0
	for i in range(n):
		if(y[i]==y_pred[i]):
			count+=1
		if(y[i]==1):
			if(y[i]==y_pred[i]):
				TP += 1
			else:
				FP += 1
		if(y[i]==0):
			if(y[i]==y_pred[i]):
				TN += 1
			else:
				FN += 1
	Acc = count/n
	pur_a = TP/(TP+FP)
	pur_not_a = TN/(TN+FN)
	com_a = TP/(TP+FN)
	com_not_a = TN/(TN+FP)
	f1 = 2*TP/(2*TP+FP+FN)
	fpr = FP/(TN+FN)
	tnr = TN/(TN+FN)
	bAcc = (TP/(TP+FP)+TN/(TN+FN))/2.
	k = 2*(TP*TN-FN*FP)/((TP+FP)*(FP+TN)+(TP+FN)*(FN+TN))
	mcc = (TP*TN-FP*FN)/math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
	BinBs = (FP+FN)/(TP+FP+FN+TN)

	print(np.array([Acc,pur_a,pur_not_a,com_a,com_not_a,f1,fpr,tnr,bAcc,k,mcc,BinBs]))
	ev = pd.DataFrame([np.array([Acc,pur_a,pur_not_a,com_a,com_not_a,f1,fpr,tnr,bAcc,k,mcc,BinBs])], 
    columns=['Accuracy','AGN_purity','nonAGN_precision','AGN_completness','nonAGN_completness','F1',
    'FPR','TNR','bACC','K','MCC','BinaryBS'])
	'''
	print("Accuracy 				[worst: 0; best: 1]:",              Acc)
	print("AGN purity 				[worst: 0; best: 1]:",     pur_a)
	print("nonAGN precision 			[worst: 0; best: 1]:",    pur_not_a)
	print("AGN completness 			[worst: 0; best: 1]:",       com_a)
	print("nonAGN completness 			[worst: 0; best: 1]:",     com_not_a)
	print("F1  					[worst: 0; best: 1]:",		f1)
	#print("AGN_F:",     2*(Tq/(Tq+Fq)*Tq/(Tq+Fs))/(Tq/(Tq+Fq)+Tq/(Tq+Fs)) )
	#print("non_AGN_F:",     2*(Ts/(Ts+Fs)*Ts/(Ts+Fq))/(Ts/(Ts+Fq)+Ts/(Ts+Fs)) )
	print("FPR (false positive rate) 		[worst: 1; best: 0]:",		fpr)
	print("TNR (true negative rate) 		[worst: 0; best: 1]:",		tnr)
	print("bACC (balanced accuracy) 		[worst: 0; best: 1]:", bAcc)
	print("K (Cohen's Kappa) 			[worst:-1; best:+1]:",		k)
	print("MCC (Matthews Correlation Coef) 	[worst:-1; best:+1]:",		mcc)
	print("BinaryBS (Brierscore) 			[worst: 1; best: 0]:", BinBs)
	'''
	return ev

def i_need_more_eval(ax1_p, ax2_p, ax3_p, ax1_r, ax2_r, ax3_r, index, name, label, data_general_label):
	fpr, tpr, thresholds = skmetrics.roc_curve(label, data_general_label, pos_label=1)
	roc_curve_df = pd.DataFrame({"fpr": fpr, "tpr": tpr,
									"thresholds": thresholds})
	roc_curve_df = roc_curve_df[roc_curve_df['thresholds'] < 0.99]									
	roc_curve_df.to_csv(f"{path_save_eval}/{name}_roc.csv", index=False)
	precision, recall, thresholds = skmetrics.precision_recall_curve(label, data_general_label)
	pr_curve_df = pd.DataFrame({"precision": precision, "recall": recall, 
                                    "thresholds": np.append(thresholds, 1)})
	pr_curve_df = pr_curve_df[pr_curve_df['thresholds'] < 0.99]	
	pr_curve_df.to_csv(f"{path_save_eval}/{name}_pr.csv", index=False)

	ax1_r.plot(roc_curve_df['thresholds'],roc_curve_df['fpr'],c=c[index],label=name)
	ax2_r.plot(roc_curve_df['thresholds'],roc_curve_df['tpr'],c=c[index],label=name)
	ax3_r.plot(roc_curve_df['fpr'],roc_curve_df['tpr'],c=c[index],label=name)

	ax1_p.plot(pr_curve_df['thresholds'],pr_curve_df['precision'],c=c[index],label=name)
	ax2_p.plot(pr_curve_df['thresholds'],pr_curve_df['recall'],c=c[index],label=name)
	ax3_p.plot(pr_curve_df['recall'],pr_curve_df['precision'],c=c[index],label=name)

#c=list(mcolors.TABLEAU_COLORS)
c=np.append(list(mcolors.TABLEAU_COLORS),list(mcolors.BASE_COLORS))

save_path = path_save_eval
ml_class = ['dark', 'normal']

for fuzzy_option in fuzzy_options:
    fig, ((ax1_p, ax2_p, ax3_p), (ax1_r, ax2_r, ax3_r)) = plt.subplots(2,3)		
    fig.suptitle(f'PR_curve and ROC_curve general data')
    ax1_p.set_xlabel('Thresholds')
    ax1_p.set_ylabel('Precision')
    ax2_p.set_xlabel('Thresholds')
    ax2_p.set_ylabel('Recall')
    ax3_p.set_xlabel('Recall')
    ax3_p.set_ylabel('Precision')
    ax1_r.set_xlabel('Thresholds')
    ax1_r.set_ylabel('FPR')
    ax2_r.set_xlabel('Thresholds')
    ax2_r.set_ylabel('TPR')
    ax3_r.set_xlabel('FPR')
    ax3_r.set_ylabel('TPR')
    fig.set_size_inches(25,20)
    index=0
    
    for cl in name_classifire:
        
        data_general = pd.read_csv(f"{path_classifire}/{cl}/{fuzzy_option}/{fuzzy_option}_generalization.csv",sep=",",header=0)
        #print(data_general)
        label = []
        n=data_general.shape[0]
        print(n)
        for i in range(n):
            if (data_general['name'].iloc[i] == "AGN"):
                label.append(1)
            else: label.append(0)
        #print(label)
        
        ev = eval(label,data_general['y_pred'],n)
        ev.to_csv(f'{path_save_eval}/{cl}_evaluate.csv', index=False)
        
        i_need_more_eval(ax1_p, ax2_p, ax3_p, ax1_r, ax2_r, ax3_r, index, cl, label, data_general['y_prob_positive_class'])
        index+=1
    
    #for ind in range(5):
    for ml_c in ml_class:
        data_NN = pd.read_csv(f"{path_save_eval}/ml_{ml_c}_{1}_prob.csv",sep=",",header=0)
        n=data_NN.shape[0]
        label = []
        for i in range(n):
        	if(data_NN['y_prob'].iloc[i] > 0.5):
        	    label.append(1)
        	else:
        	    label.append(0)
        ev = eval(data_NN['Y'], label, n)
        ev.to_csv(f"{path_save_eval}/{ml_c}_{1}_evaluate.csv", index=False)
        i_need_more_eval(ax1_p, ax2_p, ax3_p, ax1_r, ax2_r, ax3_r, index, f"{ml_c}_{1}", data_NN['Y'], data_NN['y_prob'])
        index+=1
    
    ax1_p.legend(loc=3, prop={'size': legend_size})
    ax2_p.legend(loc=3, prop={'size': legend_size})
    ax3_p.legend(loc=3, prop={'size': legend_size})
    ax1_r.legend(loc=1, prop={'size': legend_size})
    ax2_r.legend(loc=3, prop={'size': legend_size})
    ax3_r.legend(loc=4, prop={'size': legend_size})
    
    fig.savefig(save_path+'/'+fuzzy_option+'PR_ROC_curve_all_g_.png')	
    plt.close(fig)


