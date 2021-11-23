#!/home/kiril/python_env_iron_ment/new_proj/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math
import os

path_save_eval = '/home/kiril/github/AGN_article_final_data/inform'

path_classifire = '/home/kiril/github/ML_with_AGN/ML/code/results'
name_classifire = os.listdir(path_classifire)

fuzzy_options = ['normal']
#fuzzy_options = ['normal', 'fuzzy_err', 'fuzzy_dist']

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

	return ev



for cl in name_classifire:
    for fuzzy_option in fuzzy_options:
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
