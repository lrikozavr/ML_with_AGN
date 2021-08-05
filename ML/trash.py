    #''
    time.sleep(100000)

    agn_sample=DataP(agn_sample,0) 									############################flag_color
	
    model1 = LoadModel(output_path_mod,output_path_weight,optimizer,loss)
    Class = model1.predict(train, batch_size)
    
    Class = np.array(Class)

    g=open(output_path_predict+"/"+name,'w') 
    Class.tofile(g,"\n")
    g.close()

    j=0
    for i in range(np.size(Class)):
        #if(Class[i]<0.5):
            #Class[i] = 0
        if(Class[i]>=0.5):
            #Class[i] = 1
            j+=1
    print(name+":	",j /np.size(Class) *100,"%")
#''

#test(data_agn_star)
#test(data_agn_qso)
#test(data_agn_gal)
#test(data_agn_star_qso)

#test(data_agn_star_gal)
#test(data_agn_qso_gal)




'''
#test(data_agn_star_qso_gal)
import sys
sys.path.insert(1, 'image_download')
from image_download import download_image
def data_download(data):
    #print(data)
    n = data.shape[0]
    for i in range(n):
        #print(float(data['RA'][i]))
        download_image(float(data['RA'][i]),float(data['DEC'][i]))
#data_download(data_gal)

#download_image(200,50)
'''

#import sys
#sys.path.insert(1, 'image_download')
#from image_download import convert_image
#convert_image("/home/kiril/github/ML_data/test/AGN")
#convert_image("/home/kiril/github/ML_data/test/GALAXY")
#from ml import Start_IMG
#Start_IMG()