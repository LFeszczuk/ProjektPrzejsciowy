import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from som import SOM
this_folder = (Path(__file__).parent / "..").resolve()

def input_signal_draw(data):
    print(data)

def data_extract():
    path_list = {
        "path_2": Path(f"{this_folder}\\Sterowanie_protezy\\data\\2"),
        "path_3": Path(f"{this_folder}\\Sterowanie_protezy\\data\\3"),
        "path_4": Path(f"{this_folder}\\Sterowanie_protezy\\data\\4"),
        "path_5": Path(f"{this_folder}\\Sterowanie_protezy\\data\\5"),
        "path_6": Path(f"{this_folder}\\Sterowanie_protezy\\data\\6"),
        "path_7": Path(f"{this_folder}\\Sterowanie_protezy\\data\\7"),
    }
    li = []
    for key in path_list:
        if not (os.path.exists(path_list[key])):
            print(["Could not find ", path_list[key]])
        file_list = [
            log
            for log in path_list[key].glob("*.csv")
        ]
        
        for filename in file_list:
            #Load data, drop unnecessary columns, replace commas, turn string to float
            df = pd.read_csv(filename, index_col=None, header=None,sep=";")
            df=df.drop([1, 3,5,7,9,11,13,14], axis=1)  
            df=df.replace(',', '.',regex=True)
            df = df.astype(float)
            #get df as a list of series map it to fourier turn back to list
            df.columns=[i for i in range(8)]
            series_list = [df[col] for col in df]
            f_df=map(feature_reduction,series_list)
            f_df=list(f_df)
            #flatten the list of lists
            f_df = [item for sublist in f_df for item in sublist]
            #make the dict and append it to temp_list with every file in folder where f_df is list of features
            li.append(f_df)
        #dict with every folder and its list of files,df
        # data_dict[path_list[key].name]=li

        # frame = pd.concat(li, axis=0, ignore_index=True)
    df_processed=pd.DataFrame(li)

    return df_processed

def feature_reduction(signal):
    G_prim=[]
    s=[]
    selected_f=[]
    gamma=0.4
    IP_v=4
    IPS_v=0
    e=20 #wanted number of features

    new_s=np.fft.rfft(signal)

    def sigma_sum(start, end, expression):
        return sum(expression(i) for i in range(start, end))
    def harmonic_average(i):
        return (G_prim[i]/IPE_v-IPS_v+1)
    
    #Averaging of the spectral density
    # G′(n)= G′(n −1) + γ (G(n) − G′(n −1)) 
    for idx,i in enumerate(new_s):
        if idx==0:
            G_prim.append(i)
        else:
            G_prim.append(G_prim[idx-1]+gamma*(i-G_prim[idx-1]))

    #smoothing with averaging the harmonics with their neighbor
    for idx,i in enumerate(G_prim):
        if (idx-IP_v) < 0:
            IPS_v=0
        else:
            IPS_v=idx-IP_v
        if idx+IP_v>len(G_prim):
            IPE_v=len(G_prim)
        else:
            IPE_v=idx+IP_v
        s.append(sigma_sum(IPS_v, IPE_v, harmonic_average))

    #linear selection
    for k in range(0,e):
        selected_f.append(s[int(k*(len(s)-1)/(e-1))])

    return selected_f

def main():
    data=data_extract()
    som_net=SOM(6, 6, 160, 0.5, 0.5, 100)
    som_net.train(data)
    
if __name__=="__main__":
    main()