import pandas as pd
import numpy as np
import glob
import os
from pathlib import Path
from som import SOM
from minisom import MiniSom
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
# from mglearn import discrete_scatter
from sklearn.decomposition import PCA
this_folder = (Path(__file__).parent / "..").resolve()

def input_signal_draw(data):
    print(data)

def data_extract(gamma,IP_v,IPS_v,e):
    path_list = {
        "path_2": Path(f"{this_folder}\\Sterowanie_protezy\\data\\2"),
        "path_3": Path(f"{this_folder}\\Sterowanie_protezy\\data\\3"),
        "path_4": Path(f"{this_folder}\\Sterowanie_protezy\\data\\4"),
        "path_5": Path(f"{this_folder}\\Sterowanie_protezy\\data\\5"),
        "path_6": Path(f"{this_folder}\\Sterowanie_protezy\\data\\6"),
        "path_7": Path(f"{this_folder}\\Sterowanie_protezy\\data\\7"),
    }
    li = []
    for idx,key in enumerate(path_list):
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
            f_df=[]
            #get df as a list of series map it to fourier turn back to list
            df.columns=[i for i in range(8)]
            series_list = [df[col] for col in df]
            for series in series_list:
                f_df.append(feature_reduction(series,gamma,IP_v,IPS_v,e))

            #flatten the list of lists
            f_df = [item for sublist in f_df for item in sublist]
            f_df.append(idx)
            #make the dict and append it to temp_list with every file in folder where f_df is list of features
            li.append(f_df)
    df_processed=pd.DataFrame(li)
    return df_processed

def feature_reduction(signal,gamma,IP_v,IPS_v,e):
    G_prim=[]
    s=[]
    selected_f=[]
    #wanted number of features

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

    s=[i.real for i in s]

    #linear selection
    for k in range(0,e):
        selected_f.append(s[int(k*(len(s)-1)/(e-1))])

    return selected_f

def classify(som, data,X_train,y_train):
    """Classifies each sample in data in one of the classes definited
    using the method labels_map.
    Returns a list of the same length of data where the i-th element
    is the class assigned to data[i].
    """
    winmap = som.labels_map(X_train, y_train)
    default_class = np.sum(list(winmap.values())).most_common()[0][0]
    result = []
    for d in data:
        win_position = som.winner(d)
        if win_position in winmap:
            result.append(winmap[win_position].most_common()[0][0])
        else:
            result.append(default_class)
    return result

def som_clasificator(data):

    Y_data = data.iloc[:, -1].values
    X_data = data.iloc[:, :-1].values
    X_data = np.apply_along_axis(lambda L: (L - np.min(L))/(np.max(L) - np.min(L)),1,X_data)

    # labels=np.unique(Y_data)
    X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, stratify=Y_data)

    som = MiniSom(10, 10, 64, sigma=2, learning_rate=0.5, 
                  neighborhood_function='triangle', random_seed=10)
    som.pca_weights_init(X_train)
    som.train_random(X_train, 500, verbose=False)
    class_report=classification_report(y_test, classify(som, X_test,X_train,y_train))
    print(class_report)


# def plot_scattermatrix(data, target):
#     df = pd.DataFrame(data)
#     df['target'] = target
#     return sns.pairplot(df, hue='target', diag_kind='hist')

# def lvq_classificator(data):
#     X_data = data.iloc[:, :-1].values
#     Y_data = data.iloc[:, -1].values
#     x_real = X_data.real  
#     x_im = X_data.imag
#     X_data=x_real

#     X_data = np.apply_along_axis(lambda L: (L - np.min(L))/(np.max(L) - np.min(L)),1,X_data)
#     # labels=np.unique(Y_data)
#     X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, stratify=Y_data)
#     lvqnet = algorithms.LVQ3(n_inputs=64, n_classes=6)
#     lvqnet.train(X_train, y_train, epochs=100)
#     plot_scattermatrix(X_train, y_train)
#     plot_scattermatrix(X_train=lvqnet.weight, y_train=lvqnet.subclass_to_class)
#     plt.show()
def main():
    # gamma=0.4
    # IP_v=4
    # IPS_v=0
    # e=8
    data=data_extract(0.4,4,0,8)

    som_clasificator(data)
    # lvq_classificator(data)


if __name__=="__main__":
    main()