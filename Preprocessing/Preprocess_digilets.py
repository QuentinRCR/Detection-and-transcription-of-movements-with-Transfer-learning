# Extraction of each letter of the digilets dataset and create the preprocessed file 

import numpy as np
import pandas as pd
from utils import convert_points_to_images, inter, remove_close_points, smoothen_points,scale_and_center
import sys
import os

# Import necessary paths
absolute_root_path = os.path.dirname(os.path.realpath(__file__)).replace("\\","/") + "/.."
sys.path.insert(1,absolute_root_path)
from paths import preprocessed_datasets_path, digilets_path

sys.path.insert(1,f'{digilets_path}/scripts')
from data import read_original_data # import lib from digilets directory


save_file_name = 'digilets-inter3-28-21.csv'

# get all the files containing interesting data
files = np.array(os.listdir(f'{digilets_path}/data/preprocessed/complete'))
files = (files[np.char.endswith(files,"preprocessed")]) #take only interesting data

def load_dataset(file,df):
    df_raw = read_original_data(f'{digilets_path}/data/preprocessed/complete/{file}')
    hand = df_raw['hand']
    gender = df_raw['gender']
    for letter_index,letter_data in enumerate(df_raw['trajectories']):
        if letter_index in list(range(10,36)):
            for instances in letter_data:

                X = instances[:,0]
                Y = instances[:,1]

                time = instances[:,4]
                pendown = instances[:,3]

                xy_shift = np.array([X[~np.isnan(X)],Y[~np.isnan(Y)]]).T

                # scale the values between 0 and 1 while keeping the aspect ratio + center the number
                xy_scaled = scale_and_center(xy_shift)

                # do the average and remove redundant points 
                xy_smoothed = smoothen_points(xy_scaled,3)
                xy_clean,_ = remove_close_points(xy_smoothed)
                if(xy_clean.shape[0]>3): # to respect m > k must hold
                    interpolation = inter(xy_clean,100,3)
                else:
                    print("skiped")
                    continue

                image = convert_points_to_images(interpolation.T,gaussian_kernel=21,final_dimension=28,initial_dimension=128,line_thickness=1.5,border_thickness=3)



                new_row_data = {'letter':chr(letter_index+87),'gender': gender, 'hand': hand,"images":list(image.flatten()), 'X': list(interpolation[0]),'Y':list(interpolation[1]),'time':list(time),'pendown':list(pendown)}
                df = pd.concat([df, pd.DataFrame([new_row_data])],ignore_index=True)
    return df

df = pd.DataFrame(columns = ['letter','gender','hand','time','pendown','X','Y','images'])
for file in files:
    df = load_dataset(file,df)

df.to_csv(f'{preprocessed_datasets_path}/{save_file_name}', index=False) # save the dataframe