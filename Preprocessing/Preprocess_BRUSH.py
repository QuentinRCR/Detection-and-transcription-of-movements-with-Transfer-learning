# Extraction of 2 instance of each letters per person from the BRUSH dataset and create the preprocessed file 

import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from utils import convert_points_to_images, inter, remove_close_points, smoothen_points,scale_and_center
import sys

# Import necessary paths
absolute_root_path = os.path.dirname(os.path.realpath(__file__)).replace("\\","/") + "/.."
sys.path.insert(1,absolute_root_path)
from paths import brush_path,preprocessed_datasets_path

save_file_name = "BRUSH-inter3--.csv"
source_dataset = ["BRUSH"]
network_type= ["LSTM"]

alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
# extract 2 examples from each letter from each person
try:
    df = pd.DataFrame(columns = ['letter','X','Y','image'])
    letter_count = {user_id: {letter: 0 for letter in alphabet} for user_id in os.listdir(brush_path)}
    for writer_id in os.listdir(brush_path):
        print("User id: ",writer_id)
        for drawing_id in os.listdir(f'{brush_path}/{writer_id}'):
            if("_" not in drawing_id):
                with open(f'{brush_path}/{writer_id}/{drawing_id}', 'rb') as f:
                    [sentence, drawing, label] = pickle.load(f)
                    Xs,Ys = np.array(drawing[:,:2]).T
                    for character_index,character in enumerate(sentence):
                        if character in alphabet:
                            if letter_count[writer_id][character]<2:
                                corresponding_indexes = np.where(label[:,character_index] == 1)
                                X,Y = Xs[corresponding_indexes],-Ys[corresponding_indexes]

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

                                letter_count[writer_id][character] += 1

                                # if letter_count[writer_id][character]==2:
                                #      print("COunt reached for letter: ",character)

                                if all(count == 2 for count in letter_count[writer_id].values()):
                                    print(f"Skipping user {writer_id}")
                                    break  # Skip to the next user

                                # image = convert_points_to_images(interpolation.T,gaussian_kernel=21,final_dimension=28,initial_dimension=128,line_thickness=1.5,border_thickness=3)
                                new_row_data = {'letter':character, 'X': list(interpolation[0]),'Y':list(interpolation[1])}
                                df = pd.concat([df, pd.DataFrame([new_row_data])],ignore_index=True)

                        if all(count == 2 for count in letter_count[writer_id].values()):
                                print(f"Skipping user {writer_id}")
                                break  # Skip to the next user
                        
                    if all(count == 2 for count in letter_count[writer_id].values()):
                                print(f"Skipping user {writer_id}")
                                break  # Skip to the next user

finally:
    df.to_csv(f'{preprocessed_datasets_path}/{save_file_name}', index=False) # save the dataframe