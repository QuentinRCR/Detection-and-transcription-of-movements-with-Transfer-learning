# Extraction of each letter of the multi-user dataset and create the preprocessed file 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick
from ahrs.common.orientation import q2rpy
import json
from utils import convert_points_to_images, remove_close_points, smoothen_points,inter,scale_and_center
import sys
import os

# Import necessary paths
absolute_root_path = os.path.dirname(os.path.realpath(__file__)).replace("\\","/") + "/.."
sys.path.insert(1,absolute_root_path)
from paths import preprocessed_datasets_path, smart_glove_df_path

save_file_name = 'multi_user-inter3-28-21.csv'

# load the dataset
def load_dataset(train_dataset_path):
    train_df = pd.read_csv(train_dataset_path)

    # Convert'features' from string to JSON
    train_df['features'] = train_df['features'].str.replace("'", '"')
    train_df['features'] = train_df['features'].apply(json.loads)

    return train_df

def rotation_matrix(theta):
    return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

def get_angles(feature_df):
        # Extract accelerometer and gyroscope data
        acc_data = feature_df[['acc_x', 'acc_y', 'acc_z']].values
        mag_data = feature_df[['mag_x', 'mag_y', 'mag_z']].values
        gyro_data = feature_df[['gyr_x', 'gyr_y', 'gyr_z']].values

        madgwick = Madgwick(acc=acc_data, gyr=gyro_data, mag=mag_data, frequency=27.0)
        angles = [q2rpy(i, in_deg = False) for i in madgwick.Q] # x,y,z = roll, pitch, yaw
        angles = np.array(angles)
        return angles

def rotate_image(angles,xy):
     return np.array([np.matmul(rotation_matrix(angles),xyi) for xyi in xy])

def main():
    train_dataset_path = f'{smart_glove_df_path}/multiusers.csv'
    axes_to_remove = ['features']

    df = load_dataset(train_dataset_path)

    images = []
    X=[]
    Y=[]
    for _, df_row in df.iterrows():

        feature_df = pd.DataFrame(df_row['features'])
        angles = get_angles(feature_df)

        xy = angles[:,[2,1]]

        xy_shift = rotate_image(np.pi,xy)

        # scale the values between 0 and 1 while keeping the aspect ratio + center the number
        xy_scaled = scale_and_center(xy_shift)

        xy_smoothed = smoothen_points(xy_scaled,3)
        xy_clean,_ = remove_close_points(xy_smoothed)
        interpolation = inter(xy_clean,100,3)

        image = convert_points_to_images(interpolation.T,gaussian_kernel=21,final_dimension=28,initial_dimension=128,line_thickness=1.5,border_thickness=3)


        X.append(list(interpolation[0]))
        Y.append(list(interpolation[1]))
        images.append(list(image.flatten()))

    df['images'] = images # add images to the dataframe
    df['X'] = X # add images to the dataframe
    df['Y'] = Y # add images to the dataframe

    df= df.drop(axes_to_remove,axis=1) # remove unnecessary data

    # Save csv with images
    df.to_csv(f'{preprocessed_datasets_path}/{save_file_name}', index=False) # save the dataframe

if __name__ == "__main__":
    main()