# Generates a memory usage report to infer one example

from memory_profiler import profile
import os
import sys

# Import necessary paths
absolute_root_path = os.path.dirname(os.path.realpath(__file__)).replace("\\","/") + "/.."
sys.path.insert(1,absolute_root_path)
from paths import save_weights_path

@profile
def main():
    from numpy import argmax,matmul,array,sin,cos,pi
    from pandas import read_csv
    from ahrs.filters import Madgwick
    from ahrs.common.orientation import q2rpy
    from keras.models import load_model

    # go to the correct directory to import utils
    sys.path.insert(1,f'{absolute_root_path}/Preprocessing')
    from utils import convert_points_to_images, remove_close_points, smoothen_points,inter,scale_and_center

    def rotation_matrix(theta):
        return array([[cos(theta),-sin(theta)],[sin(theta),cos(theta)]])

    # Load the specified try
    def load_try(try_name):
        # get the name of the model used for the try
        with open(f'{save_weights_path}/{try_name}/model_type.txt', newline='') as csvfile:
            model_name = csvfile.readline()

        # load the model
        model = load_model(f'{save_weights_path}/initialization/{model_name}.h5')

        # load weights from the try
        model.load_weights(f'{save_weights_path}/{try_name}/data.h5')

        print(f'Loading model {model_name} for try {try_name}')

        return model


    model_to_load = "61-LSTM-FT-1"#"57-CNN-FT-0"
    model_LSTM = True 

    # Load the model
    model = load_try(model_to_load)

    # transform the row data
    df = read_csv(f'{absolute_root_path}/Performance_evaluation/example_data.csv')
    acc_data = df[['acc_x', 'acc_y', 'acc_z']].values
    mag_data = df[['mag_x', 'mag_y', 'mag_z']].values
    gyro_data = df[['gyr_x', 'gyr_y', 'gyr_z']].values

    madgwick = Madgwick(acc=acc_data, gyr=gyro_data, mag=mag_data, frequency=27.0)
    angles = [q2rpy(i, in_deg = False) for i in madgwick.Q] # x,y,z = roll, pitch, yaw
    angles = array(angles)

    xy = angles[:,[2,1]] # x= pitch; y= yaw

    xy_shift = array([matmul(rotation_matrix(pi),xyi) for xyi in xy]) # put letters upside down

    xy_scaled = scale_and_center(xy_shift) # scale the values between 0 and 1 while keeping the aspect ratio + center the number

    # do the average and remove redundant points 
    xy_smoothed = smoothen_points(xy_scaled,3)
    xy_clean,_ = remove_close_points(xy_smoothed)
    interpolation = inter(xy_clean,100,3)

    # inference
    if model_LSTM:
        # shape correctly for the LSTM 
        interpolation = interpolation.transpose(1,0).reshape(1,100,2)
        predicated_value = chr(argmax(model.predict(interpolation))+97)
    else:
        image = convert_points_to_images(interpolation.T,gaussian_kernel=21,final_dimension=28,initial_dimension=128,line_thickness=1.5,border_thickness=3)
        image = image.reshape(1,28,28)
        predicated_value = chr(argmax(model.predict(image))+97)

    # print inferred letter
    print(predicated_value)

main() #run the encapsulated program to get memory data

# run with "python -m memory_profiler evaluate_performance.py" to get the memory report