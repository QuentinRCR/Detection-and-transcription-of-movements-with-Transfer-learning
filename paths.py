import os

absolute_root_path = os.path.dirname(os.path.realpath(__file__)).replace("\\","/")

brush_path = f'{absolute_root_path}/../Datasets/BRUSH'
digilets_path = f'{absolute_root_path}/../DigiLeTs'
EMNIST_path = f'{absolute_root_path}/../Datasets/EMNIST'

smart_glove_df_path = f'{absolute_root_path}/Row_datasets/smart_glove'
preprocessed_datasets_path = f'{absolute_root_path}/../Previous_work/Personal_Experiments/Dataframes'
save_weights_path = f'{absolute_root_path}/../Previous_work/Personal_Experiments/ModelWeights'