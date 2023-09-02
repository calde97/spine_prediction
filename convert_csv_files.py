import os
from utils import get_df_frames_and_coordinates, get_data_for_animation, getxyz
import pandas as pd
# read all the files inside the folder

'''
Read all the data coming from the csv file inside AnimationData folder. We want to split the coordinates of each column
into 3 columns, one for each axis => Nose : (x;y;z) => Nose_x, Nose_y, Nose_z
'''


folder = 'AnimationData'
files = os.listdir(folder)
file_paths = [os.path.join(folder, file) for file in files]

df, frames, coordinates = get_df_frames_and_coordinates(file_paths[15])
data = get_data_for_animation(file_paths[15])
df_without_frames = df.drop(columns=['Frame'])


df_better = df.copy()
folder = 'AnimationDataNew'


for file_path, filename in zip(file_paths, files):
    df, frames, coordinates = get_df_frames_and_coordinates(file_path)
    df_better = df.copy()
    for joint in coordinates:
        df_joint = df[joint].apply(getxyz)
        df_joint = pd.DataFrame(df_joint.values.tolist(), columns=[f'{joint}_x', f'{joint}_y', f'{joint}_z'])
        df_better = pd.concat([df_better, df_joint], axis=1)
        # remove the joint from the df
        df_better = df_better.drop(columns=[joint])
        df_better.to_csv(os.path.join(folder, filename), index=False)


folder = 'AnimationDataNew'
files = os.listdir(folder)
file_paths = [os.path.join(folder, file) for file in files]
df_list = []
for file_path, filename in zip(file_paths, files):
    df, frames, coordinates = get_df_frames_and_coordinates(file_path)
    df_better = df.copy()
    df['id'] = filename
    df_list.append(df)

df = pd.concat(df_list)


df.to_csv('whole_data.csv', index=False)

