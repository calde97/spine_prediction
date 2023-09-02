import pandas as pd
import random

from models import heuristic_model

df = pd.read_csv('AnimationDataNew/whole_data.csv')
ids = df['id'].unique()

# divide the ids into train, test and validation. set a fixed seed for reproducibility

random.seed(42)
random.shuffle(ids)

train_id = ids[:int(0.8 * len(ids))]
test_id = ids[int(0.8 * len(ids)):int(0.9 * len(ids))]
validation_id = ids[int(0.9 * len(ids)):]

train = df[df['id'].isin(train_id)]
test = df[df['id'].isin(test_id)]
validation = df[df['id'].isin(validation_id)]

#%%

# save the dataframes into csv files
path = 'preprocess_data/'

train.to_csv(path + 'train.csv', index=False)
test.to_csv(path + 'test.csv', index=False)
validation.to_csv(path + 'validation.csv', index=False)

#%%

train = train[train['id'] == 'MOB_Idle_To_Run_R180_Fwd-h_180-Positions.csv']
x, y, z = heuristic_model(train)

#%%

# calculate the MSE between spine and spine_p, spine2 and spine2_p, spine3 and spine3_p
from sklearn.metrics import mean_squared_error
import numpy as np

mse_spine_x = mean_squared_error(train['Spine_x'], train['Spine_x'])
mse_spine_y = mean_squared_error(train['Spine_y'], train['Spine_y'])
mse_spine_z = mean_squared_error(train['Spine_z'], train['Spine_z'])
average_mse_spine = (mse_spine_x + mse_spine_y + mse_spine_z) / 3
print(f'Average MSE for Spine: {average_mse_spine}')
#%%
from sklearn.metrics import mean_squared_error
mse = 0
targets_label = ['Spine', 'Spine1' ,'Spine2', 'Spine3']
for x_spine, y_spine, z_spine, label in zip(x, y, z, targets_label):
    mean_squared_error_x = mean_squared_error(train[label+'_x'], x_spine)
    mse += mean_squared_error_x
    mean_squared_error_y = mean_squared_error(train[label+'_y'], y_spine)
    mse += mean_squared_error_y
    mean_squared_error_z = mean_squared_error(train[label+'_z'], z_spine)
    mse += mean_squared_error_z




