from mpl_toolkits.mplot3d.art3d import Line3D
import numpy as np
import pandas as pd

LINKS = [
    ('ShoulderLeft', 'ElbowLeft'), ('ElbowLeft', 'WristLeft'),
    ('ShoulderRight', 'ElbowRight'), ('ElbowRight', 'WristRight'),
    ('HipLeft', 'KneeLeft'), ('KneeLeft', 'AnkleLeft'), ('AnkleLeft', 'ToesLeft'),
    ('HipRight', 'KneeRight'), ('KneeRight', 'AnkleRight'), ('AnkleRight', 'ToesRight'),
    ('HipLeft', 'HipRight')
]

IMPORTANT_NODES_WITHOUT_SPINE = ['ShoulderLeft', 'ShoulderRight', 'HipLeft', 'HipRight',
                                 'KneeLeft', 'KneeRight', 'AnkleLeft', 'AnkleRight',
                                 'WristLeft', 'WristRight', 'ElbowLeft', 'ElbowRight',
                                 'Nose', 'ToesLeft', 'ToesRight']

ONLY_SPINE_JOINTS = ['Spine', 'Spine1', 'Spine2', 'Spine3']

ALL_IMPORTANT_JOINTS = IMPORTANT_NODES_WITHOUT_SPINE + ONLY_SPINE_JOINTS

PREDICTION_JOINTS = ['Spine1_prediction', 'Spine2_prediction', 'Spine3_prediction', 'Spine4_prediction']



def getxyz(string_cordinates):
    coordinates = string_cordinates.split(';')
    return float(coordinates[0]), float(coordinates[1]), float(coordinates[2])


def calculate_spine(data_from_single_frame):
    mid_shoulder_x = (data_from_single_frame['ShoulderLeft'][0] + data_from_single_frame['ShoulderRight'][0]) / 2
    mid_shoulder_y = (data_from_single_frame['ShoulderLeft'][1] + data_from_single_frame['ShoulderRight'][1]) / 2
    mid_shoulder_z = (data_from_single_frame['ShoulderLeft'][2] + data_from_single_frame['ShoulderRight'][2]) / 2

    mid_hip_x = (data_from_single_frame['HipLeft'][0] + data_from_single_frame['HipRight'][0]) / 2
    mid_hip_y = (data_from_single_frame['HipLeft'][1] + data_from_single_frame['HipRight'][1]) / 2
    mid_hip_z = (data_from_single_frame['HipLeft'][2] + data_from_single_frame['HipRight'][2]) / 2

    delta_x = (mid_shoulder_x - mid_hip_x) / 5
    delta_y = (mid_shoulder_y - mid_hip_y) / 5
    delta_z = (mid_shoulder_z - mid_hip_z) / 5

    spine_xs = []
    spine_ys = []
    spine_zs = []

    for i in range(1, 5):
        spine_xs.append(mid_hip_x + delta_x * i)
        spine_ys.append(mid_hip_y + delta_y * i)
        spine_zs.append(mid_hip_z + delta_z * i)

    return spine_xs, spine_ys, spine_zs



def add_links_df(ax, df, links):
    for link in links:
        starting_node = link[0] + '_x', link[0] + '_y', link[0] + '_z'
        ending_node = link[1] + '_x', link[1] + '_y', link[1] + '_z'
        starting_x = df[starting_node[0]].values[0]
        starting_y = df[starting_node[1]].values[0]
        starting_z = df[starting_node[2]].values[0]
        ending_x = df[ending_node[0]].values[0]
        ending_y = df[ending_node[1]].values[0]
        ending_z = df[ending_node[2]].values[0]
        line_segment = Line3D([starting_x, ending_x],
                              [starting_y, ending_y],
                              [starting_z, ending_z],
                              color='blue', linewidth=2, label='Line')
        ax.add_line(line_segment)

# Obsolete function. Used before the data was handled in a dictionary
def add_links(ax, links, joints):
    for link in links:
        line_start = joints[link[0]]
        line_end = joints[link[1]]
        line_segment = Line3D([line_start[0], line_end[0]],
                              [line_start[1], line_end[1]],
                              [line_start[2], line_end[2]],
                              color='red', linewidth=2, label='Line')
        ax.add_line(line_segment)


def normalize_data(data):
    # get all the values of a dictionary
    values = []
    for frame in data.keys():
        for joint in data[frame].keys():
            values.extend(data[frame][joint])
    # flatten the list
    # get the maximum value of the dictionary
    max_value = max(values)
    # get the minimum value of the dictionary
    min_value = min(values)
    print(max_value, min_value)
    # get the difference between the maximum and minimum value
    difference = max_value - min_value

    for frame in data:
        for joint in data[frame]:
            data[frame][joint] = list(((np.array(data[frame][joint]) - min_value) / difference))

    return data


def unnormalize_data(data):
    # get all values in a single list from dictionary of dictionary
    values = [item for sublist in data.values() for item in sublist.values()]

    # get all the values of a dictionary
    # flatten the list
    values = [item for sublist in values for item in sublist]
    # get the maximum value of the dictionary
    max_value = max(values)
    # get the minimum value of the dictionary
    min_value = min(values)
    # get the difference between the maximum and minimum value
    difference = max_value - min_value
    # normalize the data
    for data_single_frame in data:
        for key in data_single_frame:
            data_single_frame[key] = list(((np.array(data_single_frame[key]) * difference) + min_value))
    return data

def get_data_for_animation(filename):
    df = pd.read_csv(filename)
    columns = df.columns
    coordinates = columns[1:]
    frames = df['Frame'].unique()
    data = {}
    joints = {}

    for i in range(len(frames)):
        for coordinate in coordinates:
            x, y, z = getxyz(df[coordinate][i])
            joints[coordinate] = [x, y, z]
        data[i] = joints.copy()

    return data


def get_df_frames_and_coordinates(filename):
    df = pd.read_csv(filename)
    columns = df.columns
    coordinates = columns[1:]
    frames = df['Frame'].unique()
    return df, frames, coordinates

