import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3D
from utils import add_links, ALL_IMPORTANT_JOINTS, ONLY_SPINE_JOINTS, IMPORTANT_NODES_WITHOUT_SPINE, LINKS, \
    add_links_df, PREDICTION_JOINTS
import os


def plot_data_4_views(x=None, y=None, z=None, queried_df=None,
                      x_p=None, y_p=None, z_p=None, elevations=None,
                      azimuths=None, vertical_axis='y',
                      folder=None, name='test', min=None, max=None):
    # Create a 2x2 grid of subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 8), subplot_kw={'projection': '3d'})

    # Iterate through subplots and customize each one
    for i, ax in enumerate(axs.flatten()):
        ax.scatter(x, y, z, c='blue', marker='o')  # Scatter plot your data here
        add_links_df(ax, queried_df, LINKS)
        ax.scatter(x_p, y_p, z_p, c='red', marker='o')  # Scatter plot your data here

        azimuth = azimuths[i]
        elevation = elevations[i]

        ax.view_init(elev=elevation, azim=azimuth, vertical_axis=vertical_axis)  # Set azimuth and elevation

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.set_xlim(min[0] - 0.1 * abs(min[0]), max[0] + 0.1 * abs(max[0]))
        ax.set_ylim(min[1] - 0.1 * abs(min[1]), max[1] + 0.1 * abs(max[1]))
        ax.set_zlim(min[2] - 0.1 * abs(min[2]), max[2] + 0.1 * abs(max[2]))

    # Adjust layout and display
    plt.tight_layout()

    if folder is None:
        plt.show()
    else:
        path = os.path.join('images', folder)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, f'{name}.png'))


def plot_joints_and_links_df(x=None, y=None, z=None, queried_df=None, normal_flag=False, elevation=0,
                             azimuth=0, roll=0, vertical_axis='y',
                             folder=None, name='test'):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot the dots
    # ax.scatter(spine_x, spine_y, spine_z, c='blue', marker='o', label='Dots')
    chosen_data = ALL_IMPORTANT_JOINTS

    ax.scatter(x, y, z, c='red', marker='o')
    add_links_df(ax, queried_df, LINKS)

    ax.view_init(elev=elevation, azim=azimuth, roll=roll, vertical_axis=vertical_axis)  # Adjust these angles as needed

    # do not show axes labels and ticks

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.set_xlim(0, 1)  # Adjust xmin and xmax as needed
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    if normal_flag == False:
        ax.set_xlim(-2.5, 5.5)  # Adjust xmin and xmax as needed
        ax.set_ylim(0, 2.5)
        ax.set_zlim(-3, 8)

    if folder is None:
        plt.show()
    else:
        path = os.path.join('images', folder)
        if not os.path.exists(path):
            os.makedirs(path)
        plt.savefig(os.path.join(path, f'{name}.png'))
