# -----------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------

import pandas as pd
import gzip
import json
import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.stats import gaussian_kde
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import networkx as nx
import torch
from torch import nn
from sklearn.model_selection import KFold
import torch.optim as optim
import optuna
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader
# from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version())

#pd.set_option("display.max_colwidth", None)
#pd.set_option("display.max_colwidth", 50)

sns.set_theme(style = "whitegrid")

# Pre-processing ----------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Prep 1
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# User data Pre-processing
# -----------------------------------------------------------------------

# user_df = pd.read_json("../../data/raw/2019-12-16-latest-users.json.gz", lines = True)
# user_df.to_csv("../../data/interim/2019-12-16-latest-users.csv", index = None)
# user_df = pd.read_csv("../../data/interim/2019-12-16-latest-users.csv")

# user_df
# user_df.info()

# -----------------------------------------------------------------------
# Test
# -----------------------------------------------------------------------

# # Reads x rows of radial data (just for exploration)
# radial_df = pd.read_json("../../data/raw/2019-12-16-latest-radial.json.gz", lines = True, nrows = 100)
# radial_df['level_id'] = radial_df['level_instance_json'].apply(lambda x: x['meta']['level_id'])
# radial_df_filtered = radial_df[radial_df['level_id'] == 56]
# # Reads x rows of level data (just for exploration)
# level_df = pd.read_json("../../data/raw/2019-12-16-latest-level.json.gz", lines = True, nrows = 20)
# level_df.head(1)
# level_df.info()

# -----------------------------------------------------------------------
# Level 56 Data Extraction and Export to csv
# -----------------------------------------------------------------------

# input_path = "../../data/raw/2019-12-16-latest-level.json.gz"
# output_path = "../../data/interim/2019-12-16-latest-level-56-filtered.csv"

# with gzip.open(input_path, 'rt') as file:
#     with open(output_path, 'w', newline = '') as csvfile:
#         writer = csv.writer(csvfile)
        
#         headers = ['uuid', 'user_id', 'stored_at', 'duration',  
#                    'previous_attempts', 'early_termination', 'platform', 
#                    'app_version', 'map_view_duration', 'events', 'player']
        
#         writer.writerow(headers)
        
#         for line in file:
#             obj = json.loads(line.strip())
            
#             level_instance_json = obj.get('level_instance_json')
#             if level_instance_json is None:
#                 continue
            
#             meta = level_instance_json.get('meta')
#             if meta is None:
#                 continue
            
#             level_id = obj.get('level_instance_json', {}).get('meta', {}).get('level_id')
            
#             if level_id == 56:
#                 meta = obj.get('level_instance_json', {}).get('meta', {})
#                 #player = obj.get('level_instance_json', {}).get('player', {}).get('0', {})
                
#                 data = [
#                     obj.get('uuid'),
#                     obj.get('user_id'),
#                     obj.get('stored_at'),
#                     obj.get('duration'),
#                     meta.get('previous_attempts'),
#                     meta.get('early_termination'),
#                     meta.get('platform'),
#                     meta.get('app_version'),
#                     meta.get('map_view_duration'),
#                     json.dumps(obj.get('level_instance_json', {}).get('events', {})),
#                     json.dumps(obj.get('level_instance_json', {}).get('player', {}))
#                 ]
#                 writer.writerow(data)

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Pre-processing 2
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Load Data
# -----------------------------------------------------------------------

level_df = pd.read_csv("../../data/interim/2019-12-16-latest-level-56-filtered.csv")
level_df.head(); level_df.count(); level_df.info()

duplicates_uuid = level_df['uuid'].duplicated().sum() #0 duplicated uuid
duplicates_user = level_df['user_id'].duplicated().sum() #13396 duplicated instances of user_id

# -----------------------------------------------------------------------
# Extracts out x,y,r values from player column into new cols
# -----------------------------------------------------------------------

def get_xyr_values(player_str, key):
    try:
        player_data = json.loads(player_str)
        return [entry[key] for entry in player_data.values()]
    except:
        return []

level_df['x'] = level_df['player'].apply(lambda x: get_xyr_values(x, 'x'))
level_df['y'] = level_df['player'].apply(lambda x: get_xyr_values(x, 'y'))
level_df['r'] = level_df['player'].apply(lambda x: get_xyr_values(x, 'r'))
#level_df[['x', 'y', 'r']].head()

# -----------------------------------------------------------------------
# Removes rows with an x or y array size of 0
# -----------------------------------------------------------------------

mask = (level_df['x'].apply(len) > 0) & (level_df['y'].apply(len) > 0)
count_row = level_df.shape[0]
level_df = level_df[mask]
count_row = level_df.shape[0]

mean_xlength = level_df['x'].apply(len).mean()
mean_ylength = level_df['y'].apply(len).mean()
mean_rlength = level_df['r'].apply(len).mean()
count_401 = level_df[level_df['x'].apply(len) == 401].shape[0]

# -----------------------------------------------------------------------
# Loads demographic data
# -----------------------------------------------------------------------

user_df = pd.read_csv("../../data/interim/2019-12-16-latest-users.csv")
user_df.head(); user_df.count(); user_df.info() #3,936,867

# -----------------------------------------------------------------------
# Combines level data + demographic data
# -----------------------------------------------------------------------

level_df = pd.merge(level_df, user_df, left_on = 'user_id', right_on = 'uuid', how = 'left')

# level_df.count() #80455
# level_df.info()
# level_df.head(3)

# -----------------------------------------------------------------------
# Removes participants...
# -----------------------------------------------------------------------

#1 ...without demographic info

level_df_filtered = level_df.dropna(subset = ['age', 'gender', 'sleep'])

#2 ...who have sleep data over 3 s.d. away from the mean

sleep_mean = level_df_filtered['sleep'].mean()
sleep_std = level_df_filtered['sleep'].std()
suspicious_sleepers = (level_df_filtered['sleep'] > sleep_mean + (3 * sleep_std)) | (level_df_filtered['sleep'] < sleep_mean - (3 * sleep_std))
level_df_filtered = level_df_filtered[~suspicious_sleepers]

#3 ...aged between 16-18 or 70-99

plt.figure(figsize = (12, 6))
ax = sns.countplot(x = 'age', data = level_df_filtered, palette = 'viridis')
plt.title('Distribution of the demographic variable: Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
labels = [item.get_text() for item in ax.get_xticklabels()]
n = 5
for i, label in enumerate(labels):
    if i % n == 0:
        labels[i] = str(int(float(label)))
    else:
        labels[i] = ''
ax.set_xticklabels(labels, rotation = 45)
plt.tight_layout()
plt.show()

level_df_filtered = level_df_filtered[(level_df_filtered.age > 18) & (level_df_filtered.age < 70)]

#4 ...duration 3 s.d. away from the mean #distribution of trajectory lengths and description stats

duration_mean = level_df['duration'].mean()
duration_std = level_df['duration'].std() #O.G. dataset 3 s.d. + mean = 613.61
level_df_filtered = level_df_filtered[level_df_filtered.duration < duration_mean + (3 * duration_std)]
#level_df_filtered.count() #35916

#5 ...with previous attempts > 1 ???

plt.figure(figsize = (12, 6))
sns.countplot(x = 'previous_attempts', data = level_df_filtered, palette = 'viridis')
plt.title('Distribution of previous attempts')
plt.xlabel('Number of Previous Attempts')
plt.ylabel('Frequency')
plt.show()

level_df_filtered = level_df_filtered[level_df_filtered.previous_attempts == 1]

#6 ...assigned other on gender (simplicity of model + only 1 person put other)

level_df_filtered = level_df_filtered[level_df_filtered['gender'].isin(['m', 'f'])]

#7 ...early termination = True

level_df_filtered = level_df_filtered[level_df_filtered.early_termination == False] #0

#level_df_filtered.count() #26608

# -----------------------------------------------------------------------
# Removes columns not interested in
# -----------------------------------------------------------------------

col_drop = ['stored_at_x', 'stored_at_y', 'uuid_y', 'previous_attempts', 'early_termination', 'platform', 
            'app_version', 'activity_recent', 'activityrecent', 'education', 'hand', 'home_environment', 
            'location', 'navigating_skills', 'travel_time', 'sleep']

level_df_filtered = level_df_filtered.drop(columns = col_drop)
#level_df_filtered.head(1)

# -----------------------------------------------------------------------
# Convert continuous demographic variables to categorical/ordinal -> Age discretization
# -----------------------------------------------------------------------

# #1... 3 equal age bins
# age_bins = [19, 37, 55, 70]
# age_labels = ['19-36', '37-54', '55-70']
# level_df_filtered['age_bins'] = pd.cut(level_df_filtered['age'], bins = age_bins, labels = age_labels, right = False)

#2... vectorised age by 5 equally spread gaussians

k = 5
age_min = 19
age_max = 69
means = np.linspace(age_min, age_max, k)
std_dev = ((age_max - age_min) / (k * 3)) * 1.6

def wiggly_gaussian_pdfs(x, mean, std_dev):
    peak_density = norm.pdf(mean, mean, std_dev)
    return norm.pdf(x, mean, std_dev) / peak_density

def map_age_to_wigglevector(age, means, std_dev):
    return np.array([wiggly_gaussian_pdfs(age, mean, std_dev) for mean in means])

level_df_filtered['age_vector'] = level_df_filtered['age'].apply(map_age_to_wigglevector, means = means, std_dev = std_dev)
level_df_filtered['age_vector'] = level_df_filtered['age_vector'].apply(lambda x: x / x.sum())

x = np.linspace(age_min, age_max, 1000)
plt.figure(figsize = (10, 3))
for mean in means:
    plt.plot(x, wiggly_gaussian_pdfs(x, mean, std_dev), label = f'Mean: {mean:.1f}')
plt.title('Gaussian Distributions of age (PPD of 1)')
plt.xlabel('Age')
plt.ylabel('Probability Density')
plt.legend(loc = 'lower right')
plt.grid(True)
plt.show()

#level_df_filtered.head(1)

#... assign each trajectory an age bin based on the probabilities
def wigglyvector_to_agebin(prob_vector):
    prob_vector = prob_vector / np.sum(prob_vector)
    cat = np.random.choice(a = np.arange(len(prob_vector)), p = prob_vector)
    return f'age_{cat}'

np.random.seed(7)

level_df_filtered['age_bins'] = level_df_filtered['age_vector'].apply(wigglyvector_to_agebin)
#level_df_filtered.head(5)

#4... check distributions

#age
plt.figure(figsize = (12, 6))
sns.countplot(x = 'age', data = level_df_filtered, palette = 'viridis')
plt.title('Distribution of the demographic variable: Age')
plt.xticks(rotation = 90)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# age bins
plt.figure(figsize = (12, 6))
sns.countplot(x = 'age_bins', data = level_df_filtered, palette = 'viridis', order = sorted(level_df_filtered['age_bins'].unique()))
plt.title('Distribution of Age Bins')
plt.xlabel('Age Bins')
plt.ylabel('Frequency')
plt.show()

# Plot for 'gender'
plt.figure(figsize = (12, 6))
sns.countplot(x = 'gender', data = level_df_filtered, palette = 'viridis')
plt.title('Distribution of Gender')
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.show()

# -----------------------------------------------------------------------
# Create label column using age and gender
# -----------------------------------------------------------------------

level_df_filtered['label'] = level_df_filtered.apply(lambda row: f"{row['gender']}-{row['age_bins']}", axis = 1)

label_counts = level_df_filtered['label'].value_counts().reset_index()
label_counts.columns = ['label', 'frequency']

#fixin order for plot
sort_order = {f"f-age_{i}": i for i in range(5)}
sort_order.update({f"m-age_{i}": i+5 for i in range(5)})
label_counts['key_fixing_order'] = label_counts['label'].map(sort_order)
label_counts_sorted = label_counts.sort_values(by = ['key_fixing_order']).reset_index(drop = True)

plt.figure(figsize = (10, 6))
sns.barplot(x = 'label', y = 'frequency', data = label_counts_sorted, palette = 'viridis')
plt.xticks(rotation = 45)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Distribution of Combined Age-Gender Labels')
plt.show()

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Pre-processing 3
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Reconstruct trajectories
# -----------------------------------------------------------------------

#1... Calculated cos(theta) and sin(theta) as a store of directional information

def conjure_directioncomps(row):
    x_list = row['x']
    y_list = row['y']
    cos_theta_list = []
    sin_theta_list = []

    for i in range(len(x_list) - 1):
        delta_x = x_list[i+1] - x_list[i]
        delta_y = y_list[i+1] - y_list[i]
        hypotenuse = np.sqrt(delta_x**2 + delta_y**2)

        if hypotenuse == 0:
            cos_theta = 0
            sin_theta = 0
        else:
            cos_theta = delta_x / hypotenuse
            sin_theta = delta_y / hypotenuse

        cos_theta_list.append(cos_theta)
        sin_theta_list.append(sin_theta)
    cos_theta_list.append(cos_theta_list[-1])
    sin_theta_list.append(sin_theta_list[-1])
    
    return pd.Series([cos_theta_list, sin_theta_list], index = ['cos_theta', 'sin_theta'])

level_df_filtered[['cos_theta', 'sin_theta']] = level_df_filtered.apply(conjure_directioncomps, axis = 1)
# level_df_filtered[['cos_theta', 'sin_theta']].head(3)

#2... smooths the cos(theta) and sin(theta)

def gaussian_smooooooth(column):
    list_of_data = column
    smoothed = gaussian_filter1d(np.array(list_of_data), sigma = 2)
    return smoothed.tolist()

level_df_filtered['smoothed_cos_theta'] = level_df_filtered['cos_theta'].apply(gaussian_smooooooth)
level_df_filtered['smoothed_sin_theta'] = level_df_filtered['sin_theta'].apply(gaussian_smooooooth)
#level_df_filtered[['smoothed_cos_theta', 'smoothed_sin_theta']].head(3)

#plot example first trajectory
cos_theta_first_trajectory = level_df_filtered.iloc[0]['cos_theta']
smoothed_cos_theta_first_trajectory = level_df_filtered.iloc[0]['smoothed_cos_theta']
t = list(range(len(cos_theta_first_trajectory)))
plt.figure(figsize = (10, 4))
plt.plot(t, cos_theta_first_trajectory, label = 'cos(θ)', color = 'green', alpha = 0.5)
plt.plot(t, smoothed_cos_theta_first_trajectory, label = 'reconstructed cos(θ)', color = 'orange')
plt.xlabel('t')
plt.ylabel('cos(θ)')
plt.title('Example Trajectory: Reconstructed cos(θ) by guassian smoothing (sigma = 2)')
plt.legend()
plt.show()

sin_theta_first_trajectory = level_df_filtered.iloc[0]['sin_theta']
smoothed_sin_theta_first_trajectory = level_df_filtered.iloc[0]['smoothed_sin_theta']
t = list(range(len(sin_theta_first_trajectory)))
plt.figure(figsize = (10, 4))
plt.plot(t, sin_theta_first_trajectory, label = 'sin(θ)', color = 'green', alpha = 0.5)
plt.plot(t, smoothed_sin_theta_first_trajectory, label = 'reconstructed sin(θ)', color = 'orange')
plt.xlabel('t')
plt.ylabel('sin(θ)')
plt.title('Example Trajectory: Reconstructed sin(θ) by guassian smoothing (sigma = 2)')
plt.legend()
plt.show()

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Pre-processing 4
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Resampling by cubic interpolation
# -----------------------------------------------------------------------

x_len = level_df_filtered['x'].apply(len)
plt.figure(figsize = (12, 6))
sns.histplot(x_len, bins = 100)
plt.xlim(0, 2000)
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.title('Distribution of the Length of all Trajectories')
plt.grid(True)
plt.show()
x_len.describe() #75% ss = 523

samplesize = 523

def resample_cubinterp(trajectory):
    samp = np.linspace(0, len(trajectory) - 1, len(trajectory))
    resamp = np.linspace(0, len(trajectory) - 1, samplesize)
    interp = interp1d(samp, trajectory, kind = 'cubic')
    return interp(resamp).tolist()

level_df_filtered['x_resampled'] = level_df_filtered['x'].apply(resample_cubinterp)
level_df_filtered['y_resampled'] = level_df_filtered['y'].apply(resample_cubinterp)
level_df_filtered['smoothed_cos_theta_resampled'] = level_df_filtered['smoothed_cos_theta'].apply(resample_cubinterp)
level_df_filtered['smoothed_sin_theta_resampled'] = level_df_filtered['smoothed_sin_theta'].apply(resample_cubinterp)
#level_df_filtered[['x_resampled', 'y_resampled', 'smoothed_cos_theta_resampled', 'smoothed_sin_theta_resampled']].head().applymap(lambda x: x[:5])

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Pre-processing Visualisations
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Plot example on 2 different trajectories
# -----------------------------------------------------------------------

x_coords = level_df_filtered['x'].iloc[0]
y_coords = level_df_filtered['y'].iloc[0]

x_coords2 = level_df_filtered['x'].iloc[11]
y_coords2 = level_df_filtered['y'].iloc[11]

fig, axs = plt.subplots(1, 2, figsize = (12, 8))

axs[0].plot(x_coords, y_coords, '-o', color = 'midnightblue', linewidth = 2, markersize = 6, markeredgecolor = 'midnightblue')
axs[0].set_title('Example Trajectory A')
axs[0].set_xlabel('x', fontsize = 14, fontweight = 'bold')
axs[0].set_ylabel('y', fontsize = 14, fontweight = 'bold')
axs[0].grid(True, which = 'both', linestyle = '--', linewidth = 0.5, color = 'grey')
axs[0].minorticks_on()
axs[0].tick_params(axis = 'both', which = 'major', labelsize = 12, colors = 'black')
axs[0].xaxis.set_major_locator(MultipleLocator(5))
axs[0].yaxis.set_major_locator(MultipleLocator(5))
axs[0].set_aspect('equal', adjustable = 'box')

axs[1].plot(x_coords2, y_coords2, '-o', color = 'darkred', linewidth = 2, markersize = 6, markeredgecolor = 'darkred')
axs[1].set_title('Example Trajectory B')
axs[1].set_xlabel('x', fontsize = 14, fontweight = 'bold')
axs[1].set_ylabel('y', fontsize = 14, fontweight = 'bold')
axs[1].grid(True, which = 'both', linestyle = '--', linewidth = 0.5, color = 'grey')
axs[1].minorticks_on()
axs[1].tick_params(axis = 'both', which = 'major', labelsize = 12, colors = 'black')
axs[1].xaxis.set_major_locator(MultipleLocator(5))
axs[1].yaxis.set_major_locator(MultipleLocator(5))
axs[1].set_aspect('equal', adjustable = 'box')

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# Plots to show the cubic interpolation extreme min and max cases
# -----------------------------------------------------------------------

#find shortest trajectory
min_len_x = level_df_filtered['x'].apply(len).min()
trajectories_with_min_len_x = level_df_filtered[level_df_filtered['x'].apply(len) == min_len_x]
shortest_trajectory_loc = trajectories_with_min_len_x.index[0] #59097

x_short = level_df_filtered['x'][shortest_trajectory_loc]
y_short = level_df_filtered['y'][shortest_trajectory_loc]
x_resampled_short = level_df_filtered['x_resampled'][shortest_trajectory_loc]
y_resampled_short = level_df_filtered['y_resampled'][shortest_trajectory_loc]

#find longest trajectory
max_len_x = level_df_filtered['x'].apply(len).max()
trajectories_with_max_len_x = level_df_filtered[level_df_filtered['x'].apply(len) == max_len_x]
longest_trajectory_loc = trajectories_with_max_len_x.index[0] #34405

x_long = level_df_filtered['x'][longest_trajectory_loc]
y_long = level_df_filtered['y'][longest_trajectory_loc]
x_resampled_long = level_df_filtered['x_resampled'][longest_trajectory_loc]
y_resampled_long = level_df_filtered['y_resampled'][longest_trajectory_loc]

fig, ax = plt.subplots(2, 2, figsize = (8, 12))

#... original shortest trajectory
ax[0, 0].plot(x_short, y_short, '-o', markersize = 4, label = 'Original')
ax[0, 0].set_title('Original Shortest Player Trajectory')
ax[0, 0].set_xlabel('x')
ax[0, 0].set_ylabel('y')
ax[0, 0].grid(True)
ax[0, 0].set_aspect('equal', adjustable = 'box')
ax[0, 0].legend()

#... resampled shortest trajectory
ax[0, 1].plot(x_resampled_short, y_resampled_short, '-o', markersize = 4, label = 'Resampled', color = 'orange')
ax[0, 1].set_title('Resampled Shortest Player Trajectory')
ax[0, 1].set_xlabel('x')
ax[0, 1].set_ylabel('y')
ax[0, 1].grid(True)
ax[0, 1].set_aspect('equal', adjustable = 'box')
ax[0, 1].legend()

#... original longest trajectory
ax[1, 0].plot(x_long, y_long, '-o', markersize = 4, label = 'Original')
ax[1, 0].set_title('Original Longest Player Trajectory')
ax[1, 0].set_xlabel('x')
ax[1, 0].set_ylabel('y')
ax[1, 0].grid(True)
ax[1, 0].set_aspect('equal', adjustable = 'box')
ax[1, 0].legend()

#... resampled longest trajectory
ax[1, 1].plot(x_resampled_long, y_resampled_long, '-o', markersize = 4, label = 'Resampled', color = 'orange')
ax[1, 1].set_title('Resampled Longest Player Trajectory')
ax[1, 1].set_xlabel('x')
ax[1, 1].set_ylabel('y')
ax[1, 1].grid(True)
ax[1, 1].set_aspect('equal', adjustable = 'box')
ax[1, 1].legend()

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# Plot example on first trajectories x and y independently
# -----------------------------------------------------------------------

x_original = level_df_filtered['x'][longest_trajectory_loc]
y_original = level_df_filtered['y'][longest_trajectory_loc]

x_resampled = level_df_filtered['x_resampled'][longest_trajectory_loc]
y_resampled = level_df_filtered['y_resampled'][longest_trajectory_loc]

fig, ax = plt.subplots(2, 2, figsize = (12, 12))

#... x original vs resampled
ax[0, 0].plot(x_original, label = 'Original')
ax[0, 0].set_title('Example Shortest Trajectory: x - Original')
ax[0, 0].set_xlabel('t')
ax[0, 0].set_ylabel('x')

ax[0, 1].plot(x_resampled, label = 'Resampled', color = 'orange')
ax[0, 1].set_title('Example Shortest Trajectory: x - Resampled')
ax[0, 1].set_xlabel('t')
ax[0, 1].set_ylabel('x')

#... y original vs resampled
ax[1, 0].plot(y_original, label = 'Original')
ax[1, 0].set_title('Example Shortest Trajectory: y - Original')
ax[1, 0].set_xlabel('t')
ax[1, 0].set_ylabel('y')

ax[1, 1].plot(y_resampled, label = 'Resampled', color = 'orange')
ax[1, 1].set_title('Example Shortest Trajectory: y - Resampled')
ax[1, 1].set_xlabel('t')
ax[1, 1].set_ylabel('y')

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# Log Heatmap
# -----------------------------------------------------------------------

all_x = [x for sublist in level_df_filtered['x'].tolist() for x in sublist]
all_y = [y for sublist in level_df_filtered['y'].tolist() for y in sublist]

plt.figure(figsize = (10, 8))
hist, x_boundaries, y_boundaries = np.histogram2d(all_x, all_y, bins = [np.arange(min(all_x), max(all_x)+1), np.arange(min(all_y), max(all_y)+1)])
log_hist = np.log1p(hist)
mesh_x, mesh_y = np.meshgrid(x_boundaries, y_boundaries)
abc = plt.pcolormesh(mesh_x, mesh_y, log_hist.T, cmap = 'viridis')
plt.colorbar(abc, ax = plt.gca(), label = 'Density')
plt.title('Log Heatmap')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal', adjustable = 'box')
plt.grid(False)
plt.show()

# Analysing the temporal patterns ----------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Temporal CNN construction
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# features - x, y, speed, acceleration, direction, curvature, x derivative, y derivtive, local entropy, local variance
# note: applied to resampled trajectories
# -----------------------------------------------------------------------

def calculate_features(row, dt = 1):
    #read
    x = np.array(row['x_resampled'])
    y = np.array(row['y_resampled'])
    cos_theta = np.array(row['smoothed_cos_theta_resampled'])
    sin_theta = np.array(row['smoothed_sin_theta_resampled'])
    #1... x derivative & #2... y derivative
    dx = np.gradient(x, dt)
    dy = np.gradient(y, dt)
    #2... speed
    speed = np.sqrt(dx**2 + dy**2)
    #3... acceleration
    acceleration = np.gradient(speed, dt)
    #4... curvature -> delta cos(θ), delta sin(θ)
    delta_cos_theta = np.gradient(cos_theta, dt)
    delta_sin_theta = np.gradient(sin_theta, dt)
    #return
    return pd.Series({
        'x_derivative': dx,
        'y_derivative': dy,
        'speed': speed,
        'acceleration': acceleration,
        'delta_cos_theta': delta_cos_theta,
        'delta_sin_theta': delta_sin_theta
    })

level_df_filtered[['x_derivative', 'y_derivative', 'speed', 'acceleration', 'delta_cos_theta', 'delta_sin_theta']] = level_df_filtered.apply(calculate_features, axis = 1)

# level_df_filtered[['x_derivative', 'y_derivative', 'speed', 'acceleration', 'delta_cos_theta', 'delta_sin_theta']].head(3)
# level_df_filtered.info()

# -----------------------------------------------------------------------
# 1000 samples - builds data tensor for CNN and Temp2SpaceNN
# -----------------------------------------------------------------------

conv_df = level_df_filtered[['x_resampled', 'y_resampled', 'speed', 'acceleration', 'smoothed_cos_theta_resampled', 'smoothed_sin_theta_resampled', 'delta_cos_theta', 'delta_sin_theta', 'label']]
conv_df.info()

#gets 1000 samples from stratified samplin
sample_size = 1000
_, conv_dataframe = train_test_split(conv_df, test_size = sample_size, stratify = conv_df['label'], random_state = 7)

#conv_dataframe.info()
#conv_dataframe.head(1)

label_counts = conv_dataframe['label'].value_counts().reset_index()
label_counts.columns = ['label', 'frequency']
plt.figure(figsize = (10, 6))
sns.barplot(x = 'label', y = 'frequency', data = label_counts, palette = 'viridis')
plt.xticks(rotation = 45)
plt.xlabel('Labels')
plt.ylabel('Frequency')
plt.title('Distribution of Combined Age-Gender Labels')
plt.show()

#split into test and train (80/20)
conv_dataframe_train, conv_dataframe_test = train_test_split(conv_dataframe, test_size = 0.2, stratify = conv_dataframe['label'], random_state = 7)

def temporal_tensor_function(df, device, data_type):
    
    X = df.drop('label', axis = 1)
    y = df['label']
    
    label_counts = y.value_counts().reset_index()
    label_counts.columns = ['label', 'frequency']
    plt.figure(figsize = (10, 6))
    sns.barplot(x = 'label', y = 'frequency', data = label_counts, palette = 'viridis')
    plt.xticks(rotation = 45)
    plt.xlabel('Labels')
    plt.ylabel('Frequency')
    title = 'Distribution of Combined Age-Gender Labels - ' + data_type.capitalize()
    plt.title(title)
    plt.show()
    
    feature_data = []
    for column in X.columns:
        stacked_feats = np.stack(X[column].values)
        feature_data.append(stacked_feats)
    feature_array = np.stack(feature_data, axis = 1)
    feature_array = np.transpose(feature_array, (0, 2, 1))
    X_tensor = torch.tensor(feature_array, dtype = torch.float).to(device)
    
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(y)
    y_tensor = torch.tensor(encoded_labels, dtype = torch.long).to(device)
    
    return X_tensor, y_tensor

X_train_tensor, y_train_tensor = temporal_tensor_function(conv_dataframe_train, device, data_type = 'train')
print("Train:", X_train_tensor.shape, y_train_tensor.shape)

X_test_tensor, y_test_tensor = temporal_tensor_function(conv_dataframe_test, device, data_type = 'test')
print("Test:", X_test_tensor.shape, y_test_tensor.shape)

# -----------------------------------------------------------------------
# Convolutional neural network
# -----------------------------------------------------------------------

class MultiScaleConv1d(nn.Module):
    def __init__(self, hi_in, lo_in, hi_out, lo_out, k_hi = 5, k_lo = 7, d_lo = 2) -> None:
        super().__init__()
        self.hi_conv = nn.Conv1d(hi_in, hi_out, k_hi, padding = k_hi // 2)
        self.lo_conv = nn.Conv1d(
            lo_in, lo_out, k_lo, dilation = d_lo, padding = (d_lo * (k_lo - 1)) // 2
        )

    def forward(self, x):
        x_hi = self.hi_conv(x)
        x_lo = self.lo_conv(x)
        # print("x_hi shape:", x_hi.shape)
        # print("x_lo shape:", x_lo.shape)
        # print(x_hi[:, :, 1])
        return torch.cat([x_hi, x_lo], dim = 1)

class TempNN(nn.Module):
    def __init__(self, layers, h = 64, out_size = 10, l = 523) -> None:
        super().__init__()
        f = []
        for l_in, l_out in zip(layers[:-1], layers[1:]):
            hi_out = l_out // 2
            lo_out = l_out - hi_out
            f.append(MultiScaleConv1d(l_in, l_in, hi_out, lo_out, k_hi = 5, k_lo = 7, d_lo = 2))
            f.append(nn.MaxPool1d(2))
            f.append(nn.BatchNorm1d(l_out))
            f.append(nn.ReLU())
        self.conv = nn.Sequential(*f)
        
        self.out = nn.Sequential(
            nn.Linear((l // (2 ** (len(layers) - 1))) * layers[-1], h),
            nn.ReLU(),
            nn.BatchNorm1d(h),
            nn.Linear(h, out_size),
        )

    def forward(self, x):
        x = torch.swapaxes(x, 1, 2)
        n = x.shape[0]
        x = self.conv(x)
        return self.out(x.reshape(n, -1))

# -----------------------------------------------------------------------
# Training and Validation
# -----------------------------------------------------------------------

num_epochs = 50

#hyperparameter optimisation

def optuna_hyparam_opt(n_trial):
    lr = n_trial.suggest_loguniform('lr', 1e-5, 1e-1)
    momentum = n_trial.suggest_uniform('momentum', 0.5, 0.99)
    layers = n_trial.suggest_categorical('layers', [[8, 16, 32], [8, 32, 64]])
    h = n_trial.suggest_categorical('h', [64, 128, 256])

    #5-fold cross-val 80/20 train/val
    crossvalidationfold = KFold(n_splits = 5, shuffle = True, random_state = 7)
    crossvalidationfoldfoldresults = []
    for train_i, validation_i in crossvalidationfold.split(X_train_tensor):
        verytense_train_i = torch.tensor(train_i, dtype = torch.long).to(device)
        verytense_val_i = torch.tensor(validation_i, dtype = torch.long).to(device)

        X_train_fold = X_train_tensor.index_select(0, verytense_train_i).to(device)
        y_train_fold = y_train_tensor.index_select(0, verytense_train_i).to(device)
        X_val_fold = X_train_tensor.index_select(0, verytense_val_i).to(device)
        y_val_fold = y_train_tensor.index_select(0, verytense_val_i).to(device)

        model = TempNN(layers = layers, h = h, out_size = 10, l = 523).to(device)
        SGDoptimiser = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
        CrossEntLossF = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            SGDoptimiser.zero_grad()
            outputs = model(X_train_fold)
            learnin_on_trainsetloss = CrossEntLossF(outputs, y_train_fold)
            learnin_on_trainsetloss.backward()
            SGDoptimiser.step()

            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_fold)
                val_loss = CrossEntLossF(val_outputs, y_val_fold)

        crossvalidationfoldfoldresults.append(val_loss.item())

    return np.mean(crossvalidationfoldfoldresults)

study = optuna.create_study(direction = 'minimize') #american spellin
study.optimize(optuna_hyparam_opt, n_trials = 100)

#results
print("Trial | ", len(study.trials))
print("Best n_trial:")
n_trial = study.best_trial

print("- Value: ", n_trial.value)
print("- Parameters: ")
for key, value in n_trial.params.items():
    print(f" {key} : {value}")

best_model_params = n_trial.params

lr = best_model_params['lr'] # lr: 0.00017537008137213277
momentum = best_model_params['momentum'] # momentum: 0.9187273846444615
layers = best_model_params['layers'] # layers: [8, 16, 32]
h = best_model_params['h'] # h:  64

#train and validation with best model parameters

num_epochs = 15

comb_train_loss = []
comb_validation_loss = []
comb_train_acc = []
comb_validation_acc = []

crossvalidationfold = KFold(n_splits = 5, shuffle = True, random_state = 7)

for fold, (train_i, validation_i) in enumerate(crossvalidationfold.split(X_train_tensor)):
    print(f'Fold {fold+1}')

    verytense_train_i = torch.tensor(train_i, dtype = torch.long).to(device)
    verytense_val_i = torch.tensor(validation_i, dtype = torch.long).to(device)

    X_train_fold = X_train_tensor.index_select(0, verytense_train_i).to(device)
    y_train_fold = y_train_tensor.index_select(0, verytense_train_i).to(device)
    X_val_fold = X_train_tensor.index_select(0, verytense_val_i).to(device)
    y_val_fold = y_train_tensor.index_select(0, verytense_val_i).to(device)

    model = TempNN(layers = layers, h = h, out_size = 10, l = 523).to(device)
    SGDoptimiser = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    CrossEntLossF = nn.CrossEntropyLoss()

    listof_train_losses = []
    listof_validation_losses = []
    listof_train_acc = []
    listof_validation_acc = []

    for epoch in range(num_epochs):
        model.train()
        SGDoptimiser.zero_grad()
        outputs = model(X_train_fold)
        learnin_on_trainsetloss = CrossEntLossF(outputs, y_train_fold)
        learnin_on_trainsetloss.backward()
        SGDoptimiser.step()
        
        listof_train_losses.append(learnin_on_trainsetloss.item())
        predicted_labs = torch.argmax(outputs, dim = 1)
        correct_lab_predictions = (predicted_labs == y_train_fold).sum().item()
        train_acc = correct_lab_predictions / y_train_fold.size(0)
        listof_train_acc.append(train_acc)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_fold)
            val_loss = CrossEntLossF(val_outputs, y_val_fold)
            listof_validation_losses.append(val_loss.item())
            val_predicted_labels = torch.argmax(val_outputs, dim = 1)
            val_correct_predictions = (val_predicted_labels == y_val_fold).sum().item()
            val_acc = val_correct_predictions / y_val_fold.size(0)
            listof_validation_acc.append(val_acc)

        print(f"Epoch - {epoch + 1} ; Train_L{learnin_on_trainsetloss.item()} ; Train_A{train_acc} ; Val_L{val_loss.item()} ; Val_A{val_acc}") #1 to epoch cus starts at 0
        
    comb_train_loss.append(listof_train_losses)
    comb_validation_loss.append(listof_validation_losses)
    comb_train_acc.append(listof_train_acc)
    comb_validation_acc.append(listof_validation_acc)

comb_train_loss = np.mean(comb_train_loss, axis = 0)
comb_validation_loss = np.mean(comb_validation_loss, axis = 0)
comb_train_acc = np.mean(comb_train_acc, axis = 0)
comb_validation_acc = np.mean(comb_validation_acc, axis = 0)

#loss curve
plt.figure(figsize = (10, 6))
plt.plot(comb_train_loss, label = 'Training Loss')
plt.plot(comb_validation_loss, label = 'Validation Loss')
plt.title('Cross Entropy Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.legend()
plt.show()

#accuracy curve
plt.figure(figsize = (10, 6))
plt.plot(comb_train_acc, label = 'Training Accuracy')
plt.plot(comb_validation_acc, label = 'Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# -----------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------

model.eval()

test_preds_list = []
true_labs_list = []

with torch.no_grad():
    for i in range(len(X_test_tensor)):
        X_batch = X_test_tensor[i].unsqueeze(0).to(device)
        y_batch = y_test_tensor[i].to(device)

        outputs = model(X_batch)
        _, predicted_labs = torch.max(outputs, dim = 1)

        test_preds_list.append(predicted_labs.item())
        true_labs_list.append(y_batch.item())

#accuracy
accuracy = sum([1 for true, pred in zip(true_labs_list, test_preds_list) if true == pred]) / len(true_labs_list)
print(f'Accuracy: {accuracy * 100:.2f}%')

#confusion matrix
matrix_confusion = confusion_matrix(true_labs_list, test_preds_list)
plt.figure(figsize = (10, 8))
ax = sns.heatmap(matrix_confusion, annot = True, fmt = 'g', cmap = 'viridis', 
                 xticklabels = ['f-age_0', 'f-age_1', 'f-age_2', 'f-age_3', 'f-age_4', 'm-age_0', 'm-age_1', 'm-age_2', 'm-age_3', 'm-age_4'], 
                 yticklabels = ['f-age_0', 'f-age_1', 'f-age_2', 'f-age_3', 'f-age_4', 'm-age_0', 'm-age_1', 'm-age_2', 'm-age_3', 'm-age_4'])
ax.invert_yaxis()
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

#classification report
print(classification_report(true_labs_list, test_preds_list, target_names = [
    'f-age_0', 'f-age_1', 'f-age_2', 'f-age_3', 'f-age_4', 
    'm-age_0', 'm-age_1', 'm-age_2', 'm-age_3', 'm-age_4']))

# Analysing the spatial patterns ----------------------------------------------------------------------------------------------------------------------------------------------

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Graph construction
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Heatmap of the trajectories (not log)
# -----------------------------------------------------------------------

all_x = [x for sublist in level_df_filtered['x'].tolist() for x in sublist]
all_y = [y for sublist in level_df_filtered['y'].tolist() for y in sublist]

plt.figure(figsize = (10, 8))
hist, x_boundaries, y_boundaries, abc = plt.hist2d(all_x, all_y, bins = [np.arange(min(all_x), max(all_x)+1), np.arange(min(all_y), max(all_y)+1)], cmin = 1, cmap = 'coolwarm_r') #coolwarm_r to diff and out-of-bounds 
plt.colorbar(abc, ax = plt.gca(), label = 'frequency')
plt.title('Normal Heatmap')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal', adjustable = 'box')
plt.grid(False)
plt.show()

# -----------------------------------------------------------------------
# Local maxima of the KDE inverse distribution - CURRENTLY READ FROM FILE TO SAVE TIME RERUNING IF CHANGE, THEN SWAP #'s TO RERUN KDE ON NEW DATA
# -----------------------------------------------------------------------

all_x_coords = np.hstack(level_df_filtered['x'].values) #numpyarraynotlist
all_y_coords = np.hstack(level_df_filtered['y'].values)

KDE_of_all_xy = gaussian_kde(np.vstack([all_x_coords, all_y_coords]))

x_min, x_max = min(all_x_coords), max(all_x_coords)
y_min, y_max = min(all_y_coords), max(all_y_coords)

grid_size_factor = 100
x_grid_1000 = np.linspace(x_min, x_max, grid_size_factor)
y_grid_1000 = np.linspace(y_min, y_max, grid_size_factor)
x_mesh_kde, y_mesh_kde = np.meshgrid(x_grid_1000, y_grid_1000)

#kde_results = KDE_of_all_xy(np.vstack([x_mesh_kde.ravel(), y_mesh_kde.ravel()])).reshape(x_mesh_kde.shape)
#np.save("../../data/interim/z_level.npy", kde_results)
kde_results = np.load("../../data/interim/z_level.npy")

local_maxima = peak_local_max(kde_results, min_distance = 5, threshold_abs = None)
# Plot of KDE + local maximas
plt.figure(figsize = (10, 8))
plt.imshow(kde_results, origin = 'lower', aspect = 'auto', extent = [x_min, x_max, y_min, y_max], cmap = 'coolwarm_r')
plt.colorbar(label = 'Density')
for maxima in local_maxima:
    plt.scatter(x_grid_1000[maxima[1]], y_grid_1000[maxima[0]], color = 'blue', s = 50, marker = 'x')
plt.title('KDE Heatmap with local maxima (marked x)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# -----------------------------------------------------------------------
# Map KDE local maxima coordinates to actual trajectory coordinate grid
# -----------------------------------------------------------------------

x_res = (x_max - x_min) / 99
y_res = (y_max - y_min) / 99

local_maxima_actual_coordinates_rounded = []
for maxima in local_maxima:
    map_localmax_x = (maxima[1] * x_res) + x_min
    map_localmax_y = (maxima[0] * y_res) + y_min
    actual_x_rounded, actual_y_rounded = int(round(map_localmax_x)), int(round(map_localmax_y)) #round because discretisation :/
    local_maxima_actual_coordinates_rounded.append((actual_x_rounded, actual_y_rounded))

# plt.figure(figsize = (10, 8))
# plt.imshow(kde_results, origin = 'lower', aspect = 'auto', extent = [x_min, x_max, y_min, y_max], cmap = 'coolwarm_r')
# plt.colorbar(label = 'Density')
# for x in local_maxima_actual_coordinates_rounded:
#     plt.scatter(x[0], x[1], color = 'blue', marker = 'x')
# plt.title('')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

plt.figure(figsize = (10, 8))
hist, x_boundaries, y_boundaries, abc = plt.hist2d(all_x, all_y, bins = [np.arange(min(all_x), max(all_x)+1), np.arange(min(all_y), max(all_y)+1)], cmin = 1, cmap = 'coolwarm_r')
plt.colorbar(abc, ax = plt.gca(), label = 'Density')
for x in local_maxima_actual_coordinates_rounded:
    plt.scatter(x[0] + 0.5, x[1] + 0.5, color = 'blue', marker = 'x')
plt.title('Local maxima (marked x) mapped to original trajectory\ncoordinate grid (illustrated on normal heatmap)')
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal', adjustable = 'box')
plt.grid(False)
plt.show()

# -----------------------------------------------------------------------
# Watershed algorithm segmentation for macro signal
# -----------------------------------------------------------------------

grid = np.zeros((y_max + 1, x_max + 1))
for x, y in zip(all_x_coords, all_y_coords):
    grid[y, x] = 1

markers = np.zeros_like(grid)
for i, (x, y) in enumerate(local_maxima_actual_coordinates_rounded):
    markers[y, x] = i + 1
markers = markers.astype(np.int32)

# Watershed without density consideration
# ws = watershed(grid, markers, mask = grid)

# plt.figure(figsize = (10, 10))
# plt.imshow(ws, cmap = 'nipy_spectral', origin = 'lower')
# plt.scatter(*zip(*local_maxima_actual_coordinates_rounded), color = 'white', marker = 'x')
# plt.title('ABC')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# Watershed with density consideration
density_map = np.zeros_like(grid)
for x, y in zip(all_x_coords, all_y_coords):
    density_map[y, x] += 1

ws2 = watershed(-density_map, markers, mask = grid) #neg

plt.figure(figsize = (10, 10))
plt.imshow(ws2, cmap = 'nipy_spectral', origin = 'lower')
plt.scatter(*zip(*local_maxima_actual_coordinates_rounded), color = 'white', marker = 'x')
plt.title('Macro Watershed Segmentation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# -----------------------------------------------------------------------
# Finds the centres of each macro region
# -----------------------------------------------------------------------

unique_regions = np.unique(ws2)
centres = {}
for region in unique_regions:
    if region == 0:
        continue
    coords = np.column_stack(np.where(ws2 == region))
    centre = coords.mean(axis = 0)
    roundcentre = np.round(centre).astype(int)
    centres[region] = roundcentre

plt.figure(figsize = (10, 10))
plt.imshow(ws2, cmap = 'nipy_spectral', origin = 'lower')
plt.scatter(*zip(*local_maxima_actual_coordinates_rounded), color = 'white', marker = 'x')
for region, centre in centres.items():
    plt.scatter(centre[1], centre[0], color = 'red', marker = 'o', edgecolors = 'white')
plt.title('Macro Watershed Segmentation - local maxima (X), Centres (O)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# -----------------------------------------------------------------------
# Watershed algorithm segmentation for micro signal
# -----------------------------------------------------------------------

ws2_micro = np.zeros_like(ws2)

for segment_region_i in np.unique(ws2):
    if segment_region_i == 0:
        continue
    
    mask_microisolation = (ws2 == segment_region_i)
    isolated_region = np.where(mask_microisolation, density_map, 0)
    local_maximas_micro = peak_local_max(-isolated_region, labels = mask_microisolation)
    markers_micro = np.zeros_like(isolated_region)
    for i, (y, x) in enumerate(local_maximas_micro, 1):
        markers_micro[y, x] = i
    markers_micro = markers_micro.astype(np.int32)

    ws2_region_micro = watershed(isolated_region, markers_micro, mask = mask_microisolation)
    ws2_region_micro[ws2_region_micro > 0] += ws2_micro.max()
    ws2_micro[mask_microisolation] = ws2_region_micro[mask_microisolation]

#micro region plot
plt.figure(figsize = (10, 10))
plt.imshow(ws2_micro, cmap = 'nipy_spectral', origin = 'lower')
plt.title('Micro Watershed Segmentation')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

#macro vs micro region plot
fig, ax = plt.subplots(1, 2, figsize = (15, 10))
ax[0].imshow(ws2, cmap = 'nipy_spectral', origin = 'lower')
ax[0].scatter(*zip(*local_maxima_actual_coordinates_rounded), color = 'white', marker = 'x')
ax[0].set_title('Macro Segments')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')

ax[1].imshow(ws2_micro, cmap = 'nipy_spectral', origin = 'lower')
ax[1].scatter(*zip(*local_maxima_actual_coordinates_rounded), color = 'white', marker = 'x')
ax[1].set_title('Micro segments')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# Finds the centres of each micro region
# -----------------------------------------------------------------------

unique_regions_micro = np.unique(ws2_micro)
centres_micro = {}

for segment_region_i in unique_regions_micro:
    if segment_region_i == 0:
        continue
    
    coords = np.column_stack(np.where(ws2_micro == segment_region_i))
    centre = coords.mean(axis = 0)
    roundcentre = np.round(centre).astype(int)
    centres_micro[segment_region_i] = roundcentre

plt.figure(figsize = (10, 10))
plt.imshow(ws2_micro, cmap = 'nipy_spectral', origin = 'lower')
for region, centre in centres.items():
    plt.scatter(centre[1], centre[0], color = 'red', marker = 'o', edgecolors = 'white') #macromarks
for region, centre in centres_micro.items():
    plt.scatter(centre[1], centre[0], color = 'black', marker = 'o', edgecolors = 'white') #micromarks
plt.title('Segmentation with centres (macro centres in red, micro centres in black)')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# -----------------------------------------------------------------------
# Construct graph structure using the centres as nodes
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Part 1a: Micro graph nodes
# -----------------------------------------------------------------------

G_micro = nx.Graph()

for region, centre in centres_micro.items():
    G_micro.add_node(region, pos = (centre[1], centre[0]))

def get_micro_node_from_coord(x, y, segmentation):
    x = int(x) #rounds (important for resampled)
    y = int(y) #rounds (important for resampled)
    if 0 <=  x < segmentation.shape[1] and 0 <=  y < segmentation.shape[0]:
        segment_region_i = segmentation[y, x]
        if segment_region_i !=  0:
            return segment_region_i
    return None

# -----------------------------------------------------------------------
# Part 1b: Micro graph node visit signals
# -----------------------------------------------------------------------

visit_signals = {node: {} for node in np.unique(ws2_micro[ws2_micro > 0])}

for _, row in level_df_filtered.iterrows():
    x_list = row['x']
    y_list = row['y']

    trajectory_id = row['uuid_x']
    for node in visit_signals:
        if trajectory_id not in visit_signals[node]:
            visit_signals[node][trajectory_id] = 0

    for x, y in zip(x_list, y_list):
        micro_node = get_micro_node_from_coord(x, y, ws2_micro)
        if micro_node:
            visit_signals[micro_node][trajectory_id] = 1

G_micro = nx.Graph()
for region, centre in centres_micro.items():
    G_micro.add_node(region, pos = (centre[1], centre[0]), visit_signal = visit_signals[region])

# -----------------------------------------------------------------------
# Part 1c: Micro graph node visit signal visualisations
# -----------------------------------------------------------------------

#example plot of visualisation of example of all concatenated visit signals on one graph
#num_rows_len = len(level_df_filtered)
total_visits = np.array([sum(visit_signals[node].values()) for node in G_micro.nodes()])
max_visits = total_visits.max()
node_colors = total_visits
fig, ax = plt.subplots(figsize = (12, 10))
pos = nx.get_node_attributes(G_micro, 'pos')
nx.draw(G_micro, pos, ax = ax, node_color = node_colors, with_labels = True, cmap = plt.cm.Blues, node_size = 400)
plt.title('Example: Visit Signal (concatenated for all trajectories)')
sm = plt.cm.ScalarMappable(cmap = plt.cm.Blues, norm = plt.Normalize(vmin = 0, vmax = max_visits))
sm.set_array([])
cbar = plt.colorbar(sm, ax = ax, label = 'Total Visits')
plt.show()

#example plot of visualisation of a signal on an example trajectory
trajectory_id = '2abf1c97-db65-42e8-adef-e2573667f717'
node_colors = [visit_signals[node].get(trajectory_id, 0) for node in G_micro.nodes()]
visited_nodes = [node for node, signal in zip(G_micro.nodes(), node_colors) if signal > 0]
non_visited_nodes = [node for node, signal in zip(G_micro.nodes(), node_colors) if signal == 0]
plt.figure(figsize = (12, 10))
pos = nx.get_node_attributes(G_micro, 'pos')
nx.draw_networkx_nodes(G_micro, pos, nodelist = non_visited_nodes, node_color = 'lightgray', node_size = 400)
nx.draw_networkx_labels(G_micro, pos)
nx.draw_networkx_nodes(G_micro, pos, nodelist = visited_nodes, node_color = 'yellow', node_size = 400)
nx.draw_networkx_edges(G_micro, pos)
plt.title(f'Example Trajectory Signal: {trajectory_id}')
plt.grid(False)
plt.show()

#trajectory plot of said trajectory
trajectory_vis = level_df_filtered[level_df_filtered['uuid_x'] == trajectory_id]
x_coords = trajectory_vis['x'].iloc[0]
y_coords = trajectory_vis['y'].iloc[0]
plt.figure(figsize = (10, 8))
plt.plot(x_coords, y_coords, '-o', color = 'midnightblue', linewidth = 2, markersize = 6, markeredgecolor = 'black')
plt.title('Associated trajectory (for visual comparison)')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(False)
plt.show()

# -----------------------------------------------------------------------
# Part 1d: Micro graph edge construction - for visualisations
# -----------------------------------------------------------------------

edge_weights = {}

for _, row in level_df_filtered.iterrows():
    x_list = row['x']
    y_list = row['y']
    trajectory_id = row['uuid_x']

    previous_node = None
    for x, y in zip(x_list, y_list):
        micro_node = get_micro_node_from_coord(x, y, ws2_micro)
        if micro_node and micro_node !=  previous_node:
            if previous_node is not None:
                edge = tuple(sorted((previous_node, micro_node)))
                if edge in edge_weights:
                    edge_weights[edge] += 1
                else:
                    edge_weights[edge] = 1
            previous_node = micro_node

for region, centre in centres_micro.items():
    G_micro.add_node(region, pos = (centre[1], centre[0]))

for edge, weight in edge_weights.items():
    G_micro.add_edge(edge[0], edge[1], weight = weight)

# -----------------------------------------------------------------------
# Part 1c: Micro graph edge visualisations
# -----------------------------------------------------------------------

pos = nx.get_node_attributes(G_micro, 'pos')

#plot without edge weights
plt.figure(figsize = (12, 12))
nx.draw_networkx_nodes(G_micro, pos, node_color = 'skyblue', node_size = 400)
nx.draw_networkx_labels(G_micro, pos)
nx.draw_networkx_edges(G_micro, pos, edge_color = 'black')
plt.title('Graph without edge weights', fontsize = 16, fontweight = 'bold')
plt.axis('off')
plt.tight_layout()
plt.show()

#plot with edge weights
plt.figure(figsize = (12, 12))
nx.draw_networkx_nodes(G_micro, pos, node_color = 'skyblue', node_size = 400)
nx.draw_networkx_labels(G_micro, pos)
edge_weights = nx.get_edge_attributes(G_micro, 'weight')
nx.draw_networkx_edges(G_micro, pos, edge_color = 'black', width = [weight / max(edge_weights.values()) * 3 for weight in edge_weights.values()])
plt.title('Graph with edge weights', fontsize = 16, fontweight = 'bold')
plt.axis('off')
plt.tight_layout()
plt.show()

# -----------------------------------------------------------------------
# Additional features needed
# -----------------------------------------------------------------------

# def calculate_features2(row, dt = 1):
#     #get
#     x = np.array(row['x'])
#     y = np.array(row['y'])
#     cos_theta = np.array(row['smoothed_cos_theta'])
#     sin_theta = np.array(row['smoothed_sin_theta'])
#     #1... x derivative & #2... y derivative
#     dx = np.gradient(x, dt)
#     dy = np.gradient(y, dt)
#     #2... speed
#     speed = np.sqrt(dx**2 + dy**2)
#     #3... acceleration
#     acceleration = np.gradient(speed, dt)
#     #4... curvature -> δcos(θ), δsin(θ)
#     delta_cos_theta = np.gradient(cos_theta, dt)
#     delta_sin_theta = np.gradient(sin_theta, dt)
#     #return
#     return pd.Series({
#         'speed(OG)': speed,
#         'acceleration(OG)': acceleration,
#         'delta_cos_theta(OG)': delta_cos_theta,
#         'delta_sin_theta(OG)': delta_sin_theta
#     })

# level_df_filtered[['speed(OG)', 'acceleration(OG)', 'delta_cos_theta(OG)', 'delta_sin_theta(OG)']] = level_df_filtered.apply(calculate_features2, axis = 1)
# level_df_filtered.info()

# -----------------------------------------------------------------------
# Contribution matrix
# -----------------------------------------------------------------------

def contribution_matrix_construction(input_tensor, segmentation):
    xy_tensor = input_tensor[:, :, :2]
    trajectories, length, _ = xy_tensor.shape
    contribution_matrix = torch.zeros((trajectories, length), dtype = torch.long)
    
    for b in range(trajectories):
        for t in range(length):
            x, y = xy_tensor[b, t, :]
            node = get_micro_node_from_coord(x.item(), y.item(), segmentation)
            if node is not None:
                contribution_matrix[b, t] = node
            else:
                contribution_matrix[b, t] = -1 #assigns -1 if not valid
    
    #replace -1s with last seen node (fixes resampling issue)
    for trajectories in range(contribution_matrix.shape[0]):
        for i in range(1, contribution_matrix.shape[1]):
            if contribution_matrix[trajectories, i] == -1:
                contribution_matrix[trajectories, i] = contribution_matrix[trajectories, i-1]
    
    contribution_matrix = contribution_matrix.unsqueeze(-1)
    return contribution_matrix

contribution_matrix_train = contribution_matrix_construction(X_train_tensor, ws2_micro)
contribution_matrix_test = contribution_matrix_construction(X_test_tensor, ws2_micro)

contribution_matrix_train.to(device)
contribution_matrix_test.to(device)

# print("Shape of the modified train tensor:", contribution_matrix_train.shape)
# print("Shape of the modified test tensor:", contribution_matrix_test.shape)
# print((contribution_matrix_train == -1).any())
# print((contribution_matrix_test == -1).any())
# print(contribution_matrix_train[0, 283:293])

#adapts current contribution matrix to have binary contribution matrix on the 3rd dimension instead of 1 column and the number node it corresponds to
n_nodes = G_micro.number_of_nodes()
print("Number of nodes in the graph:", n_nodes)

binary_matrix_train = torch.zeros((contribution_matrix_train.size(0), contribution_matrix_train.size(1), n_nodes), dtype = torch.float)

for a in range(contribution_matrix_train.size(0)):
    for b in range(contribution_matrix_train.size(1)):
        node_index = contribution_matrix_train[a, b, 0].item()
        if node_index !=  -1:
            binary_matrix_train[a, b, node_index - 1] = 1

#check
print(binary_matrix_train.shape)
node_i = contribution_matrix_train[0, 283, 0].item()
print(node_i)
print(binary_matrix_train[0, 283, :10].numpy())

#apply to test set
binary_matrix_test = torch.zeros((contribution_matrix_test.size(0), contribution_matrix_test.size(1), n_nodes), dtype = torch.float)

for a in range(contribution_matrix_test.size(0)):
    for b in range(contribution_matrix_test.size(1)):
        node_index = contribution_matrix_test[a, b, 0].item()
        if node_index !=  -1:
            binary_matrix_test[a, b, node_index - 1] = 1

print(binary_matrix_test.shape)

# -----------------------------------------------------------------------
# Temp2SpaceNN
# -----------------------------------------------------------------------
class Temp2SpaceNN(nn.Module):
    def __init__(self, n_nodes, out_size = 10) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(8, 32, 5, padding = 2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding = 2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.spatial = nn.Sequential(
            nn.Linear(n_nodes * 64, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Linear(100, out_size),
        )

    def pooling(self, x, b) -> torch.Tensor:
        y = torch.einsum("nij,nik->nijk", b, x)
        z, _ = torch.max(y, axis = 1)
        return z

    def forward(self, x, b) -> torch.Tensor:       
        n = x.shape[0]
        x = torch.swapaxes(x, 1, 2)
        x = self.temporal(x)
        x = torch.swapaxes(x, 1, 2)
        x = self.pooling(x, b).reshape(n, -1)
        #print(x.shape)
        return self.spatial(x)

# -----------------------------------------------------------------------
# Training and Validation
# -----------------------------------------------------------------------

#input data
#X_train_tensor, binary_matrix_train, y_train_tensor #train
#X_test_tensor, binary_matrix_test, y_test_tensor #test

#hyperparameter optimisation

# def optuna_hyparam_opt(n_trial):
#     lr = n_trial.suggest_loguniform('lr', 1e-5, 1e-1)
#     momentum = n_trial.suggest_uniform('momentum', 0.5, 0.99)
    
#     n_nodes = G_micro.number_of_nodes()
#     out_size = 10
    
#     crossvalidationfoldfoldresults = []
#     crossvalidationfold = KFold(n_splits = 5, shuffle = True, random_state = 7)
#     for train_i, validation_i in crossvalidationfold.split(X_train_tensor):
#         verytense_train_i = torch.tensor(train_i, dtype = torch.long).to(device)
#         verytense_val_i = torch.tensor(validation_i, dtype = torch.long).to(device)

#         X_train_fold = X_train_tensor.index_select(0, verytense_train_i).to(device)
#         y_train_fold = y_train_tensor.index_select(0, verytense_train_i).to(device)
#         X_val_fold = X_train_tensor.index_select(0, verytense_val_i).to(device)
#         y_val_fold = y_train_tensor.index_select(0, verytense_val_i).to(device)

#         contribution_matrix_train_fold = binary_matrix_train[train_i].to(device)
#         contribution_matrix_val_fold = binary_matrix_train[validation_i].to(device)

#         model = Temp2SpaceNN(n_nodes = n_nodes, out_size = out_size).to(device)
#         SGDoptimiser = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
#         CrossEntLossF = nn.CrossEntropyLoss()

#         for epoch in range(num_epochs):
#             model.train()
#             SGDoptimiser.zero_grad()
#             outputs = model(X_train_fold, contribution_matrix_train_fold)
#             learnin_on_trainsetloss = CrossEntLossF(outputs, y_train_fold)
#             learnin_on_trainsetloss.backward()
#             SGDoptimiser.step()

#             model.eval()
#             with torch.no_grad():
#                 val_outputs = model(X_val_fold, contribution_matrix_val_fold)
#                 val_loss = CrossEntLossF(val_outputs, y_val_fold)

#         crossvalidationfoldfoldresults.append(val_loss.item())

#     return np.mean(crossvalidationfoldfoldresults)

# study = optuna.create_study(direction = 'minimize')
# study.optimize(optuna_hyparam_opt, n_trials = 100)
    
# #results
# print("Trial | ", len(study.trials))
# print("Best n_trial:")
# n_trial = study.best_trial

# print("- Value: ", n_trial.value)
# print("- Parameters: ")
# for key, value in n_trial.params.items():
#     print(f" {key} : {value}")

#defined for simplicity in not having to run the above model everytime
lr = 0.0186231695512178
momentum = 0.9616195759107272

#train and validation with best model parameters

n_nodes = G_micro.number_of_nodes()
print("Number of nodes in the graph:", n_nodes)

num_epochs = 15

comb_train_loss = []
comb_validation_loss = []
comb_train_acc = []
comb_validation_acc = []

crossvalidationfold = KFold(n_splits = 5, shuffle = True, random_state = 7)

for fold, (train_i, validation_i) in enumerate(crossvalidationfold.split(X_train_tensor)):
    print(f'Fold {fold+1}')

    verytense_train_i = torch.tensor(train_i, dtype = torch.long).to(device)
    verytense_val_i = torch.tensor(validation_i, dtype = torch.long).to(device)

    X_train_fold = X_train_tensor.index_select(0, verytense_train_i).to(device)
    y_train_fold = y_train_tensor.index_select(0, verytense_train_i).to(device)
    X_val_fold = X_train_tensor.index_select(0, verytense_val_i).to(device)
    y_val_fold = y_train_tensor.index_select(0, verytense_val_i).to(device)
    
    contribution_matrix_train_fold = binary_matrix_train[train_i].to(device)
    contribution_matrix_val_fold = binary_matrix_train[validation_i].to(device)

    model_2 = Temp2SpaceNN(n_nodes = n_nodes, out_size = 10).to(device)
    SGDoptimiser = optim.SGD(model_2.parameters(), lr = lr, momentum = momentum)
    CrossEntLossF = nn.CrossEntropyLoss()

    listof_train_losses = []
    listof_validation_losses = []
    listof_train_acc = []
    listof_validation_acc = []

    for epoch in range(num_epochs):
        model_2.train()
        SGDoptimiser.zero_grad()
        outputs = model_2(X_train_fold, contribution_matrix_train_fold)
        learnin_on_trainsetloss = CrossEntLossF(outputs, y_train_fold)
        learnin_on_trainsetloss.backward()
        SGDoptimiser.step()
        
        listof_train_losses.append(learnin_on_trainsetloss.item())
        predicted_labs = torch.argmax(outputs, dim = 1)
        correct_lab_predictions = (predicted_labs == y_train_fold).sum().item()
        train_acc = correct_lab_predictions / y_train_fold.size(0)
        listof_train_acc.append(train_acc)

        model_2.eval()
        with torch.no_grad():
            val_outputs = model_2(X_val_fold, contribution_matrix_val_fold)
            val_loss = CrossEntLossF(val_outputs, y_val_fold)
            listof_validation_losses.append(val_loss.item())
            val_predicted_labels = torch.argmax(val_outputs, dim = 1)
            val_correct_predictions = (val_predicted_labels == y_val_fold).sum().item()
            val_acc = val_correct_predictions / y_val_fold.size(0)
            listof_validation_acc.append(val_acc)

        print(f"Epoch - {epoch + 1} ; Train_L{learnin_on_trainsetloss.item()} ; Train_A{train_acc} ; Val_L{val_loss.item()} ; Val_A{val_acc}")

    comb_train_loss.append(listof_train_losses)
    comb_validation_loss.append(listof_validation_losses)
    comb_train_acc.append(listof_train_acc)
    comb_validation_acc.append(listof_validation_acc)

comb_train_loss = np.mean(comb_train_loss, axis = 0)
comb_validation_loss = np.mean(comb_validation_loss, axis = 0)
comb_train_acc = np.mean(comb_train_acc, axis = 0)
comb_validation_acc = np.mean(comb_validation_acc, axis = 0)

#loss curve
plt.figure(figsize = (10, 6))
plt.plot(comb_train_loss, label = 'Training Loss')
plt.plot(comb_validation_loss, label = 'Validation Loss')
plt.title('Cross Entropy Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.legend()
plt.show()

#accuracy curve
plt.figure(figsize = (10, 6))
plt.plot(comb_train_acc, label = 'Training Accuracy')
plt.plot(comb_validation_acc, label = 'Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# -----------------------------------------------------------------------
# Testing
# -----------------------------------------------------------------------

model_2.eval()

test_preds_list = []
true_labs_list = []

with torch.no_grad():
    for i in range(len(X_test_tensor)):
        X_batch = X_test_tensor[i].unsqueeze(0).to(device)
        contribution_matrix_batch = binary_matrix_test[i].unsqueeze(0).to(device)
        y_batch = y_test_tensor[i].to(device)

        outputs = model_2(X_batch, contribution_matrix_batch)
        _, predicted_labs = torch.max(outputs, dim = 1)

        test_preds_list.append(predicted_labs.item())
        true_labs_list.append(y_batch.item())

#accuracy
accuracy = sum([1 for true, pred in zip(true_labs_list, test_preds_list) if true == pred]) / len(true_labs_list)
print(f'Accuracy: {accuracy * 100:.2f}%')

#confusion matrix
matrix_confusion = confusion_matrix(true_labs_list, test_preds_list)
plt.figure(figsize = (10, 8))
ax = sns.heatmap(matrix_confusion, annot = True, fmt = 'g', cmap = 'viridis', 
                 xticklabels = ['f-age_0', 'f-age_1', 'f-age_2', 'f-age_3', 'f-age_4', 'm-age_0', 'm-age_1', 'm-age_2', 'm-age_3', 'm-age_4'], 
                 yticklabels = ['f-age_0', 'f-age_1', 'f-age_2', 'f-age_3', 'f-age_4', 'm-age_0', 'm-age_1', 'm-age_2', 'm-age_3', 'm-age_4'])
ax.invert_yaxis()
plt.xticks(rotation = 45)
plt.yticks(rotation = 45)
plt.title('Confusion Matrix on Test Set')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

#classification report
print(classification_report(true_labs_list, test_preds_list, target_names = [
    'f-age_0', 'f-age_1', 'f-age_2', 'f-age_3', 'f-age_4', 
    'm-age_0', 'm-age_1', 'm-age_2', 'm-age_3', 'm-age_4']))

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------
# Further research
# -----------------------------------------------------------------------
# -----------------------------------------------------------------------

# -----------------------------------------------------------------------
# Macro graph construction
# -----------------------------------------------------------------------

G_macro = nx.Graph()

for region, centre in centres.items():
    G_macro.add_node(region, pos = (centre[1], centre[0]))

for region1 in centres:
    for region2 in centres:
        if region1 !=  region2:
            G_macro.add_edge(region1, region2)

pos = nx.get_node_attributes(G_macro, 'pos')
plt.figure(figsize = (10, 10))
nx.draw(G_macro, pos, node_color = 'red', with_labels = True, node_size = 100, edge_color = 'gray')
plt.title('ABC', fontsize = 16, fontweight = 'bold')
plt.show()

# -----------------------------------------------------------------------
# Bipartite graph
# -----------------------------------------------------------------------

G_bipartite = nx.DiGraph()

macro_nodes = G_macro.nodes(data = True)
micro_nodes = G_micro.nodes(data = True)

micro_labels = np.unique(ws2_micro)

for node, data in micro_nodes:
    G_bipartite.add_node(node, **data, bipartite = 0)

#sorts out both starting from 1 (puts macro last)
max_micro_label = max(micro_labels)
for node, data in macro_nodes:
    new_label = node + max_micro_label
    G_bipartite.add_node(new_label, **data, bipartite = 1)

#maps macro to micro for directed edges
micro_to_macro_mapping = {}

for micro_label in micro_labels:
    if micro_label == 0:
        continue
    micro_region_pixels = np.where(ws2_micro == micro_label)
    y, x = micro_region_pixels[0][0], micro_region_pixels[1][0]
    macro_label = ws2[y, x] + max_micro_label
    micro_to_macro_mapping[micro_label] = macro_label

for micro, macro in micro_to_macro_mapping.items():
    G_bipartite.add_edge(micro, macro)

to_bipartite_or_not_to_bipartite = nx.is_bipartite(G_bipartite)
print(f"to bipartite or not to bipartite, that is the question | {to_bipartite_or_not_to_bipartite}")

pos = {node: data['pos'] for node, data in G_bipartite.nodes(data = True) if 'pos' in data}
node_colors = ['skyblue' if data['bipartite'] == 0 else 'red' for node, data in G_bipartite.nodes(data = True)]

plt.figure(figsize = (12, 12))
nx.draw_networkx_nodes(G_bipartite, pos, node_color = node_colors, node_size = 400)
nx.draw_networkx_labels(G_bipartite, pos)
nx.draw_networkx_edges(G_bipartite, pos, alpha = 0.5, edge_color = 'gray', arrows = True) #americanspellin
plt.title('Directed Bipartite Graph', fontsize = 16, fontweight = 'bold')
plt.axis('off')
plt.tight_layout()
plt.show()