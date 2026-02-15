#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 14:04:39 2024

@author: spaggi
"""
import csv
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
#number of cards for a player max 52
#number of rounds max 120
card_mat=np.zeros((100,160))

filename="machiagame1p2ptableround.csv"
# Open the CSV file
with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    
    # Read one line at a time
    
    for row in reader:
        
        rlist=list(row)
        # print(rlist)
        rounds=int(rlist[3])
        card_mat[int(rlist[2]),rounds]+=1

#normalize over each column
norm_card_mat=card_mat.copy()
for index in range(len(card_mat.T)):
    norm=np.sum(card_mat[:,index])
    if norm !=0:
        norm_card_mat[:,index]/=norm
        
# Plot the matrix
white_to_red = LinearSegmentedColormap.from_list('white_to_red', ['white', 'red'])
colors = [(0, "lightblue"), (1e-5, "white"), (1, "red")]  # Custom stops: Black at 0, white-to-red
black_white_red = LinearSegmentedColormap.from_list("black_white_red", colors)
turbo = plt.cm.get_cmap('turbo')  # Turbo colormap
colors = turbo(np.linspace(0, 1, 256))  # Get the 256 colors in the Turbo colormap
colors[0] = [0, 0, 0, 1]  # Set the first color (index 0) to black
custom_turbo = LinearSegmentedColormap.from_list('custom_turbo', colors)
minround=0
maxround=len(card_mat.T)-1
mincards=0
maxcards=len(card_mat)-1
matrix=norm_card_mat[mincards:maxcards,minround:maxround]

plt.imshow(matrix, cmap=custom_turbo,aspect='auto')  # `cmap` sets the color map
plt.colorbar(label='Abs val')  # Add a color bar for scale
plt.gca().invert_yaxis()
# plt.xlim(minround,maxround)  # Set x-axis range from 10 to 100
plt.ylim(mincards,maxcards)   # Set y-axis range from 20 to 40
# plt.grid(axis='y')
# plt.grid(axis='x')
# plt.xticks(np.arange(0, matrix.shape[1], 1))  # x-axis ticks
# plt.yticks(np.arange(0, matrix.shape[0], 1))  # y-axis ticks

# add max cards line
# y = []
# n=0
# for i in range(maxround):
#     y.append(min(n+5,52))
#     if i%2==0:
#         n+=1
# x = np.arange(0,maxround)
# plt.plot(x, y, color='red', linewidth=1, label='Max Card Line')  # Thinner line

plt.title("Distribution of table for a two player game")
plt.xlabel("Round")
plt.ylabel("#cards on table")

plt.savefig("img2pl/"+filename+str(time.time())+".png", format='png', dpi=300)
plt.show()


# matrix = gaussian_filter(norm_card_mat[:20,:120],sigma=.5)

# # Create a meshgrid for the x and y axes
# x = np.arange(matrix.shape[1])  # Columns
# y = np.arange(matrix.shape[0])  # Rows
# x, y = np.meshgrid(x, y)

# # Create a figure and 3D axis
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111, projection='3d')
# ax.view_init(elev=30, azim=40)

# # Create the surface plot
# ax.plot_surface(x, y, matrix)
# # Add colorbar and labels
# ax.set_title("Distribution of cards for a three player game")
# ax.set_xlabel("Round")
# ax.set_ylabel("#cards for 1 player")
# ax.set_zlabel("Probability density")
# # ax.set_xlim([0, 150])  # Set range for x-axis (columns)
# # ax.set_ylim([0, 20])   # Set range for y-axis (rows)
# # fig.colorbar(ax.plot_surface(x, y, matrix, cmap='viridis'), label='Value')
# # plt.savefig(filename+str(time.time())+".png", format='png', dpi=300)
# plt.show()