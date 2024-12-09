import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Load the data
data = pd.read_csv("hole_card_simulation_results.csv")

# Define a mapping for card values
card_value_map = {'Ace': 14, 'King': 13, 'Queen': 12, 'Jack': 11, '10': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}

# Function to extract card rank (e.g., '[2♣]' -> '2', '[A♣]' -> 'Ace')
def extract_card_rank(card_str):
    # Extract the rank (e.g., '2' from '[2♣]', 'A' from '[A♣]')
    rank = card_str.strip('[]').split(' ')[0][:-1]  # Remove suit and brackets, keep the rank
    # Map special card ranks
    if rank == 'A':  # If the rank is 'A', map it to 'Ace'
        return 'Ace'
    elif rank == 'K':  # Map 'K' to 'King'
        return 'King'
    elif rank == 'Q':  # Map 'Q' to 'Queen'
        return 'Queen'
    elif rank == 'J':  # Map 'J' to 'Jack'
        return 'Jack'
    return rank  # Return the rank as it is for other cards

# Split the 'Hole cards' column into two separate card columns
data[['Card1', 'Card2']] = data['Hole cards'].str.split(' ', expand=True)

# Extract the ranks for each card
data['Card1_rank'] = data['Card1'].apply(extract_card_rank)
data['Card2_rank'] = data['Card2'].apply(extract_card_rank)

# Drop duplicates based on the unique card pairs (Card1_rank and Card2_rank)
data_unique = data.drop_duplicates(subset=['Card1_rank', 'Card2_rank'])

# Calculate the Card_pair_value for each unique card pair
data_unique['Card1_value'] = data_unique['Card1_rank'].apply(lambda x: card_value_map.get(x, 0))
data_unique['Card2_value'] = data_unique['Card2_rank'].apply(lambda x: card_value_map.get(x, 0))
data_unique['Card_pair_value'] = data_unique['Card1_value'] + data_unique['Card2_value']

# Normalize the card pair values to use in colormap with specific range for 2-2 to Ace-Ace
norm = mcolors.Normalize(vmin=4, vmax=26)  # Set vmax=4 (2 2) and vmin=26 (Ace Ace)
cmap = cm.viridis  # Use the viridis colormap, but you can choose others

# Stretch factor for the vertices to space out the triangles
stretch_factor_x = 1.5  # Stretch the x-axis
stretch_factor_y = 1.5  # Stretch the y-axis

# Transform to special triangular coordinates with stretching
def transformed_coords(win, loss, split):
    x = (win - 0.5 * loss - 0.5 * split) * stretch_factor_x
    y = (np.sqrt(3) / 2) * (split - loss) * stretch_factor_y
    return x, y

# Function to rotate coordinates by an arbitrary angle counterclockwise about the origin (0, 0)
def rotate_point(x, y, angle_deg):
    angle_rad = np.radians(angle_deg)  # Convert degrees to radians
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)

    # Rotate the point
    x_new = cos_angle * x - sin_angle * y
    y_new = sin_angle * x + cos_angle * y

    return x_new, y_new

# Set the rotation angle (degrees)
rotation_angle = 90  # Change this value to rotate by a different angle

# Plot the ternary plot for all unique card pairs
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')

# Draw the triangle boundaries (apply rotation to the vertices)
triangle = np.array([
    transformed_coords(1, 0, 0),  # Win
    transformed_coords(0, 1, 0),  # Loss
    transformed_coords(0, 0, 1),  # Split
    transformed_coords(1, 0, 0)   # Close the triangle
])

# Rotate the triangle coordinates
triangle_rotated = np.array([rotate_point(x, y, rotation_angle) for x, y in triangle])

# Plot the rotated triangle
ax.plot(triangle_rotated[:, 0], triangle_rotated[:, 1], 'k-', lw=1)

# Draw axis lines from the center to each vertex (apply rotation to the center and vertices)
center_coords = transformed_coords(1/3, 1/3, 1/3)
center_rotated = rotate_point(center_coords[0], center_coords[1], rotation_angle)

vertices = [
    transformed_coords(1, 0, 0),  # Win
    transformed_coords(0, 1, 0),  # Loss
    transformed_coords(0, 0, 1)   # Split
]
vertices_rotated = np.array([rotate_point(x, y, rotation_angle) for x, y in vertices])

for vertex in vertices_rotated:
    ax.plot([center_rotated[0], vertex[0]], [center_rotated[1], vertex[1]], 'k-', lw=0.8, linestyle='--')

# Plot all unique data points (apply rotation to each point)
for i, row in data_unique.iterrows():
    win_prob = row['Win prob'] / (row['Win prob'] + row['Loss prob'] + row['Split prob'])
    loss_prob = row['Loss prob'] / (row['Win prob'] + row['Loss prob'] + row['Split prob'])
    split_prob = row['Split prob'] / (row['Win prob'] + row['Loss prob'] + row['Split prob'])
    
    win_coords = transformed_coords(win_prob, 0, 0)
    loss_coords = transformed_coords(0, loss_prob, 0)
    split_coords = transformed_coords(0, 0, split_prob)

    # Rotate the points
    win_rotated = rotate_point(win_coords[0], win_coords[1], rotation_angle)
    loss_rotated = rotate_point(loss_coords[0], loss_coords[1], rotation_angle)
    split_rotated = rotate_point(split_coords[0], split_coords[1], rotation_angle)

    # Scatter the rotated point with color based on the card pair value
    ax.scatter(win_rotated[0], win_rotated[1], color=cmap(norm(row['Card_pair_value'])), s=100)
    ax.scatter(loss_rotated[0], loss_rotated[1], color=cmap(norm(row['Card_pair_value'])), s=100)
    ax.scatter(split_rotated[0], split_rotated[1], color=cmap(norm(row['Card_pair_value'])), s=100)

    # Draw a thin line connecting the points (Win -> Loss -> Split -> Win)
    point_coords = [win_rotated, loss_rotated, split_rotated, win_rotated]  # Revisit 'Win' at the end to close the loop
    point_x, point_y = zip(*point_coords)  # Unzip the coordinates for plotting
    ax.plot(point_x, point_y, color=cmap(norm(row['Card_pair_value'])), lw=0.5)  # Thin line connecting the points

# Add labels at the triangle vertices (apply rotation to the labels)
ax.text(
    *rotate_point(*transformed_coords(1.03, 0, 0), rotation_angle),
    'Win',
    verticalalignment='bottom',
    horizontalalignment='center',
    fontsize=14,  # Increase the font size
    fontweight='bold'  # Make the text bold
)

ax.text(
    *rotate_point(*transformed_coords(0, 1.1, 0), rotation_angle),
    'Loss',
    verticalalignment='bottom',
    horizontalalignment='center',
    fontsize=14,  # Increase the font size
    fontweight='bold'  # Make the text bold
)

ax.text(
    *rotate_point(*transformed_coords(0, 0, 1.12), rotation_angle),
    'Split',
    verticalalignment='bottom',
    horizontalalignment='center',
    fontsize=14,  # Increase the font size
    fontweight='bold'  # Make the text bold
)

# Add gridlines inside the triangle
num_gridlines = 10  # Number of gridlines inside the triangle

for i in range(1, num_gridlines):
    # Interpolate points along the edges
    fraction = i / num_gridlines
    start1 = transformed_coords(fraction, 1 - fraction, 0)  # Parallel to Win-Loss
    end1 = transformed_coords(fraction, 0, 1 - fraction)
    start2 = transformed_coords(0, fraction, 1 - fraction)  # Parallel to Loss-Split
    end2 = transformed_coords(1 - fraction, fraction, 0)
    start3 = transformed_coords(1 - fraction, 0, fraction)  # Parallel to Split-Win
    end3 = transformed_coords(0, 1 - fraction, fraction)

    # Rotate the gridline points
    start1_rotated = rotate_point(start1[0], start1[1], rotation_angle)
    end1_rotated = rotate_point(end1[0], end1[1], rotation_angle)
    start2_rotated = rotate_point(start2[0], start2[1], rotation_angle)
    end2_rotated = rotate_point(end2[0], end2[1], rotation_angle)
    start3_rotated = rotate_point(start3[0], start3[1], rotation_angle)
    end3_rotated = rotate_point(end3[0], end3[1], rotation_angle)

    # Plot gridlines
    ax.plot([start1_rotated[0], end1_rotated[0]], [start1_rotated[1], end1_rotated[1]], 'gray', lw=0.5, linestyle='--')
    ax.plot([start2_rotated[0], end2_rotated[0]], [start2_rotated[1], end2_rotated[1]], 'gray', lw=0.5, linestyle='--')
    ax.plot([start3_rotated[0], end3_rotated[0]], [start3_rotated[1], end3_rotated[1]], 'gray', lw=0.5, linestyle='--')

# Add a color bar as the legend
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # Empty array for the colorbar
cbar = fig.colorbar(sm, ax=ax)  # Use 'ax=ax' to link the color bar with the plot
cbar.set_label("Card Pair Value (2 2 -> Ace Ace)")

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

# Show the plot
plt.show()
