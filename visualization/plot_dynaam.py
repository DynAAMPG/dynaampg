import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import matplotlib.colors as mcolors
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
from utils import *

plt.rcParams['font.family'] = 'Times New Roman'

def draw_arcface_plot(ax, dataset='iscx', C=3, beta=6, legend_columns=2):
    file_path = os.path.join(SAVED_MARGINS_DIR, 'iscx.xlsx')
    title = ''

    if dataset == 'iscx':
        file_path = os.path.join(SAVED_MARGINS_DIR, 'iscx.xlsx')
        title = 'ISCX-VPN Dataset'
    elif dataset == 'vnat':
        file_path = os.path.join(SAVED_MARGINS_DIR, 'vnat.xlsx')
        title = 'VNAT Dataset'
    elif dataset == 'tor':
        file_path = os.path.join(SAVED_MARGINS_DIR, 'tor.xlsx')
        title = 'ISCX-Tor Dataset'

    # Load the dataset statistics from the uploaded Excel file
    df = pd.read_excel(file_path)

    # Extract classes and number of instances
    classes = df['class_name'][0:].values
    num_instances = df['n_instances'][0:].astype(int).values

    # Calculate total number of instances
    total_instances = sum(num_instances)

    # Parameters for ArcFace
    num_classes = len(classes)
    radius = 10  # Radius of circle representing embeddings
    circle_radius = 0.5  # Radius of the sample circles
    centroid_offset = 2  # Offset distance for additional centroids
    min_margin_between_classes_deg = 5  # Minimum margin angle in degrees between classes
    min_margin_between_classes = np.deg2rad(min_margin_between_classes_deg)  # Convert to radians
    alpha_main_centroid = 1.0  # Opacity for main centroid
    alpha_other_centroids = 0.5  # Opacity for other centroids
    fill_opacity = 0.2  # Opacity for the fill color of each class
    sample_point_size = 8 # Size of the sample points
    line_width = 3.0  # Line width for boundary lines
    highlight_opacity = 0.1  # Opacity for highlighting angles between class boundaries
    highlight_color = 'black'  # Color for highlighting angles between class boundaries
    highlight_radius_odd = 3  # Radius for odd highlights
    highlight_radius_even = 3  # Radius for even highlights
    highlight_line_width = 1.0  # Line width for highlighting angles
    margin_label_offset = 2.2  # Offset distance to move margin labels away from the origin
    angle_label_color = 'black'  # Color for angle labels
    highlight_centroid_angle_color = 'black'  # Color for highlighting angles between main centroid and ending boundary
    angle_label_offset = 1.5 # Offset value for positioning angle labels
    font_size = 18
    sample_opacity = 0.2
    C = C  # Number of subcenters including the main centroid

    # Generate angles for each class centroid, with added dynamic margin between classes
    max_instances = max(num_instances)
    dynamic_margin_scaling = np.pi / beta  # Maximum additional margin scaling for dynamic margins
    dynamic_margins = min_margin_between_classes + ((max_instances - num_instances) / max_instances) * dynamic_margin_scaling  # Dynamic margin inversely proportional to number of samples

    angles = np.cumsum(dynamic_margins) % (2 * np.pi)
    centroids = np.array([np.cos(angles), np.sin(angles)]) * radius


    # Create a list variable for class margins
    class_margins = (num_instances / total_instances) * (2 * np.pi - sum(dynamic_margins))  # Adjust to account for margins

    # Generate a color map for the classes
    colors = [plt.get_cmap('tab20')(i % 20) for i in range(num_classes)]
    colors = ['#e6194B', '#3cb44b', '#b19a00', '#4363d8', '#f58231', '#911eb4', '#0097b6', '#f032e6', '#668b00', '#000075', '#469990', '#800000', '#9A6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff', '#000000']

    # Calculate margins based on number of instances (inverse relationship)
    total_instances = sum(num_instances)
    margins = class_margins  # Use the class_margins list variable


    

    # Plot the decision boundaries and the angular margin
    start_angle = 0  # Start angle for plotting each class
    for i, (angle, class_name, margin, num_samples) in enumerate(zip(angles, classes, margins, num_instances)):
        # Generate main centroid and C-1 additional centroids spread around the main centroid within the same class
        main_centroid = [radius * np.cos(start_angle + margin / 2), radius * np.sin(start_angle + margin / 2)]
        subcenters = [main_centroid]
        for j in range(1, C):
            # Alternate the offset direction to spread centroids on both sides of the main centroid
            direction = -1 if j % 2 == 0 else 1
            offset = direction * normalize_std(0.2, 0.5) * margin
            subcenter = [radius * np.cos(start_angle + margin / 2 + offset), radius * np.sin(start_angle + margin / 2 + offset)]
            subcenters.append(subcenter)
        centroids = np.array(subcenters)
        
        # Fill the pie area for each class with a lighter variation of the corresponding color
        theta1 = start_angle
        theta2 = start_angle + margin
        margin_arc = np.linspace(theta1, theta2, 100)
        x_arc = np.concatenate(([0], radius * np.cos(margin_arc), [0]))
        y_arc = np.concatenate(([0], radius * np.sin(margin_arc), [0]))
        lighter_color = mcolors.to_rgba(colors[i], alpha=fill_opacity)  # Lighter variation with reduced alpha
        ax.fill(x_arc, y_arc, color=lighter_color)
        
        # Highlight the angle between the ending boundary of the current class and the starting boundary of the next class
        next_start_angle = start_angle + margin + dynamic_margins[i]
        highlight_arc = np.linspace(theta2, next_start_angle, 100)
        highlight_radius = highlight_radius_odd if i % 2 == 0 else highlight_radius_even  # Alternate highlight radius
        x_highlight = highlight_radius * np.cos(highlight_arc)
        y_highlight = highlight_radius * np.sin(highlight_arc)
        ax.plot(x_highlight, y_highlight, color=highlight_color, linewidth=highlight_line_width)

        # Label these highlights as m1=2.5degree, replace 2.5 degree with the actual corresponding margin value in degrees
        mid_angle = (theta2 + next_start_angle) / 2
        label_x = (highlight_radius + margin_label_offset) * np.cos(mid_angle)
        label_y = (highlight_radius + margin_label_offset) * np.sin(mid_angle)
        margin_value_deg = np.degrees(next_start_angle - theta2)
        rotation_angle = np.degrees(mid_angle)
        ax.text(label_x, label_y, f'm$_{{{i + 1}}}$={margin_value_deg:.1f}°', fontsize=font_size, ha='center', va='center', color='black', rotation=rotation_angle)

        # Draw class centroids with unique color
        for idx, centroid in enumerate(centroids):
            alpha_value = alpha_main_centroid if idx == 0 else alpha_other_centroids  # Set opacity based on centroid type
            ax.plot(centroid[0], centroid[1], 'o', color=colors[i], alpha=alpha_value, label=f'{class_name}' if idx == 0 else None)
        
        # Draw non-overlapping angular margin boundaries
        ax.plot(radius * np.cos(margin_arc), radius * np.sin(margin_arc), '--', color=colors[i])
        
        # Calculate number of sample points based on the provided number of instances
        arc_extent = theta2 - theta1
        num_sample_points = num_samples  # Use the number of instances from the dataset as the number of sample points
        
        # Draw multiple sample points at the edge for each class with unique color
        for j in range(num_sample_points):
            sample_angle = theta1 + j * (arc_extent / (num_sample_points - 1)) if num_sample_points > 1 else theta1
            sample_x = (radius + circle_radius) * np.cos(sample_angle)
            sample_y = (radius + circle_radius) * np.sin(sample_angle)
            ax.plot(sample_x, sample_y, 'o', color=colors[i], markersize=sample_point_size, alpha=sample_opacity)
        
        # Draw line representing decision boundary for each centroid
        for idx, centroid in enumerate(centroids):
            alpha_value = alpha_main_centroid if idx == 0 else alpha_other_centroids  # Set opacity for lines corresponding to centroids
            boundary_x = [0, centroid[0]]
            boundary_y = [0, centroid[1]]
            ax.plot(boundary_x, boundary_y, ':', color=colors[i], alpha=alpha_value)

        # Draw and label angle between the main centroid and the ending boundary as theta1, theta2, etc.
        if len(centroids) > 0:
            main_centroid_angle = start_angle + margin / 2
            theta_end_angle = theta2
            theta_mid_angle = (main_centroid_angle + theta_end_angle) / 2
            angle_label_x = (radius + angle_label_offset) * np.cos(theta_mid_angle)
            angle_label_y = (radius + angle_label_offset) * np.sin(theta_mid_angle)
            angle_value_deg = np.degrees(theta_end_angle - main_centroid_angle)
            ax.text(angle_label_x, angle_label_y, f'$\\theta_{{{i + 1}}}$={angle_value_deg:.1f}°', fontsize=font_size, ha='center', va='center', color=angle_label_color, rotation=np.degrees(theta_mid_angle) - 90)
            
            # Highlight the angle between the main centroid and the ending boundary of each corresponding class
            highlight_arc_centroid = np.linspace(main_centroid_angle, theta_end_angle, 100)
            x_highlight_centroid = (radius - 1) * np.cos(highlight_arc_centroid)
            y_highlight_centroid = (radius - 1) * np.sin(highlight_arc_centroid)
            ax.plot(x_highlight_centroid, y_highlight_centroid, color=highlight_centroid_angle_color, linewidth=highlight_line_width)

        # Draw boundary lines for each class
        boundary_arc = np.linspace(theta1, theta2, 100)
        boundary_x = (radius + circle_radius) * np.cos(boundary_arc)
        boundary_y = (radius + circle_radius) * np.sin(boundary_arc)
        ax.plot(boundary_x, boundary_y, '-', color=colors[i], linewidth=line_width)
        
        # Update start_angle for next class
        start_angle += margin + dynamic_margins[i]

    # Set plot limits and labels
    ax.set_xlim(-radius - 2, radius + 2)
    ax.set_ylim(-radius - 2, radius + 2)
    ax.set_title(title, fontsize=font_size + 4, y=0, pad=-30)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', borderaxespad=0., fontsize=font_size, ncol=legend_columns, columnspacing=0.2, labelspacing=0.2)

    # Hide border and tick lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    


if __name__ == '__main__':
    fig, axs = plt.subplots(1, 3, figsize=(22, 13))  # Create a figure with 3 subplots arranged horizontally
    draw_arcface_plot(axs[0], dataset='iscx', C=3, beta=6, legend_columns=3)
    draw_arcface_plot(axs[1], dataset='vnat', C=3, beta=6, legend_columns=2)
    draw_arcface_plot(axs[2], dataset='tor', C=3, beta=12, legend_columns=4)
    plt.subplots_adjust(wspace=0.1)
    plt.tight_layout(pad=0)
    plt.savefig('visualization/fig_arcface_results.pdf')
    plt.show()

    
