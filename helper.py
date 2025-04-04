import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math

def plot_image_grid(folder_path):
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
    image_files = [f for f in os.listdir(folder_path) if os.path.splitext(f)[1].lower() in image_extensions]
    
    n_images = len(image_files)
    grid_size = math.ceil(math.sqrt(n_images))
    
    fig = plt.figure(figsize=(15, 15))
    

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(folder_path, img_file)
        try:
            img = Image.open(img_path)
            ax = fig.add_subplot(grid_size, grid_size, i+1)
            ax.imshow(np.array(img))
            ax.set_title(f"{img_file}\n{img.width}x{img.height}", fontsize=8)
            ax.axis('off')
            
        except Exception as e:
            print(f"Error loading {img_file}: {e}")
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()