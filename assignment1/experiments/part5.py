from starter.render_generic import load_rgbd_data
from starter.utils import unproject_depth_image, get_points_renderer, get_device

import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch

device = get_device()
print(load_rgbd_data().keys())

data = load_rgbd_data() 


all_points = []
all_rgba = []

for i in range(3):
        
    # TODO: rotate and add todevice 
    if i == 2:
        points = torch.cat(all_points, dim=0)
        rgba = torch.cat(all_rgba, dim=0)
        
    else:

        image = torch.Tensor(data[f'rgb{i+1}'])
        mask = torch.Tensor(data[f'mask{i + 1}'])
        depth = torch.Tensor(data[f'depth{i + 1}'])
        camera = data[f'cameras{i + 1}']

        points, rgba = unproject_depth_image(image, mask, depth, camera)
        
        all_points.append(points)
        all_rgba.append(rgba)

    points = points.unsqueeze(0)
    rgba = rgba.unsqueeze(0)


   

    image_size = 256
    background_color=(1, 1, 1)

    renderer = get_points_renderer(image_size=image_size, background_color=background_color)
    p_cloud = pytorch3d.structures.Pointclouds(points=points, features=rgba)
    rend = renderer(p_cloud, cameras=camera)
    img = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    plt.imsave(f'plant{i + 1}.png', img)


# union
