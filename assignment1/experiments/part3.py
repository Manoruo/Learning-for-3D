import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh
from pytorch3d.renderer.cameras import look_at_view_transform
import numpy as np
import imageio


def render_cow_360(cow_path="data/cow.obj", image_size=256, color1=[0, 0, 1], color2=[1, 0, 0], device=None,
):
    if device is None:
        device = get_device()
    
    images = []
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices, faces = load_cow_mesh(cow_path)
    
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    
    
    # Compute alpha per vertex based on z-coordinate
    z = vertices[0, :, 2]               # shape (N_v,)
    z_min = z.min()
    z_max = z.max()
    alpha = (z - z_min) / (z_max - z_min)  # shape (N_v,)
    alpha = alpha.unsqueeze(1)  # shape (N_v, 1) needs to be vector for broadcasting --> think alpha is a percentage and needs to scale down eahc RGB array

    # Convert color1 and color2 to tensors (RGB arrays)
    color1 = torch.tensor(color1, dtype=torch.float32)  # shape (3,)
    color2 = torch.tensor(color2, dtype=torch.float32)  # shape (3,)


    color = alpha * color2 + (1 - alpha) * color1
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * color  # (1, N_v, 3)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)


    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    
    for degree in range(360):
        rot, t = look_at_view_transform(dist=3, elev=3, azim=degree, degrees=True, device=device)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(
            R=rot, T=t, fov=60, device=device
        )
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]
        rend = (np.clip(rend, 0, 1) * 255).astype(np.uint8)
        images.append(rend)
    return images

        



if __name__ == "__main__":
    
    my_images = np.array(render_cow_360())  # List of images [(H, W, 3)]
    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)

    imageio.mimsave('cow_rainbow.gif', my_images, duration=duration)
