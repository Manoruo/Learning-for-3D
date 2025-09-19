"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_mesh_renderer, load_cow_mesh
from pytorch3d.renderer.cameras import look_at_view_transform
import numpy as np
import imageio

def render_tetrahedral(
        image_size=256, color=[0.7, 0.7, 1], device=None,
):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    # Get the vertices, faces, and textures.
    vertices = torch.tensor([
    [0.0, 0.0, 0.0],   # v0
    [1.0, 0.0, 0.0],   # v1
    [0, 0, 1.0],   # v2
    [0.5, 1.0, 0.5],   # v3 (apex)
    ], dtype=torch.float32)

    faces = torch.tensor([
        [0, 1, 2],  # base
        [0, 1, 3],  # side 1
        [1, 2, 3],  # side 2
        [2, 0, 3],  # side 3
    ], dtype=torch.int64)

    
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    mesh = mesh.to(device)

    # Prepare the camera:
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Place a point light in front of the cow.
    lights = pytorch3d.renderer.PointLights(device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
    # The .cpu moves the tensor to GPU (if needed).
    return rend, mesh

def render_cube(
        image_size=256, color=[0.7, 0.7, 1], device=None,
):
    if device is None:
        device = get_device()

    renderer = get_mesh_renderer(image_size=image_size)

    # Cube vertices (0,0,0) to (1,1,1)
    vertices = torch.tensor([
        [0, 0, 0],  # v0
        [1, 0, 0],  # v1
        [1, 1, 0],  # v2
        [0, 1, 0],  # v3
        [0, 0, 1],  # v4
        [1, 0, 1],  # v5
        [1, 1, 1],  # v6
        [0, 1, 1],  # v7
    ], dtype=torch.float32)

    # Cube faces (triangles)
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # bottom
        [4, 5, 6], [4, 6, 7],  # top
        [0, 1, 5], [0, 5, 4],  # front
        [1, 2, 6], [1, 6, 5],  # right
        [2, 3, 7], [2, 7, 6],  # back
        [3, 0, 4], [3, 4, 7],  # left
    ], dtype=torch.int64)

    vertices = vertices.unsqueeze(0)  # (1, 8, 3)
    faces = faces.unsqueeze(0)        # (1, 12, 3)
    textures = torch.ones_like(vertices) * torch.tensor(color)  # single-color

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    ).to(device)

    # Camera
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(
        R=torch.eye(3).unsqueeze(0), T=torch.tensor([[0, 0, 3]]), fov=60, device=device
    )

    # Lights
    lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

    rend = renderer(mesh, cameras=cameras, lights=lights)
    rend = rend.cpu().numpy()[0, ..., :3]  # (H, W, 3)
    return rend, mesh

def render360(mesh, image_size=256, device=None):
    if device is None:
        device = get_device()
    
    images = []
    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()


    image, mesh = render_tetrahedral(image_size=args.image_size)
    images = render360(mesh, args.image_size)
    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave('tetrahedral.gif', images, duration=duration)

    image, mesh = render_cube(image_size=args.image_size)
    images = render360(mesh, args.image_size)
    duration = 1000 // 15  # Convert FPS (frames per second) to duration (ms per frame)
    imageio.mimsave('cube.gif', images, duration=duration)


