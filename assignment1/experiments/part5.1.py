import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
from pytorch3d.renderer import TexturesVertex, PointLights, FoVPerspectiveCameras

from starter.utils import get_device, get_mesh_renderer, get_points_renderer


def render_torus_mesh(image_size=256, R=1.0, r=0.3, res=100, device=None):
    """
    Render a torus mesh parametrically.

    Args:
        image_size (int): Output image size (image_size x image_size)
        R (float): Major radius
        r (float): Minor radius
        res (int): Resolution along u and v
        device: torch device
    Returns:
        numpy.ndarray: Rendered RGB image, shape [H, W, 3], values in [0,1]
    """
    if device is None:
        device = get_device()

    # --- Parametric grid ---
    theta = torch.linspace(0, 2 * torch.pi, res)
    phi = torch.linspace(0, 2 * torch.pi, res)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")

    # --- Torus parametric equations ---
    X = (R + r * torch.cos(theta)) * torch.cos(phi) # Note: Capital R is major radius (distance from center of tube to center of torus)
    Y = (R + r * torch.cos(theta)) * torch.sin(phi) # Note: Small r is minor radius (radius of the tube)
    Z = r * torch.sin(theta)

    # X, Y, Z are all going to be shape (theta len, phi len) where each entry is a specific theta evaluated using different phi's 

    # Flatten points
    points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1) # this gives us each vertex as a row vector, shape (N, 3)
    color = (points - points.min()) / (points.max() - points.min()) # Create color gradient by min-max normalizing the points 

    # Create the point cloud
    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 3]], device=device)
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(sphere_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()

def render_torus_mesh_implictly(image_size=256, R=1, r=0.3, voxel_size=64, device=None):
    """
    Render a torus mesh implicitly.

    Args:
        image_size (int): Output image size (image_size x image_size)
        R (float): Major radius
        r (float): Minor radius
        res (int): Resolution along u and v
        device: torch device
    Returns:
        numpy.ndarray: Rendered RGB image, shape [H, W, 3], values in [0,1]
    """
    if device is None:
        device = get_device()
    
    extents = R + r + 0.1 # +0.1 to give some padding, make sure torus fits in grid
    min_value = -extents
    max_value = extents

    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (torch.sqrt(X**2 + Y**2) - R) ** 2 + Z**2 - r**2
    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))

    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value # convert from voxel coordinates (grid space) to cartesian coordinates (world space)
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device,)
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    R, T = pytorch3d.renderer.look_at_view_transform(dist=3, elev=0, azim=180)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
    rend = renderer(mesh, cameras=cameras, lights=lights)
    return rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)


def render_heart_mesh(image_size=256, res=200, size=1.0, device=None):
    """
    Render a heart mesh parametrically.

    Args:
        image_size (int): Output image size (image_size x image_size)
        res (int): Resolution along u (theta) and v (phi)
        size (float): Scaling factor for heart size
        device: torch device
    Returns:
        numpy.ndarray: Rendered RGB image, shape [H, W, 3], values in [0,1]
    """
    if device is None:
        device = get_device()

    # --- Parametric grid ---
    t = torch.linspace(0, 2 * torch.pi, res)
    u, v = torch.meshgrid(t, t, indexing="ij")  # using same resolution for simplicity

    # --- Heart parametric equations ---
    X = 16 * torch.sin(u)**3
    Y = 13 * torch.cos(u) - 5 * torch.cos(2*u) - 2 * torch.cos(3*u) - torch.cos(4*u)
    Z = torch.sin(v) * 2  # add some thickness for 3D effect

    # --- Apply scaling factor ---
    X *= size
    Y *= size
    Z *= size

    # Flatten points
    points = torch.stack([X.flatten(), Y.flatten(), Z.flatten()], dim=-1)
    color = (points - points.min()) / (points.max() - points.min())

    # Create the point cloud
    heart_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)

    # Setup camera and renderer
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(T=[[0, 0, 60]], device=device)  # pull back so heart fits
    renderer = get_points_renderer(image_size=image_size, device=device)
    rend = renderer(heart_point_cloud, cameras=cameras)
    return rend[0, ..., :3].cpu().numpy()




if __name__ == "__main__":
    device = get_device()
    img = render_torus_mesh(image_size=512, R=1.0, r=0.4, res=1000, device=device)

    # Display the rendered torus
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    img = render_torus_mesh_implictly(image_size=512, R=1.0, r=0.4, voxel_size=64, device=device)

    # Display the rendered torus
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    img = render_heart_mesh(image_size=512, res=500)

    # Display the rendered torus
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()


