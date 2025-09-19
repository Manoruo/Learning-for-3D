"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch

from starter.utils import get_device, get_points_renderer, load_cow_mesh

def sample_points_on_mesh(vertices, faces, num_samples, device):

    # Get the vertices of each face
    v0 = vertices[faces[:, 0], :] # get the first vertex of each face
    v1 = vertices[faces[:, 1], :] # get the second vertex of each face
    v2 = vertices[faces[:, 2], :] # get the third vertex of each face
    
    # compute the area of each triangle
    edge1 = v1 - v0
    edge2 = v2 - v0
    area = 1/2 * torch.norm(torch.cross(edge1, edge2, dim=1), dim=1)  # Note the magnitude of the cross product of two vectors tells you the area that two vectors span "up till their respective edges" which would be base * height (Square). Divide by 2 to get triangular area 
    
    # Normalize the areas to create a probability distribution
    area_probs = area / area.sum()
    
    # Sample face indices according to area probabilities
    sampled_face_indices = torch.multinomial(area_probs, num_samples)

    # Get the sampled faces vertices
    sampled_v0 = v0[sampled_face_indices]
    sampled_v1 = v1[sampled_face_indices]
    sampled_v2 = v2[sampled_face_indices]
    

    # Now we need to sample points within each triangle --> Use randomly uniform barycentric coordinates
    u = torch.rand(num_samples, 1, device=device)
    v = torch.rand(num_samples, 1, device=device)
    # if any combo of u and v is over, subtract 1 to get it back within the correct range
    mask = u + v > 1
    u[mask] = 1 - u[mask]
    v[mask] = 1 - v[mask]
    w = 1 - (u + v) # the remaining weight goes to w
    
    # Compute the sampled points from the verticies forming the faces using barycentric coordinates
    samples = u * sampled_v0 + v * sampled_v1 + w * sampled_v2
    
    return samples



def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.7, 0.7, 1], background_color=(1, 1, 1), device=None,
):
    if device is None:
        device = get_device()

    
    renderer = get_points_renderer(image_size=image_size, device=device)
    vertices, faces = load_cow_mesh(cow_path)
    vertices = vertices.unsqueeze(0).to(device)  # (1, N_v, 3)
    faces = faces.unsqueeze(0).to(device)  # (1, N_f, 3)

    verts = sample_points_on_mesh(vertices[0], faces[0], num_samples=1000, device=device)  # (N_v, 3)
    verts = verts.unsqueeze(0)  # (1, N_v, 3)

    textures = torch.ones_like(verts, device=device)
    textures = textures * torch.tensor(color, device=device).view(1, 1, 3)

    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=textures)

    R, T = pytorch3d.renderer.look_at_view_transform(4, 10, 0, device=device)
    cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

    rend = renderer(point_cloud, cameras=cameras)
    rend = rend.cpu().numpy()[0, ..., :3]

    return rend


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    parser.add_argument("--output_path", type=str, default="images/sampled_cow.jpg")
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    image = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    plt.imsave(args.output_path, image)
