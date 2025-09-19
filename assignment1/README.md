# 3D Rendering and Point Cloud Visualization Notebook

This notebook demonstrates various 3D rendering and point cloud techniques using **PyTorch3D**.

---

## Setup
- Imports necessary libraries: `torch`, `pytorch3d`, `matplotlib`, `imageio`, and helper functions from `starter` modules.
- Sets up device (`CPU` or `GPU`) and output folder `hw_images/`.
- Uses **IPython** (`Image`, `HTML`) for displaying GIFs — you may need to `pip install ipython`.
---

## Question 1: Mesh Rendering

### Q1.1 Rotating Cow Mesh
- Load and texture a cow mesh.
- Render a 360° rotating GIF using `render_mesh360`.

### Q1.2 Dolly Zoom
- Apply a dolly zoom effect using `dolly_zoom`.
- Render GIF for visualizing the effect.

---

## Question 2: Primitive Meshes

### Q2.1 Tetrahedron
- Define vertices and faces of a tetrahedron.
- Render rotating GIF.

### Q2.2 Cube
- Define vertices and faces of a cube.
- Render rotating GIF.

---

## Question 3: Retexturing Cow
- Apply vertex-based gradient coloring based on z-coordinate.
- Render before and after comparison.

---

## Question 4: Camera Transforms
- Demonstrates rotating and translating the cow using `R_relative` and `T_relative`.
- Save rendered images for different transformations.

---

## Question 5: Point Clouds

### Q5.1 RGB-D Point Cloud Rendering
- Load RGB-D images and unproject to 3D points.
- Render single and combined point clouds.

### Q5.2 Parametric Point Clouds
- Torus and ellipsoid point clouds using parametric equations.
- Render animated GIFs with color gradients.

### Q5.3 Implicit Mesh Rendering
- Generate torus and ellipsoid meshes from implicit functions using marching cubes.
- Render 360° GIFs.

---

## Question 6: 3D Star
- Generate a 3D star parametric point cloud.
- Render rotating GIF with gradient coloring.

---

## Question 7: Sampling Points on Mesh
- Sample points uniformly on the cow mesh surface using barycentric coordinates.
- Compare original mesh and sampled point clouds at various resolutions.

---

## Notes
- **Meshes vs Point Clouds**:  
  - Point clouds: faster, memory-efficient, but may appear sparse.  
  - Meshes: higher visual quality with continuous surfaces, but heavier to render.
- All GIFs are saved to the `hw_images/` folder.
