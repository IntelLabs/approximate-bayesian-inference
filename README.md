# pyViewer

A simple OpenGL viewer for python with a simple SceneGraph. 
- pyglfw3 window managers or offscreen headless rendering.
- Pybullet support. Import scene and update the state.
- Custom shaders

# Installation
#### Linux
```
sudo apt install libglfw3
```

#### Windows
[Download libglfw3 precompiled binaries](https://www.glfw.org/download.html) and put the library **glfw3.dll** in 
the project path. 

# Usage

The examples folder are the best source to get started. You can learn how to:

- Create a scene
- Populate a scene with nodes
- Types of nodes
  * Text
  * Plots
  * Geometry
  * Images
  * Point Clouds
  
- Load meshes
- Move meshes
- Generate semantic segmentation images
- Generate depth images
- Read the rendered frames
- Pybullet integration
