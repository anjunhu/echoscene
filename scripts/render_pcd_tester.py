import trimesh
import pyrender
import numpy as np

def render_img(trimesh_objects):
    scene = pyrender.Scene()
    renderer = pyrender.OffscreenRenderer(viewport_width=256, viewport_height=256)
    for obj in trimesh_objects:
        if isinstance(obj, trimesh.Trimesh):
            pyrender_mesh = pyrender.Mesh.from_trimesh(obj, smooth=False)
            scene.add(pyrender_mesh)
        elif isinstance(obj, trimesh.PointCloud):
            # Handle PointCloud objects
            vertices = obj.vertices
            colors = obj.colors if hasattr(obj, "colors") else None
            pyrender_mesh = pyrender.Mesh.from_points(vertices, colors=colors, point_size=5)
        else:
            raise TypeError(f"Unsupported object type: {type(obj)}")

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 2)

    # set up positions and the origin
    camera_location = np.array([0.0, 8.0, 0.0])  # y axis
    look_at_point = np.array([0.0, 0.0, 0.0])
    up_vector = np.array([0.0, 0.0, -1.0])  # -z axis

    camera_direction = (look_at_point - camera_location) / np.linalg.norm(look_at_point - camera_location)
    right_vector = np.cross(camera_direction, up_vector)
    up_vector = np.cross(right_vector, camera_direction)

    camera_pose = np.identity(4)
    camera_pose[:3, 0] = right_vector
    camera_pose[:3, 1] = up_vector
    camera_pose[:3, 2] = -camera_direction
    camera_pose[:3, 3] = camera_location
    scene.add(camera, pose=camera_pose)

    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light, pose=camera_pose)

    point_light = pyrender.PointLight(color=np.ones(3), intensity=20.0)
    scene.add(point_light, pose=camera_pose)
    color, depth = renderer.render(scene)
    return color

# Create a Trimesh object
tri_vertices = np.random.rand(1000, 3)
tri_faces = np.random.randint(0, 1000, (500, 3))
trimesh_obj = trimesh.Trimesh(vertices=tri_vertices, faces=tri_faces)

# Create a PointCloud object
pcd_vertices = np.random.rand(500, 3)
pcd_colors = np.random.randint(0, 255, (500, 4))  # RGBA colors
point_cloud_obj = trimesh.PointCloud(vertices=pcd_vertices, colors=pcd_colors)

# Render both objects
rendered_image = render_img([trimesh_obj, point_cloud_obj])

# Visualize the rendered image (optional)
import matplotlib.pyplot as plt
plt.imshow(rendered_image)
plt.axis('off')
plt.show()
