import io, os, sys
import cProfile, pstats
import torch
import numpy as np
import trimesh
import json
import glob
import random
import pytorch3d
from pytorch3d.structures import Meshes
from plyfile import PlyData, PlyElement

from model.diff_utils.util_3d import sdf_to_mesh


class _CustomDataParallel(torch.nn.Module):
    def __init__(self, model):
        super(_CustomDataParallel, self).__init__()
        self.model = torch.nn.DataParallel(model).cuda()
        #self.model = model.cuda()

    def forward(self, *input):
        return self.model(*input)

    def __getattr__(self, name):
        #return getattr(self.model, name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model.module, name)

def sample_points(points_list, num):
    resampled_point_clouds = []
    for point_cloud in points_list:
        n_points = point_cloud.size(0)

        if n_points >= num:
            random_indices = torch.randperm(n_points)[:num]
            resampled_point_cloud = point_cloud[random_indices]
        else:
            random_indices = torch.randint(n_points, size=(num,))
            resampled_point_cloud = point_cloud[random_indices]

        resampled_point_clouds.append(resampled_point_cloud)
    return resampled_point_clouds

def get_cross_prod_mat(pVec_Arr):
    """ Convert pVec_Arr of shape (3) to its cross product matrix
    """
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def params_to_8points(box, degrees=False):
    """ Given bounding box as 7 parameters: w, l, h, cx, cy, cz, z, compute the 8 corners of the box
    """
    w, l, h, cx, cy, cz, z = box
    points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append([w.item()/2 * i, l.item()/2 * j, h.item()/2 * k])
    points = np.asarray(points)
    points = (get_rotation(z.item(), degree=degrees) @ points.T).T
    points += np.expand_dims(np.array([cx.item(), cy.item(), cz.item()]), 0)
    return points

def get_closest_furniture_to_box(box_dict, query_size):

    mses = {}
    lhw = np.array(list(box_dict.values()))
    ids = np.array(list(box_dict.keys()))
    mses = np.sum((lhw - query_size.detach().cpu().numpy()) ** 2, axis=-1)
    id_min = np.argmin(mses)

    # for i, oi in box_dict.items():
    #     l, h, w = oi[0], oi[1], oi[2]
    #     vol = l * h * w
    print("id: ", ids[id_min], np.min(mses))
    return ids[id_min]


def get_database_objects(boxes, datasize, cat_ids, classes, mesh_dir, render_boxes=False, colors=None, without_lamp=False):
    os.makedirs(mesh_dir, exist_ok=True)
    bbox_file = "/media/ymxlzgy/Data/Dataset/FRONT/cat_jid_trainval.json" if datasize == 'large' else "/media/ymxlzgy/Data/Dataset/FRONT/cat_jid_trainval_small.json"
    colors = iter(colors)
    with open(bbox_file, "r") as read_file:
        box_data = json.load(read_file)
    lamp_mesh_list = []
    trimesh_meshes = []
    raw_meshes = []
    model_base_path = "/media/ymxlzgy/Data/Dataset/FRONT/3D-FUTURE-model"
    instance_id = 1
    for j in range(0, boxes.shape[0]):
        query_size = boxes[j, 0:3]
        query_label = classes[cat_ids[j]].strip('\n')
        if query_label == '_scene_' or query_label == 'floor':
            continue
        print('cat: ', query_label)
        furniture_id = get_closest_furniture_to_box(
            box_data[query_label], query_size
        )

        model_path = os.path.join(model_base_path,furniture_id,"raw_model.obj")
        texture_path = os.path.join(model_base_path, furniture_id, "texture.png")
        color = next(colors)

        # Load the furniture and scale it as it is given in the dataset
        tr_mesh = trimesh.load(model_path, force="mesh")
        tr_mesh = trimesh.Trimesh(vertices=tr_mesh.vertices, faces=tr_mesh.faces, process=False)
        tr_mesh.visual.vertex_colors = color
        tr_mesh.visual.face_colors = color
        raw_meshes.append(tr_mesh.copy())

        tr_mesh.export(os.path.join(mesh_dir, query_label+'_'+str(cat_ids[j])+'_'+str(instance_id)+".obj"))
        instance_id += 1
        # tr_mesh.visual.material.image = Image.open(texture_path)
        theta = boxes[j, -1].item() * (np.pi / 180)
        R = np.zeros((3, 3))
        R[0, 0] = np.cos(theta)
        R[0, 2] = -np.sin(theta)
        R[2, 0] = np.sin(theta)
        R[2, 2] = np.cos(theta)
        R[1, 1] = 1.
        t = boxes[j, 3:6].detach().cpu().numpy()
        tr_mesh.vertices[...] = tr_mesh.vertices.dot(R) + t
        trimesh_meshes.append(tr_mesh)
        if query_label == 'lamp' and without_lamp:
            lamp_mesh_list.append(trimesh_meshes.pop())
        if render_boxes:
            box_points = params_to_8points_3dfront(boxes[j], degrees=True)
            trimesh_meshes.append(create_bbox_marker(box_points, tube_radius=0.006, color=color))


    return lamp_mesh_list, trimesh_meshes, raw_meshes

def get_bbox(boxes, cat_ids, classes, scene_asset_dir, store_path, colors, without_lamp=False):
    trimesh_meshes = []
    colors = iter(colors)
    lamp_mesh_list=[]
    splats_data = []
    profiler = cProfile.Profile() 
    
    for j in range(0, boxes.shape[0]):
        query_label = classes[cat_ids[j]].strip('\n')
        if query_label == '_scene_' or query_label == 'floor' or query_label == 'none':
            continue
        box_points = params_to_8points_3dfront(boxes[j], degrees=True)
        profiler = cProfile.Profile()
        profiler.enable()
        
        profiler.enable()
        box_wireframes, pcds, splats = create_bbox_marker(box_points, scene_asset_dir=scene_asset_dir, cat_id=cat_ids[j], tube_radius=0.02, color=next(colors))
        profiler.disable()
        
        trimesh_meshes.extend([box_wireframes, pcds])
        # if query_label == 'nightstand':
        #     trimesh_meshes.pop()
        if splats is not None:
            splats["cat_id"] = cat_ids[j]
            splats_data.append(splats)
        if query_label == 'lamp' and without_lamp:
            lamp_mesh_list.append(trimesh_meshes.pop())
    
    # Save or print the profiling results after the loop
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())  # Print the profiling result to the console, or save to a file

    for s in splats_data:
        print(f"Points: {s['points'].shape}, Opacities: {s['opacities'].shape}, Features DC: {s['features_dc'].shape}, Scaling: {s['scaling'].shape}, Rotation: {s['rotation'].shape}, Group: {s['cat_id'].shape}")
    
    if splats_data:
        # Combine all splats into a single GS-compliant .ply file
        combined_points = np.concatenate([s["points"] for s in splats_data])
        combined_opacities = np.concatenate([s["opacities"] for s in splats_data])
        combined_features_dc = np.concatenate([s["features_dc"] for s in splats_data])
        combined_scales = np.concatenate([s["scaling"] for s in splats_data])
        combined_rotations = np.concatenate([s["rotation"] for s in splats_data])
        combined_groups = np.concatenate([[s["cat_id"]] * len(s["points"]) for s in splats_data]) if not isinstance(s["cat_id"], np.ndarray) else np.concatenate([s["cat_id"] for s in splats_data]) 
        print(f"Combined Points: {combined_points.shape}, Opacities: {combined_opacities.shape}, Features DC: {combined_features_dc.shape}, Scaling: {combined_scales.shape}, Group: {combined_groups.shape}")

        scene_id = os.path.basename(os.path.normpath(scene_asset_dir))
        combined_output_path = os.path.join(store_path, f"combined_gs_{scene_id}.ply")
        save_gs_compliant_ply(combined_output_path, combined_points, combined_opacities,
                              combined_features_dc, None, combined_scales, combined_rotations, combined_groups)

    return lamp_mesh_list, trimesh_meshes

def fit_shapes_to_box_v2(obj, box, degrees=False):
    l, h, w, px, py, pz, angle = box
    if isinstance(l, torch.Tensor):
        l, h, w, px, py, pz, angle = l.item(), h.item(), w.item(), px.item(), py.item(), pz.item(), angle.item()
    box_points = []
    for i in [-1, 1]:
        for j in [0, 1]:
            for k in [-1, 1]:
                box_points.append([l / 2 * i, h * j, w / 2 * k])

    bounding_box = obj.bounding_box
    bottom_center = bounding_box.bounds[0] + (bounding_box.extents / 2)
    bottom_center[1] = bounding_box.bounds[0][1]
    rotation_matrix = trimesh.transformations.rotation_matrix(-np.pi/2, [0,1,0])
    translation_matrix = trimesh.transformations.translation_matrix(-bottom_center)
    transform = np.dot(translation_matrix, rotation_matrix)
    obj.apply_transform(transform)

    R = get_rotation_3dfront(angle, degree=degrees)
    R_inv = np.linalg.inv(R)
    t = np.array([px, py, pz])
    T = np.concatenate((R_inv,t.reshape(-1,1)),axis=1)
    T = np.concatenate((T,np.array([0,0,0,1]).reshape(1,-1)),axis=0)
    vertices = np.array(obj.vertices)
    shape_size = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    obj = obj.apply_scale(1 / shape_size)
    obj = obj.apply_scale([l, h, w])
    obj = obj.apply_transform(T)
    box_points = np.asarray(box_points)
    box_points = box_points.dot(R)
    box_points += np.expand_dims(t, 0)
    return box_points, obj

# def fit_shapes_to_box_v1(obj, box, degrees=False):
#     l, h, w, px, py, pz, angle = box
#     box_points = []
#     for i in [-1, 1]:
#         for j in [0, 1]:
#             for k in [-1, 1]:
#                 box_points.append([l.item() / 2 * i, h.item() * j, w.item() / 2 * k])
#
#     bounding_box = obj.bounding_box
#     bottom_center = bounding_box.bounds[0] + (bounding_box.extents / 2)
#     bottom_center[1] = bounding_box.bounds[0][1]
#     rotation_matrix = trimesh.transformations.rotation_matrix(0, [0,1,0])
#     translation_matrix = trimesh.transformations.translation_matrix(-bottom_center)
#     transform = np.dot(translation_matrix, rotation_matrix)
#     obj.apply_transform(transform)
#
#     R = get_rotation_3dfront(angle.item(), degree=degrees)
#     R_inv = np.linalg.inv(R)
#     t = np.array([px.item(), py.item(), pz.item()])
#     T = np.concatenate((R_inv,t.reshape(-1,1)),axis=1)
#     T = np.concatenate((T,np.array([0,0,0,1]).reshape(1,-1)),axis=0)
#     vertices = np.array(obj.vertices)
#     shape_size = np.max(vertices, axis=0) - np.min(vertices, axis=0)
#     obj = obj.apply_scale(1 / shape_size)
#     obj = obj.apply_scale([l.item(), h.item(), w.item()])
#     obj = obj.apply_transform(T)
#     box_points = np.asarray(box_points)
#     box_points = box_points.dot(R)
#     box_points += np.expand_dims(t, 0)
#     return box_points, obj

def trimeshes_to_pytorch3d(meshes):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    verts_list = []
    faces_list = []
    textures_list = []
    for mesh in meshes:
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)   # [V, 3]
        faces = torch.tensor(mesh.faces, dtype=torch.int64).to(device)   # [F, 3]
        verts_list.append(vertices)
        faces_list.append(faces)
        vertex_colors = torch.tensor(mesh.visual.vertex_colors, dtype=torch.float32).to(device)  # [V, 4]
        vertex_colors = vertex_colors[:, :3] / 255.
        textures_list.append(vertex_colors)
    textures=pytorch3d.renderer.Textures(verts_rgb=textures_list)


    pytorch3d_meshes = Meshes(verts=verts_list, faces=faces_list, textures=textures)

    return pytorch3d_meshes

def normalize_py3d_meshes(meshes):
    centers = meshes.verts_packed().mean(dim=0)
    verts_list_centered = [verts - centers for verts in meshes.verts_list()]

    verts_list_normalized = []
    for verts in verts_list_centered:
        max_distance = verts.norm(p=2, dim=1).max()
        verts_normalized = verts / max_distance
        verts_list_normalized.append(verts_normalized)

    normalized_meshes = Meshes(
        verts=verts_list_normalized,
        faces=meshes.faces_list(),
        textures=meshes.textures,
    )

    return normalized_meshes

def pytorch3d_to_trimesh(pytorch3d_mesh):
    trimesh_verts = pytorch3d_mesh.verts_list()[0].cpu().numpy()
    trimesh_faces = pytorch3d_mesh.faces_list()[0].cpu().numpy()
    trimesh_normals = pytorch3d_mesh.verts_normals_list()[0].cpu().numpy()
    tri_mesh = trimesh.Trimesh(vertices=trimesh_verts, faces=trimesh_faces, process=False)
    tri_mesh.vertex_normals = trimesh_normals
    tri_mesh.invert()
    return tri_mesh

# def get_generated_models_v1(boxes, shapes, cat_ids, mesh_dir, classes, render_boxes=False, colors=None, without_lamp=False):
#     colors = iter(colors)
#     trimesh_meshes = iter(shapes)
#     obj_list = []
#     lamp_mesh_list = []
#     raw_obj_list = []
#     instance_id = 1
#     for j in range(0, boxes.shape[0]):
#         query_label = classes[cat_ids[j]].strip('\n')
#         if query_label == '_scene_' or query_label == 'floor':
#             continue
#         obj = next(trimesh_meshes)
#         color = next(colors)
#         obj.visual.vertex_colors = color
#         obj.visual.face_colors = color
#         raw_obj_list.append(obj.copy())
#         obj.export(os.path.join(mesh_dir, query_label + '_' + str(cat_ids[j]) + "_" + str(instance_id) + ".obj"))
#         instance_id += 1
#
#         box_points, obj = fit_shapes_to_box_v1(obj, boxes[j], degrees=True)
#         obj_list.append(obj)
#         if query_label == 'lamp' and without_lamp:
#             lamp_mesh_list.append(obj_list.pop())
#
#
#         if render_boxes:
#             obj_list.append(create_bbox_marker(box_points, color=color))
#     return lamp_mesh_list, obj_list, raw_obj_list

def get_generated_shapes(boxes, shapes, cat_ids, classes, mesh_dir, render_boxes=False, colors=None, without_lamp=False):
    mesh_gen = sdf_to_mesh(shapes,render_all=True)
    colors = iter(colors)
    trimesh_meshes = iter([pytorch3d_to_trimesh(mesh) for mesh in mesh_gen])
    obj_list = []
    lamp_mesh_list = []
    raw_obj_list = []
    instance_id = 1
    for j in range(0, boxes.shape[0]):
        render_boxes_ = render_boxes
        query_label = classes[cat_ids[j]].strip('\n')
        if query_label == '_scene_' or query_label == 'floor':
            continue
        obj = next(trimesh_meshes)
        color = next(colors)
        obj.visual.vertex_colors = color
        obj.visual.face_colors = color
        raw_obj_list.append(obj.copy())
        obj.export(os.path.join(mesh_dir, query_label + '_' + str(cat_ids[j]) + "_" + str(instance_id)+".obj"))
        instance_id += 1

        box_points, obj = fit_shapes_to_box_v2(obj, boxes[j], degrees=True)
        obj_list.append(obj)
        # if query_label == 'bed':
        #     obj.export('/media/ymxlzgy/Data/asset/bedv2.glb')
        # if query_label == 'nightstand':
        #     obj_list.pop()
        #     render_boxes_ = False
        if query_label == 'lamp' and without_lamp:
            lamp_mesh_list.append(obj_list.pop())

        if render_boxes_:
            obj_list.append(create_bbox_marker(box_points, tube_radius=0.006, color=color))
    return lamp_mesh_list, obj_list, raw_obj_list


def get_sdfusion_models(boxes, cat_ids, classes, mesh_dir, render_boxes=False, colors=None, no_stool=False, without_lamp=False):
    sdfusion_model_path = "/media/ymxlzgy/Data/Dataset/FRONT/txt2shape_results_latest"
    mapping_full2simple = None
    obj_list = []
    colors = iter(colors)
    lamp_mesh_list = []
    raw_obj_list = []
    instance_id = 1
    for j in range(0, boxes.shape[0]):
        query_label = classes[cat_ids[j]].strip('\n')
        if query_label == '_scene_' or query_label == 'floor':
            continue
        color = next(colors)
        folder_path = os.path.join(sdfusion_model_path, query_label) if mapping_full2simple == None \
            else os.path.join(sdfusion_model_path, mapping_full2simple[query_label])
        a=random.randint(0, 100)
        print("seed: ", a)
        random.seed(a)
        model_path = random.choice(glob.glob(os.path.join(folder_path,'*.ply')))
        if no_stool and query_label == 'chair':
            assert mapping_full2simple == None
            model_path2 = random.choice(glob.glob(os.path.join(sdfusion_model_path,'stool','*.ply')))
            model_path = random.choice([model_path, model_path2])

        obj = trimesh.load(model_path)
        obj.invert()
        obj.visual.vertex_colors = color
        obj.visual.face_colors = color
        raw_obj_list.append(obj.copy())
        obj.export(os.path.join(mesh_dir, query_label + '_' + str(cat_ids[j]) + "_" + str(instance_id)+".obj"))
        instance_id += 1
        box_points, obj = fit_shapes_to_box_v2(obj, boxes[j], degrees=True)
        obj_list.append(obj)
        if query_label == 'lamp' and without_lamp:
            lamp_mesh_list.append(obj_list.pop())

        if render_boxes:
            obj_list.append(create_bbox_marker(box_points, color=color))

    return lamp_mesh_list, obj_list, raw_obj_list


def params_to_8points_3dfront(box, degrees=False, use_numpy=True):
    """ Given bounding box as 7 parameters: l, h, w, cx, cy, cz, z, compute the 8 corners of the box
    """
    l, h, w, px, py, pz, angle = box

    points = []
    for i in [-1, 1]:
        for j in [0, 1]:
            for k in [-1, 1]:
                if use_numpy:
                    points.append([l.item()/2 * i, h.item() * j, w.item()/2 * k])
                else:
                    points.append(torch.tensor([l.item()/2 * i, h.item() * j, w.item()/2 * k]))

    if use_numpy:
        points = np.asarray(points)
        points = points.dot(get_rotation_3dfront(angle.item(), degree=degrees))
        points += np.expand_dims(np.array([px.item(), py.item(), pz.item()]), 0)
    else:
        points = torch.stack(points)
        points = torch.matmul(points, torch.tensor(get_rotation_3dfront(angle.item(), degree=degrees)).to(points.device).to(points.dtype))
        points += torch.tensor([[px.item(), py.item(), pz.item()]])

    return points

def convert_to_up(points, from_up, to_up):
    """
    Convert points between 'Z' as up and 'Y' as up.

    Args:
        points (ndarray): Input points (Nx3).
        from_up (str): Current up direction ('Z' or 'Y').
        to_up (str): Desired up direction ('Z' or 'Y').

    Returns:
        ndarray: Points converted to the specified up direction.
    """
    if from_up == 'Z' and to_up == 'Y':
        R_up = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])  # Swap Y and Z
        points_transformed = points @ R_up.T    
        assert np.allclose(points_transformed, points[:, [0, 2, 1]])  # Swap Y and Z
    
    elif from_up == 'Y' and to_up == 'Z':
        R_up = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])  # Swap Y and Z
        points_transformed = points @ R_up.T
        assert np.allclose(points_transformed, points[:, [0, 2, 1]])  # Swap Y and Z
    else:
        points_transformed = points
        R_up = np.eye(3)
        
    return points_transformed, R_up


def apply_random_azimuthal_rotation(points, up_axis='Y'):
    """
    Apply a random azimuthal rotation (90-degree increments) around the specified up axis.

    Args:
        points (ndarray): Input points (Nx3).
        up_axis (int): Index of the axis representing "up" (0: X, 1: Y, 2: Z).

    Returns:
        ndarray: Rotated points.
    """
    angles = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
    angle = random.choice(angles)

    if up_axis == 'Z':  # Z-up
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle),  np.cos(angle), 0],
            [0,              0,             1]
        ])
    elif up_axis == 'Y':  # Y-up
        rotation_matrix = np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0,             1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    else:
        raise ValueError("Unsupported up axis for azimuthal rotation")

    # Rotate the points
    return points @ rotation_matrix.T, rotation_matrix


def scale_gs_to_bbox(points, bbox_size, pcd_bbox_size, maintain_aspect_ratio):
    """
    Scale points using a constructed scaling matrix S.

    Args:
        points (ndarray): Input points (Nx3).
        bbox_size (ndarray): Desired bounding box size (3,).
        pcd_bbox_size (ndarray): Current PCD bounding box size (3,).
        maintain_aspect_ratio (bool): Whether to maintain the aspect ratio.

    Returns:
        tuple:
            - ndarray: Scaled points.
            - ndarray: Scaling matrix S (3x3).
    """
    if maintain_aspect_ratio:
        scale_factor = min(bbox_size / pcd_bbox_size)  # Isotropic scaling
        S = np.eye(3) * scale_factor
    else:
        scale_factors = bbox_size / pcd_bbox_size  # Anisotropic scaling
        S = np.diag(scale_factors)

    points_scaled = points @ S.T  # Apply scaling using the scaling matrix
    return points_scaled, S


def transform_gaussian_parameters(R_up, R_az, S, T, xyz, scales, rots):
    """
    Apply the transformations R_up, R_az, S, and T to Gaussian parameters.

    Args:
        R_up (ndarray): Rotation matrix for converting to up direction (3x3).
        R_az (ndarray): Random azimuthal rotation matrix (3x3).
        S (ndarray): Scaling matrix (3x3).
        T (ndarray): Translation vector (3,).
        xyz (ndarray): Points (Nx3).
        scales (ndarray): Gaussian scales (Nx3).
        rots (ndarray): Gaussian rotations as quaternions (Nx4).

    Returns:
        tuple:
            - xyz (ndarray): Transformed points (Nx3).
            - scales (ndarray): Transformed scales (Nx3).
            - rots (ndarray): Transformed rotations as quaternions (Nx4).
    """
    # Apply R_up to points
    xyz = xyz @ R_up.T
    
    # Apply R_az to points
    xyz = xyz @ R_az.T

    # Apply scaling
    xyz = xyz @ S.T
    scales = scales + np.log(S.diagonal())

    # Apply translation
    xyz += T

    # Update rotations (as quaternions)
    def quat_multiply(q1, q2):
        """Multiply two quaternions."""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ])

    # Convert R_up and R_az to quaternions
    def rotmat_to_quat(rotmat):
        """Convert a rotation matrix to a quaternion."""
        q = np.empty(4)
        trace = np.trace(rotmat)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            q[0] = 0.25 / s
            q[1] = (rotmat[2, 1] - rotmat[1, 2]) * s
            q[2] = (rotmat[0, 2] - rotmat[2, 0]) * s
            q[3] = (rotmat[1, 0] - rotmat[0, 1]) * s
        else:
            if rotmat[0, 0] > rotmat[1, 1] and rotmat[0, 0] > rotmat[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotmat[0, 0] - rotmat[1, 1] - rotmat[2, 2])
                q[0] = (rotmat[2, 1] - rotmat[1, 2]) / s
                q[1] = 0.25 * s
                q[2] = (rotmat[0, 1] + rotmat[1, 0]) / s
                q[3] = (rotmat[0, 2] + rotmat[2, 0]) / s
            elif rotmat[1, 1] > rotmat[2, 2]:
                s = 2.0 * np.sqrt(1.0 + rotmat[1, 1] - rotmat[0, 0] - rotmat[2, 2])
                q[0] = (rotmat[0, 2] - rotmat[2, 0]) / s
                q[1] = (rotmat[0, 1] + rotmat[1, 0]) / s
                q[2] = 0.25 * s
                q[3] = (rotmat[1, 2] + rotmat[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + rotmat[2, 2] - rotmat[0, 0] - rotmat[1, 1])
                q[0] = (rotmat[1, 0] - rotmat[0, 1]) / s
                q[1] = (rotmat[0, 2] + rotmat[2, 0]) / s
                q[2] = (rotmat[1, 2] + rotmat[2, 1]) / s
                q[3] = 0.25 * s
        return q / np.linalg.norm(q)

    q_up = rotmat_to_quat(R_up)
    q_az = rotmat_to_quat(R_az)
    combined_quaternion = quat_multiply(q_up, q_az)  # Combine R_up and R_az

    # Apply the quaternion to all rots
    rots = np.array([quat_multiply(r, combined_quaternion) for r in rots])
    rots = rots / np.linalg.norm(rots, axis=-1, keepdims=True)  # Normalize

    return xyz, scales, rots


def get_dirname_by_number(root_dir, number):
    print(root_dir, number)
    if not os.path.isdir(root_dir):
        return None
    prefix = f"{number}_"
    for dirname in os.listdir(root_dir):
        if dirname.startswith(prefix):
            return dirname
    return os.listdir(root_dir)[0]

def transform_gaussian_splats(xyz, scaling, rotation, bbox_size, bbox_center, maintain_aspect_ratio):
    # Compute PCD min and max
    pcd_min = xyz.min(axis=0)
    pcd_max = xyz.max(axis=0)
    pcd_bbox_size = pcd_max - pcd_min

    # Compute scale factors
    if maintain_aspect_ratio:
        scale_factor = min(bbox_size / pcd_bbox_size)
    else:
        scale_factor = bbox_size / pcd_bbox_size

    # Convert xyz to a torch tensor
    xyz_torch = torch.tensor(xyz, dtype=torch.float)

    # Compute the current center of the Gaussian splats
    current_center = (torch.max(xyz_torch, dim=0).values + torch.min(xyz_torch, dim=0).values) / 2

    # Compute the translation to align the center with the bounding box center
    translation = torch.tensor(bbox_center, dtype=torch.float) - current_center

    # Apply the translation to the points
    transformed_points = xyz_torch + translation

    # The scaling remains unchanged!!
    rescaled_scaling = scaling

    return transformed_points, rescaled_scaling


def save_gs_compliant_ply(output_path, points, opacities, features_dc, features_rest, scaling, rotation, cat_id):
    print("save_gs_compliant_ply")
    vertex_data = []
    
    if not isinstance(cat_id, np.ndarray):
        cat_id = np.full(opacities.shape, cat_id)
    
    for i in range(len(points)):
        vertex = (
            points[i][0], points[i][1], points[i][2],
            opacities[i],
            features_dc[i][0], features_dc[i][1], features_dc[i][2],
            # *features_rest[i].flatten(),
            *scaling[i],
            *rotation[i],
            cat_id[i],
        )
        vertex_data.append(vertex)

    vertex_dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('opacity', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]
    # vertex_dtype += [(f'f_rest_{i}', 'f4') for i in range(features_rest.shape[2])]
    vertex_dtype += [(f'scale_{i}', 'f4') for i in range(scaling.shape[1])]
    vertex_dtype += [(f'rot_{i}', 'f4') for i in range(rotation.shape[1])]
    vertex_dtype += [('cat_id', 'f4'),]

    vertex_array = np.array(vertex_data, dtype=vertex_dtype)
    element = PlyElement.describe(vertex_array, 'vertex')
    PlyData([element]).write(output_path)
    print("saved gs compliant ply to ", output_path)
    

def create_bbox_marker(corner_points, scene_asset_dir, cat_id=0, color=[0, 0, 255], tube_radius=0.002, sections=4,
                       maintain_aspect_ratio=False, up_direction='Y', use_cached_plys=False):
    corner_points, _ = convert_to_up(corner_points, from_up='Y', to_up=up_direction)
    
    if up_direction == 'Y':
        y_coords = [point[1] for point in corner_points]
        min_y = min(y_coords)
        corner_points = [[x, y - min_y, z] for x, y, z in corner_points]
        
    print("create_bbox_marker")
    edges = [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]]
    bbox_edge_list = []
    for edge in edges:
        bbox_edge = trimesh.creation.cylinder(radius=tube_radius, sections=sections, segment=[corner_points[edge[0]], corner_points[edge[1]]])
        bbox_edge_list.append(bbox_edge)

    box_wireframes = trimesh.util.concatenate(bbox_edge_list)
    box_wireframes.visual.face_colors = color

    bbox_min = np.min(corner_points, axis=0)
    bbox_max = np.max(corner_points, axis=0)
    bbox_size = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2

    # Optionally insert pre-made GS assets
    asset_name = get_dirname_by_number(scene_asset_dir, cat_id)
    if asset_name and os.path.isfile(os.path.join(scene_asset_dir, asset_name, "save/last_3dgs.ply")):
        print(cat_id, asset_name)
        ####################### Visualize PCD: Superimpose with BBOXes ####################### 
        # print("####################### Visualize PCD: Superimpose with BBOXes ####################### ")
        pcd = trimesh.load(os.path.join(scene_asset_dir, asset_name, "save/last_3dgs.ply"))
        print('PCD\t\t',box_wireframes)
        pcd_points = np.array(pcd.vertices)
        pcd_points, R_up = convert_to_up(pcd_points, from_up='Z', to_up=up_direction)
        
        pcd_points, R_az = apply_random_azimuthal_rotation(pcd_points, up_direction)
        
        pcd_min = pcd_points.min(axis=0)
        pcd_max = pcd_points.max(axis=0)
        pcd_bbox_size = pcd_max - pcd_min
        pcd_points, S = scale_gs_to_bbox(pcd_points, bbox_size, pcd_bbox_size, maintain_aspect_ratio)
        
        pcd_center = pcd_points.mean(axis=0)
        translation = bbox_center - pcd_center
        pcd_transformed_points = pcd_points + translation
        
        pcd_transformed = trimesh.points.PointCloud(pcd_transformed_points)

        ####################### Visualize GS: Standalone ####################### 
        # print("####################### Visualize GS: Standalone ####################### ")
        output_path = os.path.join(scene_asset_dir, asset_name, "echoscene.ply")
        # Check if the transformed GS already exists
        if os.path.exists(output_path) and use_cached_plys:
            print(f"GS file already exists at {output_path}. Loading existing file.")
            ply_data = PlyData.read(output_path)
            
            # Extract data from existing .ply file
            xyz = np.stack((ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']), axis=1)
            opacities = np.asarray(ply_data['vertex']['opacity'])
            features_dc = np.stack((ply_data['vertex']['f_dc_0'], ply_data['vertex']['f_dc_1'], ply_data['vertex']['f_dc_2']), axis=1)

            scale_names = [p.name for p in ply_data['vertex'].properties if p.name.startswith("scale_")]
            scales = np.stack([ply_data['vertex'][name] for name in scale_names], axis=1)

            rot_names = [p.name for p in ply_data['vertex'].properties if p.name.startswith("rot")]
            rotations = np.stack([ply_data['vertex'][name] for name in rot_names], axis=1)

            splats_transformed = {
                "output_path": output_path,
                "points": xyz,
                "opacities": opacities,
                "features_dc": features_dc,
                "features_rest": None,  # Assuming extra features aren't stored
                "scaling": scales,
                "rotation": rotations,
                "cat_id": np.full(opacities.shape, cat_id),
            }
            
        else:
            pcd = trimesh.load(os.path.join(scene_asset_dir, asset_name, "save/last_3dgs.ply"))
            ply_path  = os.path.join(scene_asset_dir, asset_name, "save/last_3dgs.ply")
            ply_data = PlyData.read(ply_path)

            xyz = np.stack((ply_data['vertex']['x'], ply_data['vertex']['y'], ply_data['vertex']['z']), axis=1)
            assert np.allclose(xyz, np.array(pcd.vertices))
            opacities = np.asarray(ply_data['vertex']['opacity'])
            features_dc = np.stack((ply_data['vertex']['f_dc_0'], ply_data['vertex']['f_dc_1'], ply_data['vertex']['f_dc_2']), axis=1)
            
            features_dc = np.zeros((xyz.shape[0], 3, 1))
            features_dc[:, 0, 0] = np.asarray(ply_data.elements[0]["f_dc_0"])
            features_dc[:, 1, 0] = np.asarray(ply_data.elements[0]["f_dc_1"])
            features_dc[:, 2, 0] = np.asarray(ply_data.elements[0]["f_dc_2"])

            # extra_f_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("f_rest_")]
            # extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
            # assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
            # features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
            # for idx, attr_name in enumerate(extra_f_names):
            #     features_extra[:, idx] = np.asarray(ply_data.elements[0][attr_name])
            # # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
            # features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

            scale_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("scale_")]
            scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
            scales = np.zeros((xyz.shape[0], len(scale_names)))
            for idx, attr_name in enumerate(scale_names):
                scales[:, idx] = np.asarray(ply_data.elements[0][attr_name])

            rot_names = [p.name for p in ply_data.elements[0].properties if p.name.startswith("rot")]
            rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
            rots = np.zeros((xyz.shape[0], len(rot_names)))
            for idx, attr_name in enumerate(rot_names):
                rots[:, idx] = np.asarray(ply_data.elements[0][attr_name])
                
            features_rest = None #torch.nn.Parameter(torch.tensor(features_extra, dtype=torch.float).transpose(1, 2).contiguous())
            xyz = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float))
            scaling = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float))
            rotation = torch.nn.Parameter(torch.tensor(rots, dtype=torch.float))

            # transformed_points, rescaled_scaling = transform_gaussian_splats(
            #     xyz, scaling, rotation, bbox_size, bbox_center, maintain_aspect_ratio)
            xyz, scaling, rotation = transform_gaussian_parameters(R_up, R_az, S, translation, xyz.detach().numpy(), scaling.detach().numpy(), rotation.detach().numpy())
            
            print(xyz[:5], pcd_transformed_points[:5])
            
            print("pcd_transformed_points.shape, xyz.shape", pcd_transformed_points.shape, xyz.shape) # should be checked with np.allclose
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            splats_transformed = {
                "output_path": output_path, 
                "points": pcd_transformed_points, 
                "opacities": opacities, 
                "features_dc": features_dc, 
                "features_rest": features_rest, 
                "scaling": scaling, 
                "rotation": rotation,
                "cat_id": np.full(opacities.shape, cat_id),
            }
            save_gs_compliant_ply(**splats_transformed)
    
    else:
        pcd_transformed = None
        splats_transformed = None

    return box_wireframes, pcd_transformed, splats_transformed


def params_to_8points_no_rot(box):
    """ Given bounding box as 6 parameters (without rotation): w, l, h, cx, cy, cz, compute the 8 corners of the box.
        Works when the box is axis aligned
    """
    w, l, h, cx, cy, cz = box
    points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append([w.item()/2 * i, l.item()/2 * j, h.item()/2 * k])
    points = np.asarray(points)
    points += np.expand_dims(np.array([cx.item(), cy.item(), cz.item()]), 0)
    return points


def fit_shapes_to_box(box, shape, withangle=True):
    """ Given normalized shape, transform it to fit the input bounding box.
        Expects denormalized bounding box with optional angle channel in degrees
        :param box: tensor
        :param shape: tensor
        :param withangle: boolean
        :return: transformed shape
    """
    box = box.detach().cpu().numpy()
    shape = shape.detach().cpu().numpy()
    if withangle:
        w, l, h, cx, cy, cz, z = box
    else:
        w, l, h, cx, cy, cz = box
    # scale
    shape_size = np.max(shape, axis=0) - np.min(shape, axis=0)
    shape = shape / shape_size
    shape *= box[:3]
    if withangle:
        # rotate
        shape = (get_rotation(z, degree=True).astype("float32") @ shape.T).T
    # translate
    shape += [cx, cy, cz]

    return shape

# TODO
def refineBoxes(boxes, objs, triples, relationships, vocab):
    for idx in range(len(boxes)):
      child_box = boxes[idx]
      w, l, h, cx, cy, cz = child_box
      for t in triples:
         if idx == t[0] and relationships[t[1]] in ["supported by", "lying on", "standing on"]:
            parent_idx = t[2]
            cat = vocab['object_idx_to_name'][objs[parent_idx]].replace('\n', '')
            if cat != 'floor':
                continue
            parent_box = boxes[parent_idx]
            base = parent_box[5] + 0.0125

            new_bottom = base
            # new_h = cz + h / 2 - new_bottom
            new_cz = new_bottom + h / 2
            shift = new_cz - cz
            boxes[idx][:] = [w, l, h, cx, cy, new_cz]

            # fix adjusmets
            for t_ in triples:
                if t_[2] == t[0] and relationships[t_[1]] in ["supported by", "lying on", "standing on"]:
                    cat = vocab['object_idx_to_name'][t_[2]].replace('\n', '')
                    if cat != 'floor':
                        continue

                    w_, l_, h_, cx_, cy_, cz_ = boxes[t_[0]]
                    boxes[t_[0]][:] = [w_, l_, h_, cx_, cy_, cz_ + shift]
    return boxes


def get_rotation(z, degree=True):
    """ Get rotation matrix given rotation angle along the z axis.
    :param z: angle of z axos rotation
    :param degree: boolean, if true angle is given in degrees, else in radians
    :return: rotation matrix as np array of shape[3,3]
    """
    if degree:
        z = np.deg2rad(z)
    rot = np.array([[np.cos(z), -np.sin(z),  0],
                    [np.sin(z),  np.cos(z),  0],
                    [        0,          0,  1]])
    return rot

def get_rotation_3dfront(y, degree=True):
    if degree:
        y = np.deg2rad(y)
    rot = np.array([[np.cos(y),     0,  -np.sin(y)],
                    [       0 ,     1,           0],
                    [np.sin(y),     0,   np.cos(y)]])
    return rot


def scale_box_params(box_params, file=None, angle=False):
    assert file is not None
    stats = np.loadtxt(file)
    if len(stats) == 14: # 3*2+3*2+1*2
        min_lhw, max_lhw, min_xyz, max_xyz, min_angle, max_angle = stats[:3], stats[3:6], stats[6:9], stats[9:12], stats[12:13], stats[13:]
    else:
        raise NotImplementedError
    box_params[:3] = (box_params[:3] - min_lhw) / (max_lhw - min_lhw) # size
    box_params[:3] = 2 * box_params[:3]-1

    box_params[3:6] = (box_params[3:6] - min_xyz) / (max_xyz - min_xyz) # loc
    box_params[3:6] = 2 * box_params[3:6] - 1
    if angle:
        box_params[6:7] = (box_params[6:7] - min_angle) / (max_angle - min_angle) # angle
        box_params[6:7] = 2 * box_params[6:7] - 1

    return box_params

def preprocess_angle2sincos(angle):
    if isinstance(angle,np.ndarray):
        return np.concatenate((np.sin(angle),np.cos(angle)),axis=-1)
    elif isinstance(angle,torch.Tensor):
        return torch.concat((torch.sin(angle), torch.cos(angle)), dim=-1)
    else:
        raise NotImplementedError

def descale_box_params(normed_box_params, file=None, angle=False):
    assert file is not None
    stats = np.loadtxt(file)
    if isinstance(normed_box_params,torch.Tensor):
        stats = torch.tensor(stats,dtype=normed_box_params.dtype, device=normed_box_params.device)
    min_lhw, max_lhw, min_xyz, max_xyz, min_angle, max_angle = stats[:3], stats[3:6], stats[6:9], stats[9:12], stats[12:13], stats[13:]
    normed_box_params[:,:3] = (normed_box_params[:,:3] + 1) / 2
    normed_box_params[:,:3] = normed_box_params[:,:3] * (max_lhw - min_lhw) + min_lhw # size

    normed_box_params[:, 3:6] = (normed_box_params[:, 3:6] + 1) / 2
    normed_box_params[:,3:6] = normed_box_params[:,3:6] * (max_xyz - min_xyz) + min_xyz # loc
    if angle:
        normed_box_params[:,6:7] = (normed_box_params[:,6:7] + 1) / 2
        normed_box_params[:,6:7] = normed_box_params[:,6:7] * (max_angle - min_angle) + min_angle  # angle

    return normed_box_params

def postprocess_sincos2arctan(sincos):
    if isinstance(sincos, np.ndarray):
        assert sincos.shape[1] == 2
        return np.arctan2(sincos[0],sincos[1])
    elif isinstance(sincos,torch.Tensor):
        B, N = sincos.shape
        assert N == 2
        return torch.arctan2(sincos[:,0], sincos[:,1]).reshape(B,1)
    else:
        raise NotImplementedError

def standardize_box_params(box_params, params=7, file=None, scale=3):
    """ Normalize the box parameters for more stable learning utilizing the accumulated dataset statistics

    :param box_params: float array of shape [7] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :return: normalized box parameters array of shape [7]
    """
    assert file is not None
    if file == None:
        mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847, 0.73127955])
        std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753, 0.50347435])
    else:
        stats = np.loadtxt(file)
        if params == 6:
            mean, std = stats[0][:6], stats[1][:6]
        elif params == 7:
            mean, std = stats[0], stats[1]
        else:
            raise NotImplementedError

    return scale * ((box_params - mean) / std)


def destandardize_box_params(box_params, file=None, scale=3, params=7):
    """ Denormalize the box parameters utilizing the accumulated dataset statistics

    :param box_params: float array of shape [params] containing the box parameters
    :param scale: float scalar that scales the parameter distribution
    :param params: number of bounding box parameters. Expects values of either 6 or 7. 6 omits the angle
    :return: denormalized box parameters array of shape [params]
    """
    assert file is not None
    if file == None:
        if params == 6:
            mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847])
            std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753])
        elif params == 7:
            mean = np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847, 0.73127955])
            std = np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753, 0.50347435])
        else:
            raise NotImplementedError
    else:
        stats = np.loadtxt(file)
        if params == 6:
            mean, std = stats[0][:6], stats[1][:6]
        else:
            mean, std = stats[0], stats[1]
    return (box_params * std) / scale + mean


def batch_torch_destandardize_box_params(box_params, file=None, scale=3, params=7):
    """ Denormalize the box parameters utilizing the accumulated dataset statistics

    :param box_params: float tensor of shape [N, 6] containing the 6 box parameters, where N is the number of boxes
    :param scale: float scalar that scales the parameter distribution
    :return: float tensor of shape [N, 6], the denormalized box parameters
    """
    assert file is not None
    if file == None:
        if params == 6:
            mean = torch.from_numpy(np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847]).reshape(1,-1)).cuda()
            std = torch.from_numpy(np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753]).reshape(1,-1)).cuda()
        elif params == 7:
            mean = torch.from_numpy(np.array([1.3827214, 1.309359, 0.9488993, -0.12464812, 0.6188591, -0.54847, 0.73127955]).reshape(1,-1)).cuda()
            std = torch.from_numpy(np.array([1.7797655, 1.657638, 0.8501885, 1.9160025, 2.0038228, 0.70099753, 0.50347435]).reshape(1,-1)).cuda()
        else:
            raise NotImplementedError
    else:
        stats = np.loadtxt(file)
        if params == 6:
            mean, std = torch.from_numpy(stats[0][:6].reshape(1,-1)).cuda(), torch.from_numpy(stats[1][:6].reshape(1,-1)).cuda()
        else:
            mean, std = torch.from_numpy(stats[0].reshape(1,-1)).cuda(), torch.from_numpy(stats[1].reshape(1,-1)).cuda()

    return (box_params * std) / scale + mean


def bool_flag(s):
    """Helper function to make argparse work with the input True and False.
    Otherwise it reads it as a string and is always set to True.

    :param s: string input
    :return: the boolean equivalent of the string s
    """
    if s == '1' or s == 'True':
      return True
    elif s == '0' or s == 'False':
      return False
    msg = 'Invalid value "%s" for bool flag (should be 0, False or 1, True)'
    raise ValueError(msg % s)
