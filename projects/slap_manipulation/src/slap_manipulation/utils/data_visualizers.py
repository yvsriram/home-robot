import numpy as np
import open3d as o3d

from home_robot.utils.point_cloud import numpy_to_pcd


def show_point_cloud_with_keypt_and_closest_pt(
    xyz: np.ndarray,
    rgb: np.ndarray,
    keyframe_orig: np.ndarray,
    keyframe_rot: np.ndarray,
    closest_pt: np.ndarray,
):
    """
    Method to visualize input point-cloud along with ee pose and labeled interaction point
    Args:
        xyz: (Nx3) point cloud points
        rgb: (Nx3) point cloud color
        keyframe_orig: (3x1 vector) ee/keyframe position
        keyframe_rot: (3x3 matrix) ee/keyframe orientation as rotation matrix
        closest_pt: (3x1 vector) labeled interaction point
    """
    if np.any(rgb) > 1:
        rgb = rgb / np.max(rgb)
    pcd = numpy_to_pcd(xyz, rgb)
    geoms = [pcd]
    if keyframe_orig is not None:
        coords = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.1, origin=keyframe_orig
        )
        if keyframe_rot is not None:
            coords = coords.rotate(keyframe_rot)
        geoms.append(coords)
    if closest_pt is not None:
        closest_pt_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        closest_pt_sphere.translate(closest_pt)
        closest_pt_sphere.paint_uniform_color([1, 0.706, 0])
        geoms.append(closest_pt_sphere)
    geoms.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.05, origin=np.array([0.0, 0.0, 0.0]).reshape(3, 1)
        )
    )
    o3d.visualization.draw_geometries(geoms)
