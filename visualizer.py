import open3d as o3d

def generic_o3d_vis(geometries):
    if not isinstance(geometries, (list, tuple)):
        geometries = [geometries]
    vis = o3d.visualization.Visualizer() # type: ignore
    vis.create_window()
    for geometry in geometries:
        vis.add_geometry(geometry)
    vis.get_render_option().background_color = np.array([0.0, 0.0, 0.0])

    view_ctl = vis.get_view_control()
    cam_params = view_ctl.convert_to_pinhole_camera_parameters()
    cam_params.extrinsic = np.eye(4)
    view_ctl.convert_from_pinhole_camera_parameters(cam_params)

    vis.run()
    vis.destroy_window()


import matplotlib.pyplot as plt
import cv2
import numpy as np

class ImageVisualizer:
    def __init__(self):
        pass

    def depth(self, depth_image):
        global_min = 0.0
        global_max = 225.0

        depth_clipped = np.clip(depth_image, global_min, global_max)
        normalized = ((depth_clipped - global_min) / (global_max - global_min + 1e-8)) * 255
        img_normalized = normalized.astype(np.uint8)

        img_colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
        plt.imshow(img_colored)
        plt.axis('off')
        plt.show()

    def depth_corrected(self, depth_image):
        global_min = depth_image.min()
        global_max = depth_image.max()

        depth_clipped = np.clip(depth_image, global_min, global_max)
        normalized = ((depth_clipped - global_min) / (global_max - global_min + 1e-8)) * 255
        img_normalized = normalized.astype(np.uint8)

        img_colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)
        plt.imshow(img_colored)
        plt.axis('off')
        plt.show()
    
    def color(self, color_image):
        plt.imshow(color_image)
        plt.axis('off')
        plt.show()

    def depth_color_side_by_side(self, depth_image, color_image):
        global_min = 0.0
        global_max = 225.0

        depth_clipped = np.clip(depth_image, global_min, global_max)
        normalized = ((depth_clipped - global_min) / (global_max - global_min + 1e-8)) * 255
        img_normalized = normalized.astype(np.uint8)

        img_colored = cv2.applyColorMap(img_normalized, cv2.COLORMAP_JET)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img_colored)
        ax[0].set_title('Depth Image')
        ax[0].axis('off')

        ax[1].imshow(color_image)
        ax[1].set_title('Color Image')
        ax[1].axis('off')

        plt.show()

    def depth_depth_side_by_side(self, depth_image_1, depth_image_2):
        global_min = 0.0
        global_max = 225.0

        depth_clipped_1 = np.clip(depth_image_1, global_min, global_max)
        normalized_1 = ((depth_clipped_1 - global_min) / (global_max - global_min + 1e-8)) * 255
        img_normalized_1 = normalized_1.astype(np.uint8)

        img_colored_1 = cv2.applyColorMap(img_normalized_1, cv2.COLORMAP_JET)

        depth_clipped_2 = np.clip(depth_image_2, global_min, global_max)
        normalized_2 = ((depth_clipped_2 - global_min) / (global_max - global_min + 1e-8)) * 255
        img_normalized_2 = normalized_2.astype(np.uint8)

        img_colored_2 = cv2.applyColorMap(img_normalized_2, cv2.COLORMAP_JET)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(img_colored_1)
        ax[0].set_title('Depth Image 1')
        ax[0].axis('off')

        ax[1].imshow(img_colored_2)
        ax[1].set_title('Depth Image 2')
        ax[1].axis('off')

        plt.show()

    def color_color_side_by_side(self, color_image_1, color_image_2):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(color_image_1)
        ax[0].set_title('Color Image 1')
        ax[0].axis('off')

        ax[1].imshow(color_image_2)
        ax[1].set_title('Color Image 2')
        ax[1].axis('off')

        plt.show()