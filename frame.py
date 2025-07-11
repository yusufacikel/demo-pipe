from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np
from pixels import PixelList, Pixel
from scipy.optimize import least_squares

WIDTH = 848
HEIGHT = 480
FOCAL_LENGTH_X = 422.068
FOCAL_LENGTH_Y = 424.824
PRINCIPAL_POINT_X = 404.892
PRINCIPAL_POINT_Y = 260.621
camera_matrix = np.array([
    [FOCAL_LENGTH_X, 0, PRINCIPAL_POINT_X],
    [0, FOCAL_LENGTH_Y, PRINCIPAL_POINT_Y],
    [0, 0, 1]
], dtype=np.float32)



@dataclass
class Frame:
    frame_idx: int
    depth_path: Path
    color_path: Path
    raw_timestamp: int
    recording_time_str: str

    def __post_init__(self):
        self.gray_img: np.ndarray = cv2.cvtColor(cv2.imread(str(self.color_path)), cv2.COLOR_BGR2GRAY)
        self.color_img: np.ndarray = cv2.cvtColor(cv2.imread(str(self.color_path)), cv2.COLOR_BGR2RGB)
        self.raw_depth_img: np.ndarray = cv2.imread(str(self.depth_path), cv2.IMREAD_ANYDEPTH).astype(np.uint16)
        self.all_pixels: PixelList = self._extract_pixels()
        self.extracted_features = self._detect_ORBs()

    def _extract_pixels(self) -> PixelList:
        pixels_list = PixelList()

        mask = self.raw_depth_img > 0
        v_coords, u_coords = np.where(mask)

        for u, v in zip(u_coords, v_coords):
            color_tuple = tuple(self.color_img[v, u].astype(int))
            depth_value = float(self.raw_depth_img[v, u])
            
            pixels_list.append(Pixel(
                self.raw_timestamp, 
                self.recording_time_str, 
                int(u), 
                int(v), 
                color_tuple,
                depth_value
            ))
        
        self.all_pixels = pixels_list
        self._classify_pixels()

        return pixels_list
    
    def _classify_pixels(self):
        ref_center_2d, ref_radius_2d, ref_depth = self._get_reference_circle()
        for pixel in self.all_pixels.pixels:
            radial_distance = np.sqrt((pixel.u - ref_center_2d[0]) ** 2 + (pixel.v - ref_center_2d[1]) ** 2)
            if radial_distance <= ref_radius_2d:
                pixel.set_background_flag(True, ref_depth)                
            else:
                pixel.set_background_flag(False, ref_depth)

    def _get_reference_circle(self):
        ref_depth_value = self.raw_depth_img.max()
        extracted_pixels = self.all_pixels.get_pixels_with_raw_depth(ref_depth_value)
        extracted_depth_img = extracted_pixels.create_raw_depth_img()
        ref_center_2d, ref_radius_2d, ref_depth = fit_circle_to_depth_image(extracted_depth_img)
        return ref_center_2d, ref_radius_2d, ref_depth

    def _filter_pixels(self, fg_min_depth: float, bg_min_depth: float) -> PixelList:
        fg_pixels = self.all_pixels.get_foreground_pixels().get_pixels_in_range_raw_depth(fg_min_depth,226)
        bg_pixels = self.all_pixels.get_background_pixels().get_pixels_in_range_raw_depth(bg_min_depth,226)
        filtered_pixels = PixelList()
        filtered_pixels.pixels = fg_pixels.pixels + bg_pixels.pixels
        return filtered_pixels

    def _detect_ORBs(self):
            orb = cv2.ORB.create()
            mask = (self.raw_depth_img > 0).astype(np.uint8) * 255
            keypoints, descriptors = orb.detectAndCompute(self.gray_img, mask)
            points3d = cv2.rgbd.depthTo3d(self.raw_depth_img, camera_matrix)
            
            xyz_list = []
            for kp in keypoints:
                x, y = int(kp.pt[0]), int(kp.pt[1])
                xyz = points3d[int(y), int(x)]
                xyz_list.append(xyz)

            extracted_features = list(zip(keypoints, descriptors, xyz_list))

            return extracted_features
    
    def set_pose(self, pose: np.ndarray):
        self.pose = pose

def fit_circle_to_depth_image(depth_img: np.ndarray):
    depth_norm = cv2.normalize(depth_img, np.zeros_like(depth_img, dtype=np.uint8), 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_uint8 = depth_norm.astype(np.uint8)

    _, binary_img = cv2.threshold(depth_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_1 = np.ones((5, 5), np.uint8)
    kernel_2 = np.ones((13, 13), np.uint8)
    dilated = cv2.dilate(binary_img, kernel_2, iterations=1)
    eroded = cv2.erode(dilated, kernel_1, iterations=3)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_points = np.vstack([cnt.squeeze() for cnt in contours if cnt.shape[0] > 4])

    X, Y = all_points[:, 0], all_points[:, 1]

    def calc_R(xc, yc):
        return np.sqrt((X - xc)**2 + (Y - yc)**2)

    def cost(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    x0 = [np.mean(X), np.mean(Y)] # type: ignore
    res = least_squares(cost, x0=x0)

    center_2d = tuple(res.x)
    radius_2d = calc_R(*center_2d).mean()

    valid_depth_mask = depth_img > 0
    mean_depth = np.mean(depth_img[valid_depth_mask])

    return center_2d, radius_2d, mean_depth


