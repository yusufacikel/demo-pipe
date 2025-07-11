from dataclasses import dataclass
import numpy as np
import open3d as o3d



WIDTH = 848
HEIGHT = 480
FOCAL_LENGTH_X = 422.068
FOCAL_LENGTH_Y = 424.824
PRINCIPAL_POINT_X = 404.892
PRINCIPAL_POINT_Y = 260.621



@dataclass 
class PixelList:
    def __post_init__(self):
        self.pixels: list[Pixel] = []

    def append(self, pixel: 'Pixel'):
        self.pixels.append(pixel)
    
    def __len__(self):
        return len(self.pixels)

    def get_pixels_in_range_raw_depth(self, min_depth: float, max_depth: float) -> 'PixelList':
        filtered_pixels = PixelList()
        for pixel in self.pixels:
            if min_depth <= pixel.raw_depth <= max_depth:
                filtered_pixels.append(pixel)
        return filtered_pixels
    
    def get_pixels_with_raw_depth(self, depth: float) -> 'PixelList':
        filtered = PixelList()
        for pixel in self.pixels:
            if pixel.raw_depth == depth:
                filtered.append(pixel)
        return filtered
    
    def get_pixels_in_range_corrected_depth(self, min_depth: float, max_depth: float) -> 'PixelList':
        filtered_pixels = PixelList()
        for pixel in self.pixels:
            if min_depth <= pixel.depth_corrected_position[2] <= max_depth:
                filtered_pixels.append(pixel)
        return filtered_pixels
    
    def get_pixels_with_corrected_depth(self, depth: float) -> 'PixelList':
        filtered = PixelList()
        for pixel in self.pixels:
            if pixel.depth_corrected_position[2] == depth:
                filtered.append(pixel)
        return filtered

    def get_background_pixels(self) -> 'PixelList':
        filtered = PixelList()
        for pixel in self.pixels:
            if pixel.is_bg == True:
                filtered.append(pixel)
        return filtered

    def get_foreground_pixels(self) -> 'PixelList':
        filtered = PixelList()
        for pixel in self.pixels:
            if pixel.is_bg == False:
                filtered.append(pixel)
        return filtered
    
    def create_color_img(self) -> np.ndarray:
        color_image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        for pixel in self.pixels:
            if 0 <= pixel.v < HEIGHT and 0 <= pixel.u < WIDTH:
                color_image[pixel.v, pixel.u] = pixel.color
        return color_image
    
    def create_raw_depth_img(self) -> np.ndarray:
        depth_image = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        for pixel in self.pixels:
            if 0 <= pixel.v < HEIGHT and 0 <= pixel.u < WIDTH:
                depth_image[pixel.v, pixel.u] = pixel.raw_depth
        return depth_image
    
    def create_depth_corrected_depth_img(self) -> np.ndarray:
        depth_image = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
        for pixel in self.pixels:
            if 0 <= pixel.v < HEIGHT and 0 <= pixel.u < WIDTH:
                depth_image[pixel.v, pixel.u] = pixel.depth_corrected_position[2]
        return depth_image
    
    def create_raw_pcd(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        
        points = []
        colors = []
        for pixel in self.pixels:
            points.append(pixel.raw_position)
            colors.append(np.array(pixel.color) / 255.0)
        
        if points:
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return pcd
    
    def create_depth_corrected_pcd(self) -> o3d.geometry.PointCloud:
        pcd = o3d.geometry.PointCloud()
        
        points = []
        colors = []
        for pixel in self.pixels:
            points.append(pixel.depth_corrected_position)
            colors.append(np.array(pixel.color) / 255.0)
        
        if points:
            pcd.points = o3d.utility.Vector3dVector(np.array(points))
            pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return pcd

        

@dataclass
class Pixel:
    raw_timestamp: int
    recording_time_str: str
    u: int
    v: int
    color: tuple[int, int, int]
    raw_depth: float

    def __post_init__(self):
        self.raw_position = self._calculate_xyz(self.u, self.v, self.raw_depth)
    
    def set_background_flag(self, is_bg: bool ,ref_depth) -> None:
        if is_bg:
            self.depth_corrected_position = self._depth_corrected_position(ref_depth)
            self.is_bg = True
        else:
            self.depth_corrected_position = self.raw_position
            self.is_bg = False

    def _depth_corrected_position(self, ref_depth) -> tuple[float, float, float]:
        raw_depth = self.raw_depth
        corrected_depth = 2 * ref_depth - raw_depth
        depth_corrected_position = self._calculate_xyz(self.u, self.v, corrected_depth)    
        return depth_corrected_position
    
    def get_radial_position(self, center):
        x, y, _ = self.depth_corrected_position
        cx, cy = center
        dx = x - cx
        dy = y - cy
        r = np.sqrt(dx**2 + dy**2)
        phi = np.arctan2(dy, dx)
        return r, phi
    
    
    @staticmethod
    def _calculate_xyz(u, v, depth):
        x = (u - PRINCIPAL_POINT_X) * depth / FOCAL_LENGTH_X
        y = (v - PRINCIPAL_POINT_Y) * depth / FOCAL_LENGTH_Y
        z = depth
        return (x, y, z)
    
