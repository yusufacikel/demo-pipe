from pathlib import Path
import re
from datetime import timedelta
from dataclasses import dataclass

INPUT_FILENAME_REGEX = re.compile(r"_(?:Color|Depth)_(\d+\.\d+)\.png")

@dataclass
class ImagePair:
    depth_path: Path
    color_path: Path
    raw_timestamp: int
    recording_time_str: str

class FileManager:
    def __init__(self, run_dir_path):
        self.run_dir_path = Path(run_dir_path)
        self.paired_files = self._load_and_pair_files()

    def _load_and_pair_files(self):
        depth_dir, color_dir = self.run_dir_path / "depth", self.run_dir_path / "color"
        if not depth_dir.exists() or not color_dir.exists():
            return []

        depth_paths = sorted(depth_dir.glob("*.png"), key=lambda p: self._extract_timestamp(p.name))
        color_paths = sorted(color_dir.glob("*.png"), key=lambda p: self._extract_timestamp(p.name))

        if not depth_paths or not color_paths:
            return []

        if len(depth_paths) != len(color_paths):
            exit(1)

        start_ts = self._extract_timestamp(depth_paths[0].name)
        pairs = []

        for i, (d_path, c_path) in enumerate(zip(depth_paths, color_paths)):
            ts = self._extract_timestamp(d_path.name)
            elapsed = timedelta(milliseconds=ts - start_ts)
            recording_time_str = f"{str(elapsed)[:-3] if '.' in str(elapsed) else str(elapsed) + '.000'}"
            pair = ImagePair(
                depth_path=d_path,
                color_path=c_path,
                raw_timestamp=int(ts),
                recording_time_str=recording_time_str
            )
            pairs.append(pair)

        return pairs

    def _extract_timestamp(self, filename):
        match = INPUT_FILENAME_REGEX.search(filename)
        if not match:
            return 0.0
        return float(match.group(1))

    def get_pair(self, index):
        if not (0 <= index < len(self.paired_files)):
            raise IndexError("Index out of range")
        pair = self.paired_files[index]
        return pair
    

from scipy.optimize import least_squares
import numpy as np
import cv2

def fit_circle_to_depth_image(depth):
    depth_norm = cv2.normalize(depth, np.zeros_like(depth, dtype=np.uint8), 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    depth_uint8 = depth_norm.astype(np.uint8)

    _, binary_img = cv2.threshold(depth_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel_1 = np.ones((5, 5), np.uint8)
    kernel_2 = np.ones((13, 13), np.uint8)
    dilated = cv2.dilate(binary_img, kernel_2, iterations=1)
    eroded = cv2.erode(dilated, kernel_1, iterations=3)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    all_points = np.vstack([cnt.squeeze() for cnt in contours if cnt.shape[0] > 4])

    if all_points.shape[0] < 5:
        return None

    X, Y = all_points[:, 0], all_points[:, 1]

    def calc_R(xc, yc):
        return np.sqrt((X - xc)**2 + (Y - yc)**2)

    def cost(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    x0 = [np.mean(X), np.mean(Y)] # type: ignore
    res = least_squares(cost, x0=x0)

    center = tuple(res.x)
    radius = calc_R(*center).mean()

    valid_depth_mask = depth > 0
    mean_depth = np.mean(depth[valid_depth_mask])

    return center, radius, mean_depth


from scipy.optimize import least_squares

def fit_circle_to_pcd(pcd):
    points = np.asarray(pcd.points)

    X, Y = points[:, 0], points[:, 1]

    def calc_R(xc, yc):
        return np.sqrt((X - xc)**2 + (Y - yc)**2)

    def cost(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    x0 = [np.mean(X), np.mean(Y)]

    lower_bounds = [np.min(X) - 1000, np.min(Y) - 1000]
    upper_bounds = [np.max(X) + 1000, np.max(Y) + 1000]

    res = least_squares(cost, x0=x0, bounds=(lower_bounds, upper_bounds))

    center = tuple(res.x)
    radius = calc_R(*center).mean()
    mean_depth = np.mean(points[:, 2])

    return center, radius, mean_depth
