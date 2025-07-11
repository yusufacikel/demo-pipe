from frame import Frame
from dataclasses import dataclass
import cv2
import numpy as np
from typing import Optional



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

class World:
    def __init__(self):
        self.feature_registry = self.FeatureRegistry()
        self.frames = []
    
    def add_frame(self, frame: 'Frame'):
        if self.frames:
            previous_frame = self.frames[-1]
            prev_pose = previous_frame.pose
            self.frames.append(frame)
            self.feature_registry.register_extracted_features(frame, prev_pose)
        else:
            self.frames.append(frame)
            self.feature_registry.register_extracted_features(frame)

    class FeatureRegistry:
        def __init__(self):
            self.active_features = []
            self.discarded_features = []
            self.state_history = []

        def register_extracted_features(self, frame: 'Frame', prev_pose: Optional[np.ndarray] = None):
            frame_idx = frame.frame_idx
            extracted_features = frame.extracted_features


            previous_active_features = self.active_features.copy()

            matched_features = []
            matched_count = 0
            new_count = 0

            for keypoint, descriptor, xyz in extracted_features:
                matched = False
                for tracked_feature in previous_active_features:
                    _, last_match_kp, last_match_desc, _ = tracked_feature.match_list[-1]

                    if self._compare_descriptors(descriptor, last_match_desc):
                        dx = last_match_kp.pt[0] - keypoint.pt[0]
                        dy = last_match_kp.pt[1] - keypoint.pt[1]
                        if dx * dx + dy * dy < 100:
                            tracked_feature.add_feature(frame_idx, keypoint, descriptor, xyz)
                            matched_features.append(tracked_feature)
                            matched = True
                            matched_count += 1
                            break

                if not matched:
                    unique_id = f"{frame_idx}_{hash(descriptor.tobytes())}_{keypoint.pt[0]:.1f}_{keypoint.pt[1]:.1f}"
                    new_feature = self.TrackedFeature(unique_id, frame_idx)
                    new_feature.add_feature(frame_idx, keypoint, descriptor, xyz)
                    self.active_features.append(new_feature)
                    new_count += 1


            self._age_features()
            self.state_history.append(self.active_features.copy())


            if frame_idx == 0:
                init_pose = np.eye(4, dtype=np.float32)
                frame.set_pose(init_pose)
            else:
                T = self._estimate_pose_change(matched_features)
                pose = prev_pose @ np.linalg.inv(T) # type: ignore
                frame.set_pose(pose)
                        
        def _estimate_pose_change(self, matched_features):
            object_points = []
            image_points = []

            for feature in matched_features:
                frame_idx_prev, _, _, xyz_prev = feature.match_list[-2]
                frame_idx_curr, curr_kp, _, _ = feature.match_list[-1]
                if frame_idx_prev != frame_idx_curr:
                    if not np.any(np.isnan(xyz_prev)) and not np.allclose(xyz_prev, [0,0,0]):
                        object_points.append(xyz_prev)
                        image_points.append((curr_kp.pt[0], curr_kp.pt[1]))
        

            object_points = np.array(object_points, dtype=np.float32)
            image_points = np.array(image_points, dtype=np.float32)

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                np.zeros((4, 1), dtype=np.float32),
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                return None

            R, _ = cv2.Rodrigues(rvec)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = tvec.flatten()

            return T              

        def _compare_descriptors(self, desc1, desc2):
            return cv2.norm(desc1, desc2, cv2.NORM_HAMMING) < 30
        
        def _age_features(self):
            for feature in self.active_features:
                feature.age += 1
                if feature.age > 10:
                    self.discarded_features.append(feature)
                    self.active_features.remove(feature)
     
        @dataclass
        class TrackedFeature:
            unique_id: str
            first_seen_frame_idx: int
            
            def __post_init__(self):
                self.age = 0
                self.match_list = []

            def add_feature(self, frame_idx, keypoint, descriptor, xyz):
                self.match_list.append((frame_idx, keypoint, descriptor, xyz))
