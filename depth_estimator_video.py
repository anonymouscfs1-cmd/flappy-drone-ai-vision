# -*- coding: utf-8 -*-
"""Video tester 2

Drone Depth Estimation - Video Pipeline
Leaf tracking with temporal stability, HSV vegetation locking, and depth-aware confidence gating.

"""

import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import os
import cv2
from torchvision import transforms
import sys
from collections import deque

# --- Configuration ---
MODEL_REPO_ID = "depth-anything/Depth-Anything-V2-Large"
MODEL_FILENAME = "depth_anything_v2_vitl.pth"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)
ENCODER = 'vitl'
MODEL_INPUT_SIZE = 518

# --- FLAPPING DRONE CONFIGURATION ---
BLUR_THRESHOLD = 100.0
ROI_SIZE = 40

# --- LEAF DETECTION CONFIGURATION ---
MIN_LEAF_AREA = 1000
MAX_LEAF_AREA = 50000
LEAF_SELECTION = "largest"

# --- VIDEO PROCESSING MODE ---
VIDEO_MODE = "temporal"  # "framewise" or "temporal"

# --- DEPTH VISUALIZATION DEBUG ---
#  IMPORTANT: Must be defined BEFORE process_video() function
SHOW_DEPTH_DEBUG = True  # Set to True to generate separate depth visualization video
print(f"🔧 CONFIG: SHOW_DEPTH_DEBUG = {SHOW_DEPTH_DEBUG}")  # Sanity check

# Setup repository
if not os.path.exists('Depth-Anything-V2'):
    print("Cloning Depth-Anything-V2 repository...")
    os.system('git clone https://github.com/DepthAnything/Depth-Anything-V2.git')
    print("Repository cloned successfully!")

sys.path.insert(0, os.path.join(os.getcwd(), 'Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2

# ============================================================================
# OBJECT TRAJECTORY TRACKER (volleyball-style single-object tracking)
# V2: HYSTERESIS + DEPTH-AWARE + STRONG COMMITMENT
# ============================================================================
class LeafTracker:
    """
    Single-object tracker with STRONG COMMITMENT to chosen leaf.

    KEY IMPROVEMENTS:
    - Hysteresis: Once locked, switching is HARD
    - Depth consistency: Prefer depth-stable candidates
    - Multi-frame confirmation: New targets need multiple confirmations
    - Stricter loss conditions: Don't give up easily
    """

    def __init__(self):
        # Tracker state
        self.state = "SEARCHING"  # TRACKING / LOST / SEARCHING
        self.tracked_bbox = None  # Current tracked leaf bbox
        self.tracked_center = None  # Center of tracked leaf
        self.confidence = 0.0  # Tracking confidence (0-1)

        # Trajectory history
        self.position_history = deque(maxlen=10)
        self.bbox_history = deque(maxlen=5)
        self.depth_history = deque(maxlen=8)  # NEW: Track depth of target

        # Loss tracking
        self.frames_since_seen = 0
        self.max_lost_frames = 20  # Increased from 15 (more patient)

        # Hysteresis: Make switching HARD
        self.frames_tracked = 0  # How long we've been tracking current target
        self.switch_threshold = 8  # Need this many frames to switch
        self.candidate_streak = {}  # Track how many times each candidate appears

        # Association thresholds (STRICTER for current target)
        self.iou_threshold_tracking = 0.25  # Lower = more forgiving for current target
        self.iou_threshold_switching = 0.5  # Higher = harder to switch
        self.max_center_distance = 100  # Increased from 80 (more forgiving)
        self.max_size_change = 3.0  # Increased from 2.5 (more forgiving)

        # Depth consistency
        self.last_target_depth = None
        self.depth_tolerance = 0.3  # 30% depth change tolerance

    def compute_iou(self, bbox1, bbox2):
        """Compute Intersection over Union between two bboxes."""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def compute_center_distance(self, bbox1, bbox2):
        """Compute distance between bbox centers."""
        cx1 = bbox1[0] + bbox1[2] / 2
        cy1 = bbox1[1] + bbox1[3] / 2
        cx2 = bbox2[0] + bbox2[2] / 2
        cy2 = bbox2[1] + bbox2[3] / 2
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    def compute_size_ratio(self, bbox1, bbox2):
        """Compute size consistency between bboxes."""
        area1 = bbox1[2] * bbox1[3]
        area2 = bbox2[2] * bbox2[3]
        return max(area1, area2) / (min(area1, area2) + 1e-6)

    def compute_depth_consistency(self, candidate_depth):
        """
        Check if candidate depth is consistent with tracked object.
        Returns score 0-1 (1 = very consistent)
        """
        if self.last_target_depth is None or len(self.depth_history) == 0:
            return 0.5  # Neutral

        avg_depth = np.mean(self.depth_history)
        depth_diff = abs(candidate_depth - avg_depth) / (avg_depth + 1e-6)

        if depth_diff < self.depth_tolerance:
            return 1.0  # Very consistent
        else:
            return max(0.0, 1.0 - depth_diff)  # Decreases with difference

    def associate_detection(self, candidate_bbox, candidate_depth=None, is_current_target=True):
        """
        Check if candidate matches tracked object.

        HYSTERESIS: Different thresholds for current target vs new candidates.

        Args:
            candidate_bbox: Bbox to check
            candidate_depth: Depth value of candidate (optional)
            is_current_target: If True, use LENIENT thresholds (favor current)

        Returns:
            (is_match, association_score)
        """
        if self.tracked_bbox is None:
            return False, 0.0

        # Compute metrics
        iou = self.compute_iou(self.tracked_bbox, candidate_bbox)
        distance = self.compute_center_distance(self.tracked_bbox, candidate_bbox)
        size_ratio = self.compute_size_ratio(self.tracked_bbox, candidate_bbox)

        # HYSTERESIS: Use different thresholds
        if is_current_target:
            # LENIENT for current target (easy to keep)
            iou_threshold = self.iou_threshold_tracking
            min_score = 0.35
        else:
            # STRICT for new candidates (hard to switch)
            iou_threshold = self.iou_threshold_switching
            min_score = 0.65

        # Check basic thresholds
        iou_pass = iou > iou_threshold
        distance_pass = distance < self.max_center_distance
        size_pass = size_ratio < self.max_size_change

        # Compute weighted score
        score = 0.0
        if iou_pass:
            score += iou * 0.4  # 40% weight
        if distance_pass:
            score += (1.0 - distance / self.max_center_distance) * 0.3  # 30% weight
        if size_pass:
            score += (1.0 / size_ratio) * 0.15  # 15% weight

        # NEW: Depth consistency bonus
        if candidate_depth is not None:
            depth_score = self.compute_depth_consistency(candidate_depth)
            score += depth_score * 0.15  # 15% weight

        # Match criteria
        passes = sum([iou_pass, distance_pass, size_pass])
        is_match = (passes >= 2) and (score > min_score)

        return is_match, score

    def update(self, detections_with_depth):
        """
        Update tracker with new detections.

        Args:
            detections_with_depth: List of tuples (bbox, depth_value)

        Returns:
            (best_bbox, center_x, center_y, confidence, state)
        """
        detections = [bbox for bbox, _ in detections_with_depth]

        if len(detections) == 0:
            # === NO DETECTIONS ===
            self.frames_since_seen += 1

            if self.state == "TRACKING":
                if self.frames_since_seen < self.max_lost_frames:
                    # Brief occlusion - HOLD position
                    self.state = "LOST"
                    self.confidence = max(0.3, self.confidence - 0.05)  # Slower decay
                    if self.tracked_bbox is not None:
                        x, y, w, h = self.tracked_bbox
                        return self.tracked_bbox, x + w//2, y + h//2, self.confidence, "HOLD"
                else:
                    # Lost for too long
                    self.state = "SEARCHING"
                    self.tracked_bbox = None
                    self.confidence = 0.0
                    self.frames_tracked = 0
                    return None, None, None, 0.0, "SEARCHING"

            elif self.state == "LOST":
                if self.frames_since_seen >= self.max_lost_frames:
                    self.state = "SEARCHING"
                    self.tracked_bbox = None
                    self.confidence = 0.0
                    self.frames_tracked = 0
                    return None, None, None, 0.0, "SEARCHING"
                else:
                    # Still holding
                    if self.tracked_bbox is not None:
                        x, y, w, h = self.tracked_bbox
                        return self.tracked_bbox, x + w//2, y + h//2, self.confidence, "HOLD"

            return None, None, None, 0.0, "SEARCHING"

        # === WE HAVE DETECTIONS ===

        if self.state == "SEARCHING":
            # No active track - initialize
            best_bbox, best_depth = detections_with_depth[0]
            x, y, w, h = best_bbox

            self.tracked_bbox = best_bbox
            self.tracked_center = (x + w//2, y + h//2)
            self.state = "TRACKING"
            self.confidence = 0.7  # Start lower
            self.frames_since_seen = 0
            self.frames_tracked = 1
            self.last_target_depth = best_depth

            self.position_history.append(self.tracked_center)
            self.bbox_history.append(best_bbox)
            self.depth_history.append(best_depth)

            return best_bbox, self.tracked_center[0], self.tracked_center[1], self.confidence, "TRACKING"

        else:  # TRACKING or LOST
            # === TRY TO MATCH CURRENT TARGET ===
            best_match = None
            best_score = 0.0
            best_depth = None

            for bbox, depth in detections_with_depth:
                is_match, score = self.associate_detection(bbox, depth, is_current_target=True)
                if is_match and score > best_score:
                    best_match = bbox
                    best_score = score
                    best_depth = depth

            if best_match is not None:
                # === SUCCESSFUL ASSOCIATION WITH CURRENT TARGET ===
                self.tracked_bbox = best_match
                x, y, w, h = best_match
                self.tracked_center = (x + w//2, y + h//2)
                self.state = "TRACKING"
                self.confidence = min(1.0, 0.5 + best_score * 0.5)  # Build confidence
                self.frames_since_seen = 0
                self.frames_tracked += 1
                self.last_target_depth = best_depth

                self.position_history.append(self.tracked_center)
                self.bbox_history.append(best_match)
                self.depth_history.append(best_depth)

                return best_match, self.tracked_center[0], self.tracked_center[1], self.confidence, "TRACKING"

            else:
                # === NO MATCH - CHECK IF WE SHOULD SWITCH ===

                # HYSTERESIS: Only consider switching if we've lost current target
                self.frames_since_seen += 1

                if self.frames_since_seen < 5:  # Give current target 5 frames grace period
                    # Too early to switch - HOLD current
                    self.state = "LOST"
                    self.confidence = max(0.2, self.confidence - 0.1)
                    if self.tracked_bbox is not None:
                        x, y, w, h = self.tracked_bbox
                        return self.tracked_bbox, x + w//2, y + h//2, self.confidence, "HOLD"

                # Check if any new candidate is CONSISTENTLY better
                for bbox, depth in detections_with_depth:
                    bbox_key = f"{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}"

                    # Track candidate streak
                    if bbox_key not in self.candidate_streak:
                        self.candidate_streak[bbox_key] = 1
                    else:
                        self.candidate_streak[bbox_key] += 1

                    # NEW TARGET needs multiple confirmations
                    if self.candidate_streak[bbox_key] >= self.switch_threshold:
                        # Switch to new target
                        self.tracked_bbox = bbox
                        x, y, w, h = bbox
                        self.tracked_center = (x + w//2, y + h//2)
                        self.state = "TRACKING"
                        self.confidence = 0.6
                        self.frames_since_seen = 0
                        self.frames_tracked = 1
                        self.last_target_depth = depth
                        self.candidate_streak.clear()

                        self.position_history.clear()
                        self.bbox_history.clear()
                        self.depth_history.clear()
                        self.position_history.append(self.tracked_center)
                        self.bbox_history.append(bbox)
                        self.depth_history.append(depth)

                        return bbox, self.tracked_center[0], self.tracked_center[1], self.confidence, "TRACKING"

                # No consistent new candidate - keep holding or search
                if self.frames_since_seen < self.max_lost_frames:
                    self.state = "LOST"
                    if self.tracked_bbox is not None:
                        x, y, w, h = self.tracked_bbox
                        return self.tracked_bbox, x + w//2, y + h//2, self.confidence, "HOLD"
                else:
                    self.state = "SEARCHING"
                    self.tracked_bbox = None
                    self.confidence = 0.0
                    self.frames_tracked = 0
                    self.candidate_streak.clear()
                    return None, None, None, 0.0, "SEARCHING"

        return None, None, None, 0.0, "SEARCHING"

    def reset(self):
        """Reset tracker state."""
        self.state = "SEARCHING"
        self.tracked_bbox = None
        self.tracked_center = None
        self.confidence = 0.0
        self.position_history.clear()
        self.bbox_history.clear()
        self.depth_history.clear()
        self.frames_since_seen = 0
        self.frames_tracked = 0
        self.candidate_streak.clear()
        self.last_target_depth = None


class DepthReferenceEstimator:
    """Depth estimator with video support."""

    def __init__(self, model_path=MODEL_PATH, encoder=ENCODER):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        if not os.path.exists(model_path):
            print(f"Downloading {MODEL_FILENAME}...")
            try:
                hf_hub_download(repo_id=MODEL_REPO_ID, filename=MODEL_FILENAME,
                              local_dir=".", local_dir_use_symlinks=False)
            except Exception as e:
                print(f"Error downloading model: {e}")
                raise

        self.model = DepthAnythingV2(encoder=encoder).to(self.device).eval()
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        print("Depth-Anything-V2 model loaded successfully.\n")

        self.transform = transforms.Compose([
            transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.depth_map = None
        self.depth_polarity = None
        self.image_width = None
        self.image_height = None

        # Temporal tracking state
        self.prev_frame_gray = None
        self.prev_target_point = None
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.distance_history = deque(maxlen=8)

        # Stability tracking
        self.last_stable_point = None
        self.last_stable_depth = None
        self.blur_history = deque(maxlen=3)

        # Leaf tracking state (OLD - will be replaced by tracker)
        self.last_leaf_bbox = None
        self.leaf_lost_frames = 0

        # === NEW: Object trajectory tracker ===
        self.leaf_tracker = LeafTracker()

    def estimate_depth(self, image_input):
        """Generate depth map from image."""
        if isinstance(image_input, str):
            raw_image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            raw_image = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        else:
            raw_image = image_input

        h, w = raw_image.height, raw_image.width
        self.image_height = h
        self.image_width = w

        image = self.transform(raw_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth_map = self.model(image)

        depth_map = torch.nn.functional.interpolate(
            depth_map[None], (h, w), mode='bilinear', align_corners=False
        )[0, 0]

        self.depth_map = depth_map.cpu().numpy()
        self._detect_depth_polarity()

        return self.depth_map

    def _detect_depth_polarity(self):
        """Detect if higher depth values mean closer or farther."""
        h, w = self.depth_map.shape

        center_y, center_x = h // 2, w // 2
        center_region = self.depth_map[
            center_y - h//8 : center_y + h//8,
            center_x - w//8 : center_x + w//8
        ]
        center_median = np.median(center_region)

        corner_size = min(h, w) // 10
        corners = [
            self.depth_map[:corner_size, :corner_size],
            self.depth_map[:corner_size, -corner_size:],
            self.depth_map[-corner_size:, :corner_size],
            self.depth_map[-corner_size:, -corner_size:]
        ]
        corners_median = np.median([np.median(c) for c in corners])

        if center_median > corners_median:
            self.depth_polarity = 'higher_is_closer'
        else:
            self.depth_polarity = 'lower_is_closer'

    def detect_motion_blur(self, frame_gray):
        """Detect motion blur using Laplacian variance."""
        laplacian = cv2.Laplacian(frame_gray, cv2.CV_64F)
        blur_score = laplacian.var()
        self.blur_history.append(blur_score)
        is_sharp = blur_score > BLUR_THRESHOLD
        return is_sharp, blur_score

    def detect_leaf_region(self, frame_bgr):
        """
        Detect stable LEAF region with quality + temporal filtering.
        IMPROVEMENTS:
        - Texture scoring (penalize flat/blurry leaves)
        - Temporal stability (prefer previously tracked leaf)
        - Multi-criteria scoring (not just size)
        """
        h, w = frame_bgr.shape[:2]
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # HSV green detection
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        mask2 = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))
        plant_mask = cv2.bitwise_or(mask1, mask2)

        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)

        # Find contours
        contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None, w//2, h//2, False

        # Filter by area
        valid_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_LEAF_AREA < area < MAX_LEAF_AREA:
                valid_contours.append(cnt)

        if len(valid_contours) == 0:
            return None, w//2, h//2, False

        # === QUALITY SCORING ===
        best_score = -1
        best_contour = None

        for cnt in valid_contours:
            x, y, w_box, h_box = cv2.boundingRect(cnt)
            cx, cy = x + w_box // 2, y + h_box // 2

            # Extract leaf region
            leaf_roi = frame_gray[y:y+h_box, x:x+w_box]
            if leaf_roi.size == 0:
                continue

            # === SCORING CRITERIA ===
            score = 0

            # 1. Size (normalized, 0-100)
            area = cv2.contourArea(cnt)
            size_score = min(100, (area / MAX_LEAF_AREA) * 100)
            score += size_score * 0.3  # 30% weight

            # 2. Texture quality (Laplacian variance)
            laplacian = cv2.Laplacian(leaf_roi, cv2.CV_64F)
            texture_score = min(100, laplacian.var() / 10)  # normalize
            score += texture_score * 0.25  # 25% weight

            # 3. Compactness (circular = better)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter > 0:
                compactness = 4 * np.pi * area / (perimeter ** 2)
                score += compactness * 40  # 40 points max

            # 4. Temporal stability (huge bonus for staying with same leaf)
            if self.last_leaf_bbox is not None:
                last_x, last_y, last_w, last_h = self.last_leaf_bbox
                last_cx = last_x + last_w // 2
                last_cy = last_y + last_h // 2

                # Distance to previous leaf center
                dist = np.sqrt((cx - last_cx)**2 + (cy - last_cy)**2)

                # Overlap check (IoU-like)
                x1, y1 = max(x, last_x), max(y, last_y)
                x2, y2 = min(x+w_box, last_x+last_w), min(y+h_box, last_y+last_h)
                overlap_area = max(0, x2-x1) * max(0, y2-y1)
                overlap_ratio = overlap_area / (w_box * h_box)

                # Strong bonus for temporal continuity
                if dist < 50:  # Close to previous leaf
                    score += 100  # Huge bonus
                if overlap_ratio > 0.5:  # Overlaps with previous
                    score += 150  # Even bigger bonus

            if score > best_score:
                best_score = score
                best_contour = cnt

        if best_contour is None:
            return None, w//2, h//2, False

        # Compute final bbox
        x, y, w_box, h_box = cv2.boundingRect(best_contour)
        bbox = (x, y, w_box, h_box)
        center_x = x + w_box // 2
        center_y = y + h_box // 2

        return bbox, center_x, center_y, True

    def get_roi_depth(self, depth_map, center_x, center_y, roi_size=ROI_SIZE):
        """
        Get depth with OUTLIER REJECTION.
        IMPROVEMENTS:
        - Uses IQR filtering (removes outliers)
        - Fallback to median if too few valid points
        - Safety checks for empty regions
        """
        h, w = depth_map.shape
        half_size = roi_size // 2

        y1 = max(0, center_y - half_size)
        y2 = min(h, center_y + half_size)
        x1 = max(0, center_x - half_size)
        x2 = min(w, center_x + half_size)

        roi = depth_map[y1:y2, x1:x2]

        if roi.size == 0:
            return depth_map[center_y, center_x]  # fallback to single pixel

        # === OUTLIER REJECTION (IQR method) ===
        flat = roi.flatten()
        q1, q3 = np.percentile(flat, [25, 75])
        iqr = q3 - q1

        # Filter values within 1.5*IQR range
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        filtered = flat[(flat >= lower_bound) & (flat <= upper_bound)]

        # Use filtered median if we have enough points
        if len(filtered) > roi.size * 0.3:  # At least 30% valid
            return np.median(filtered)
        else:
            return np.median(flat)  # fallback to full ROI median

    def get_target_point_framewise(self, frame_bgr):
        """MODE 1: Get target point independently for each frame."""
        bbox, center_x, center_y, is_valid = self.detect_leaf_region(frame_bgr)

        if is_valid:
            return center_x, center_y, bbox, True
        else:
            h, w = frame_bgr.shape[:2]
            return w // 2, h // 2, None, False

    def get_target_point_temporal(self, frame_bgr, frame_gray, is_sharp, depth_map):
        """
        MODE 2: Track leaf using HYSTERESIS-BASED trajectory tracking.

        KEY CHANGES (V2):
        - Passes depth information to tracker
        - Stronger commitment to chosen leaf
        - Multi-frame confirmation before switching
        - Depth-aware association

        Args:
            frame_bgr: Current frame (color)
            frame_gray: Current frame (grayscale)
            is_sharp: Whether frame is sharp enough
            depth_map: Depth map for current frame
        """
        h, w = frame_bgr.shape[:2]

        # Skip detection on very blurry frames
        if not is_sharp:
            if self.leaf_tracker.state == "TRACKING" or self.leaf_tracker.state == "LOST":
                # Simulate "no detections" update
                bbox, cx, cy, confidence, state = self.leaf_tracker.update([])

                if bbox is not None:
                    is_stable = (state == "TRACKING" and confidence > 0.75)  # Stricter
                    return cx, cy, bbox, is_stable
                else:
                    return w // 2, h // 2, None, False
            else:
                return w // 2, h // 2, None, False

        # === DETECT ALL CANDIDATE LEAVES ===
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        mask2 = cv2.inRange(hsv, (25, 30, 30), (95, 255, 255))
        plant_mask = cv2.bitwise_or(mask1, mask2)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_CLOSE, kernel)
        plant_mask = cv2.morphologyEx(plant_mask, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(plant_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter and score candidates WITH DEPTH
        frame_gray_local = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        candidate_detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if MIN_LEAF_AREA < area < MAX_LEAF_AREA:
                x, y, w_box, h_box = cv2.boundingRect(cnt)
                bbox = (x, y, w_box, h_box)
                cx, cy = x + w_box // 2, y + h_box // 2

                # Get depth at center
                try:
                    candidate_depth = self.get_roi_depth(depth_map, cx, cy, ROI_SIZE // 2)
                except:
                    candidate_depth = depth_map[cy, cx] if 0 <= cy < depth_map.shape[0] and 0 <= cx < depth_map.shape[1] else 0

                # Quality score
                leaf_roi = frame_gray_local[y:y+h_box, x:x+w_box]
                if leaf_roi.size > 0:
                    laplacian = cv2.Laplacian(leaf_roi, cv2.CV_64F)
                    texture_score = laplacian.var()
                    quality = (area / MAX_LEAF_AREA) * 0.5 + min(1.0, texture_score / 1000) * 0.5
                    candidate_detections.append(((bbox, candidate_depth), quality))

        # Sort by quality
        candidate_detections.sort(key=lambda x: x[1], reverse=True)
        candidate_with_depth = [data for data, _ in candidate_detections]

        # === UPDATE TRACKER (now with depth) ===
        bbox, cx, cy, confidence, state = self.leaf_tracker.update(candidate_with_depth)

        if bbox is None:
            return w // 2, h // 2, None, False

        # === STRICTER STABILITY FOR DEPTH UPDATE ===
        # Only update depth when VERY confident
        is_stable = (state == "TRACKING" and confidence > 0.75 and is_sharp)

        return cx, cy, bbox, is_stable

    def smooth_distance_temporal(self, current_distance, is_stable):
        """
        Exponential Moving Average (EMA) with outlier rejection.
        IMPROVEMENTS:
        - EMA instead of simple moving average (smoother)
        - Outlier detection (reject sudden jumps)
        - Better initialization
        """
        alpha = 0.3  # EMA smoothing factor (lower = smoother, higher = responsive)

        if is_stable:
            # Outlier detection: reject values that jump > 50% from recent history
            if len(self.distance_history) > 0:
                recent_avg = np.mean(self.distance_history)
                percent_change = abs(current_distance - recent_avg) / (recent_avg + 1e-6)

                if percent_change > 0.5:  # More than 50% jump = outlier
                    # Don't add to history, keep last stable
                    return self.last_stable_depth if self.last_stable_depth else current_distance

            # Update history and EMA
            self.distance_history.append(current_distance)
            self.last_stable_depth = current_distance

            # Compute EMA
            if len(self.distance_history) == 1:
                return current_distance  # First value
            else:
                prev_ema = np.mean(list(self.distance_history)[:-1])
                ema = alpha * current_distance + (1 - alpha) * prev_ema
                return ema
        else:
            # Unstable frame: hold last value
            return self.last_stable_depth if self.last_stable_depth else current_distance

    def reset_temporal_state(self):
        """Reset temporal tracking state."""
        self.prev_frame_gray = None
        self.prev_target_point = None
        self.distance_history.clear()
        self.last_stable_point = None
        self.last_stable_depth = None
        self.blur_history.clear()
        self.last_leaf_bbox = None
        self.leaf_lost_frames = 0

        # === NEW: Reset tracker ===
        self.leaf_tracker.reset()


def calculate_distance_from_depth(depth_value, depth_map, reference_depth, reference_distance_cm):
    """
    Calculate RELATIVE depth with safety checks.
    IMPROVEMENTS:
    - Multiple fallbacks for edge cases
    - Clamping to prevent extreme values
    - Better divide-by-zero handling
    """
    epsilon = 1e-6

    # Safety check 1: Reference depth
    if abs(reference_depth) < epsilon:
        reference_depth = np.median(depth_map) + epsilon

    # Safety check 2: Current depth
    if abs(depth_value) < epsilon:
        depth_value = reference_depth

    # Calculate ratio
    ratio = depth_value / (reference_depth + epsilon)

    # Safety check 3: Prevent extreme ratios
    ratio = np.clip(ratio, 0.1, 10.0)  # Limit to 10x range

    # Calculate distance
    estimated_distance = reference_distance_cm / ratio

    # Safety check 4: Clamp output to reasonable range
    estimated_distance = np.clip(estimated_distance, 1.0, 500.0)  # 1cm to 5m

    return estimated_distance


# ============================================================================
# DEPTH VISUALIZATION (INDEPENDENT DEBUG OUTPUT)
# ============================================================================
def colorize_depth_map(depth_map):
    """
    Convert depth map to colored visualization.

    Classic Depth-Anything style:
    - Purple/Blue = Far
    - Yellow/Red = Close

    IMPORTANT: This is PURELY for visualization, completely independent
    of tracking, scope, or any temporal logic.

    Args:
        depth_map: Raw depth map (numpy array)

    Returns:
        BGR image (numpy array) suitable for video output
    """
    # Normalize depth to 0-255 range
    depth_min = depth_map.min()
    depth_max = depth_map.max()

    if depth_max - depth_min < 1e-6:
        # Uniform depth (shouldn't happen, but safety)
        depth_norm = np.zeros_like(depth_map, dtype=np.uint8)
    else:
        depth_norm = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)

    # Apply colormap: COLORMAP_MAGMA gives purple→yellow
    # (purple = low values = far, yellow = high values = close for most models)
    depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_MAGMA)

    return depth_colored


# ============================================================================
# CRITICAL: THIS IS THE ONLY draw_scope_overlay FUNCTION
# There should be NO other definitions of this function anywhere in your code
# ============================================================================
def draw_scope_overlay(frame, target_x, target_y, distance_cm, mode_label,
                       is_stable=True, roi_size=40, leaf_bbox=None):
    """
    BIGGER, CLEARER tactical HUD scope.
    IMPROVEMENTS:
    - 1.8x larger scope elements
    - Text INSIDE scope circle
    - Better contrast for greenhouse lighting
    - Clear stability indicator
    """
    output = frame.copy()
    h, w = output.shape[:2]

    # === LEAF BOUNDING BOX ===
    if leaf_bbox is not None:
        bx, by, bw, bh = leaf_bbox
        cv2.rectangle(output, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
        # Label with background for readability
        label_y = max(20, by - 10)
        cv2.rectangle(output, (bx, label_y - 20), (bx + 80, label_y + 5), (0, 200, 0), -1)
        cv2.putText(output, "LEAF", (bx + 5, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # === SCOPE COLORS ===
    if is_stable:
        main_color = (0, 255, 255)  # Cyan (more visible than yellow)
        bg_color = (0, 128, 128)    # Dark cyan
    else:
        main_color = (0, 100, 255)  # Orange (more visible than red)
        bg_color = (0, 50, 128)     # Dark orange

    # === SCOPE CIRCLES (1.8x larger) ===
    radii = [60, 100, 140]  # Was [40, 70, 100]
    for r in radii:
        # Outer glow
        cv2.circle(output, (target_x, target_y), r, bg_color, 6)
        # Main circle
        cv2.circle(output, (target_x, target_y), r, main_color, 3)

    # === CENTER CROSSHAIRS (longer) ===
    cross_len = 35  # Was 20
    thickness = 3
    # Horizontal
    cv2.line(output, (target_x - cross_len, target_y),
             (target_x + cross_len, target_y), bg_color, thickness + 2)
    cv2.line(output, (target_x - cross_len, target_y),
             (target_x + cross_len, target_y), main_color, thickness)
    # Vertical
    cv2.line(output, (target_x, target_y - cross_len),
             (target_x, target_y + cross_len), bg_color, thickness + 2)
    cv2.line(output, (target_x, target_y - cross_len),
             (target_x, target_y + cross_len), main_color, thickness)

    # === CENTER DOT ===
    cv2.circle(output, (target_x, target_y), 8, bg_color, -1)
    cv2.circle(output, (target_x, target_y), 6, main_color, -1)

    # === ROI BOX (measurement region) ===
    half = roi_size // 2
    cv2.rectangle(output,
                 (target_x - half, target_y - half),
                 (target_x + half, target_y + half),
                 bg_color, 4)
    cv2.rectangle(output,
                 (target_x - half, target_y - half),
                 (target_x + half, target_y + half),
                 main_color, 2)

    # === TEXT INSIDE SCOPE (CRITICAL IMPROVEMENT) ===
    font = cv2.FONT_HERSHEY_DUPLEX  # More readable font

    # Distance value (larger, centered below crosshair)
    dist_text = f"{distance_cm:.0f}"
    unit_text = "rel."

    # Main number
    (tw, th), _ = cv2.getTextSize(dist_text, font, 1.8, 3)
    text_x = target_x - tw // 2
    text_y = target_y + 55  # Below center

    # Background rectangle for contrast
    pad = 12
    cv2.rectangle(output,
                 (text_x - pad, text_y - th - pad),
                 (text_x + tw + pad, text_y + pad),
                 (0, 0, 0), -1)  # Black background
    cv2.rectangle(output,
                 (text_x - pad, text_y - th - pad),
                 (text_x + tw + pad, text_y + pad),
                 main_color, 2)  # Colored border

    # Draw number
    cv2.putText(output, dist_text, (text_x, text_y),
               font, 1.8, main_color, 3, cv2.LINE_AA)

    # Unit text (smaller, below)
    (uw, uh), _ = cv2.getTextSize(unit_text, font, 0.7, 2)
    unit_x = target_x - uw // 2
    unit_y = text_y + 25
    cv2.putText(output, unit_text, (unit_x, unit_y),
               font, 0.7, main_color, 2, cv2.LINE_AA)

    # === STABILITY INDICATOR (top of scope) ===
    # Map states to display
    if is_stable:
        status_text = "TRACKING"
        status_color = (0, 255, 0)  # Green
    else:
        status_text = "HOLD"
        status_color = (100, 150, 255)  # Orange

    (sw, sh), _ = cv2.getTextSize(status_text, font, 0.6, 2)
    status_x = target_x - sw // 2
    status_y = target_y - 110  # Above scope

    # Background
    cv2.rectangle(output,
                 (status_x - 8, status_y - sh - 5),
                 (status_x + sw + 8, status_y + 5),
                 (0, 0, 0), -1)
    cv2.rectangle(output,
                 (status_x - 8, status_y - sh - 5),
                 (status_x + sw + 8, status_y + 5),
                 status_color, 2)

    cv2.putText(output, status_text, (status_x, status_y),
               font, 0.6, status_color, 2, cv2.LINE_AA)

    # === MODE LABEL (top-left corner) ===
    mode_display = f"MODE: {mode_label.upper()}"
    cv2.rectangle(output, (10, 10), (250, 50), (0, 0, 0), -1)
    cv2.rectangle(output, (10, 10), (250, 50), (255, 255, 255), 2)
    cv2.putText(output, mode_display, (20, 38),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return output


def process_video(video_path, estimator, reference_distance_cm=35.0, mode="framewise"):
    """Process video with depth estimation."""
    print(f"\n{'='*70}")
    print(f"VIDEO PROCESSING - MODE: {mode.upper()}")
    print(f"{'='*70}\n")

    estimator.reset_temporal_state()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Input: {video_path}")
    print(f"Resolution: {width}x{height} @ {fps} FPS")
    print(f"Total frames: {total_frames}\n")

    # === MAIN OUTPUT VIDEO ===
    output_path = video_path.replace(".mp4", f"_{mode}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # === DEPTH DEBUG VIDEO (NEW) ===
    out_depth = None
    depth_output_path = None
    if SHOW_DEPTH_DEBUG:
        depth_output_path = video_path.replace(".mp4", f"_{mode}_depth_debug.mp4")
        out_depth = cv2.VideoWriter(depth_output_path, fourcc, fps, (width, height))
        print(f" Depth debug video ENABLED")
        print(f"   Output: {depth_output_path}\n")

    # Reference calibration
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    _ = estimator.estimate_depth(first_frame)
    ref_x, ref_y = width // 2, height // 2
    reference_depth = estimator.depth_map[ref_y, ref_x]

    print(f"Calibration reference:")
    print(f"  Point: ({ref_x}, {ref_y})")
    print(f"  Depth value: {reference_depth:.4f}")
    print(f"  Reference scale: {reference_distance_cm:.0f} (relative units)\n")

    mode_label = "frame-wise" if mode == "framewise" else "temporal"

    blur_frames = 0
    unstable_frames = 0
    no_leaf_frames = 0

    frame_idx = 0
    max_frames = total_frames
    print("Processing frames:")

    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        is_sharp, blur_score = estimator.detect_motion_blur(frame_gray)
        if not is_sharp:
            blur_frames += 1

        depth_map = estimator.estimate_depth(frame)

        # ====================================================================
        # Get target point with leaf bounding box
        # ====================================================================
        is_stable = True
        leaf_bbox = None

        if mode == "framewise":
            target_x, target_y, leaf_bbox, is_valid = estimator.get_target_point_framewise(frame)
            if not is_valid:
                no_leaf_frames += 1
        else:  # temporal
            target_x, target_y, leaf_bbox, is_stable = estimator.get_target_point_temporal(
                frame, frame_gray, is_sharp, depth_map
            )
            if not is_stable:
                unstable_frames += 1
            if leaf_bbox is None:
                no_leaf_frames += 1

        # Get depth from REGION
        target_depth = estimator.get_roi_depth(depth_map, target_x, target_y, ROI_SIZE)

        # Calculate distance
        distance_cm = calculate_distance_from_depth(
            target_depth, depth_map, reference_depth, reference_distance_cm
        )

        # Apply smoothing in temporal mode
        if mode == "temporal":
            distance_cm = estimator.smooth_distance_temporal(distance_cm, is_stable)

        # ====================================================================
        # MAIN OUTPUT: Scope overlay video
        # ====================================================================
        result_frame = draw_scope_overlay(
            frame, target_x, target_y, distance_cm, mode_label,
            is_stable=(is_stable if mode == "temporal" else True),
            roi_size=ROI_SIZE,
            leaf_bbox=leaf_bbox
        )

        out.write(result_frame)

        # ====================================================================
        # DEPTH DEBUG OUTPUT: Colorized depth map (NEW)
        # ====================================================================
        if SHOW_DEPTH_DEBUG and out_depth is not None:
            # Generate colorized depth visualization
            depth_vis = colorize_depth_map(depth_map)

            # CRITICAL: Ensure correct size and type
            if depth_vis.shape[:2] != (height, width):
                depth_vis = cv2.resize(depth_vis, (width, height))

            # Ensure uint8 3-channel BGR
            if depth_vis.dtype != np.uint8:
                depth_vis = depth_vis.astype(np.uint8)

            if len(depth_vis.shape) == 2:
                depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2BGR)

            # Write depth frame
            out_depth.write(depth_vis)

        frame_idx += 1
        if frame_idx % 10 == 0 or frame_idx == 1:
            percent = (frame_idx / max_frames) * 100
            elapsed_bar = '█' * int(percent / 5) + '░' * (20 - int(percent / 5))
            print(f"  [{elapsed_bar}] {frame_idx}/{max_frames} ({percent:.1f}%)")
            sys.stdout.flush()

    cap.release()
    out.release()

    # Release depth video writer
    if out_depth is not None:
        out_depth.release()

    print(f"\n✓ Output saved: {output_path}")
    print(f"  Mode: {mode.upper()}")
    print(f"  Frames: {frame_idx}")

    # Depth debug confirmation
    if SHOW_DEPTH_DEBUG and depth_output_path:
        print(f"\n✓ Depth debug saved: {depth_output_path}")
        print(f"  Visualization: Purple/Blue=Far, Yellow/Red=Close")

    print(f"\n📊 LEAF DETECTION STATISTICS:")
    print(f"  • Frames with valid leaf: {frame_idx - no_leaf_frames} ({100*(frame_idx-no_leaf_frames)/frame_idx:.1f}%)")
    print(f"  • Frames without leaf: {no_leaf_frames} ({100*no_leaf_frames/frame_idx:.1f}%)")
    if mode == "temporal":
        print(f"\n📊 OBJECT TRACKING STATISTICS:")
        print(f"  • Tracking approach: Trajectory-based (volleyball-style)")
        print(f"  • Association: IoU + center distance + size consistency + depth")
        print(f"  • Occlusion handling: Hold position up to 20 frames")
        print(f"  • Depth updates: Only on high-confidence tracking")
    print(f"\n📊 STABILITY STATISTICS:")
    print(f"  • Blurry frames: {blur_frames} ({100*blur_frames/frame_idx:.1f}%)")
    if mode == "temporal":
        print(f"  • Unstable frames: {unstable_frames} ({100*unstable_frames/frame_idx:.1f}%)")
        print(f"  • (Depth held on unstable frames)")
    print(f"{'='*70}\n")

# --- Main Execution ---
if __name__ == "__main__":

    estimator = DepthReferenceEstimator()

    VIDEO_FILE = None
    try:
        from google.colab import files
        import shutil
        print("=" * 70)
        print("📹 UPLOAD VIDEO")
        print("=" * 70)
        uploaded = files.upload()
        original_name = list(uploaded.keys())[0]
        VIDEO_FILE = "input_video.mp4"
        shutil.copy(original_name, VIDEO_FILE)
        print(f"✓ Uploaded: {original_name}\n")
    except:
        import glob
        videos = glob.glob("*.mp4") + glob.glob("*.avi") + glob.glob("*.mov")
        videos = [f for f in videos if not any(x in f for x in ['framewise', 'temporal', 'output'])]
        if videos:
            VIDEO_FILE = videos[0]
            print(f"✓ Using: {VIDEO_FILE}\n")
        else:
            print("Error: No video found.")
            exit()

    REFERENCE_DISTANCE_CM = 35.0

    print(f"Processing mode: {VIDEO_MODE}")
    print(f"Reference scale: {REFERENCE_DISTANCE_CM:.0f} (relative units)\n")

    process_video(VIDEO_FILE, estimator, REFERENCE_DISTANCE_CM, mode=VIDEO_MODE)

    print("=" * 70)
    print("✓ COMPLETE")
    print("=" * 70)
