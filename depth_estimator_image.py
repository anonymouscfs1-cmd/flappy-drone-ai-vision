# -*- coding: utf-8 -*-
"""
Drone Depth Estimation - Image Pipeline
Single image depth estimation with scope overlay and manual distance calibration.
"""

import torch
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download
import os
import cv2
from torchvision import transforms
import sys

# --- Configuration ---
MODEL_REPO_ID = "depth-anything/Depth-Anything-V2-Large"
MODEL_FILENAME = "depth_anything_v2_vitl.pth"
MODEL_PATH = os.path.join(os.getcwd(), MODEL_FILENAME)
ENCODER = 'vitl'
MODEL_INPUT_SIZE = 518

# Setup repository
if not os.path.exists('Depth-Anything-V2'):
    print("Cloning Depth-Anything-V2 repository...")
    os.system('git clone https://github.com/DepthAnything/Depth-Anything-V2.git')
    print("Repository cloned successfully!")

sys.path.insert(0, os.path.join(os.getcwd(), 'Depth-Anything-V2'))
from depth_anything_v2.dpt import DepthAnythingV2

class DepthReferenceEstimator:
    """
    Depth estimator that provides a reference point on the plant.
    Distance is based on visual estimation, not automated measurement.
    """

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

    def estimate_depth(self, image_path):
        """Generate raw depth map and detect polarity."""
        print(f"Processing: {image_path}")

        raw_image = Image.open(image_path).convert("RGB")
        h, w = raw_image.height, raw_image.width

        # Store original image dimensions
        self.image_height = h
        self.image_width = w

        image = self.transform(raw_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth_map = self.model(image)

        depth_map = torch.nn.functional.interpolate(
            depth_map[None], (h, w), mode='bilinear', align_corners=False
        )[0, 0]

        self.depth_map = depth_map.cpu().numpy()

        # Detect depth polarity
        self._detect_depth_polarity()

        # Create visualization
        vis_path = self._create_visualization(image_path)

        return self.depth_map, vis_path

    def _detect_depth_polarity(self):
        """
        Detect if higher depth values mean closer or farther.
        Uses image center vs corners.
        """
        h, w = self.depth_map.shape

        # Sample center region (likely foreground)
        center_y, center_x = h // 2, w // 2
        center_region = self.depth_map[
            center_y - h//8 : center_y + h//8,
            center_x - w//8 : center_x + w//8
        ]
        center_median = np.median(center_region)

        # Sample corners (likely background)
        corner_size = min(h, w) // 10
        corners = [
            self.depth_map[:corner_size, :corner_size],
            self.depth_map[:corner_size, -corner_size:],
            self.depth_map[-corner_size:, :corner_size],
            self.depth_map[-corner_size:, -corner_size:]
        ]
        corners_median = np.median([np.median(c) for c in corners])

        # Determine polarity
        if center_median > corners_median:
            self.depth_polarity = 'higher_is_closer'
            polarity_str = "Higher values = CLOSER"
        else:
            self.depth_polarity = 'lower_is_closer'
            polarity_str = "Lower values = CLOSER"

        print(f"\n Depth Polarity: {polarity_str}\n")

    def _create_visualization(self, image_path):
        """Create colormap visualization."""
        min_d, max_d = self.depth_map.min(), self.depth_map.max()
        normalized = (self.depth_map - min_d) / (max_d - min_d)

        # Invert colormap if depth polarity is inverted
        if self.depth_polarity == 'lower_is_closer':
            normalized = 1.0 - normalized

        vis = (normalized * 255).astype(np.uint8)
        colored = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)

        output_path = image_path.replace(".jpg", "_depth.png").replace(".png", "_depth.png")
        cv2.imwrite(output_path, colored)
        print(f"✓ Depth visualization: {output_path}")
        print(f"  (Brighter = Closer, Darker = Farther)\n")

        return output_path

    def get_center_reference_point(self):
        """
        Get image center as reference point.
        This is where the user visually estimates distance.
        """
        if self.image_width is None or self.image_height is None:
            print("Error: Generate depth map first.")
            return None, None

        x_center = self.image_width // 2
        y_center = self.image_height // 2

        return x_center, y_center

def calculate_distance_range(visual_estimate_cm, uncertainty_pct=15):
    """
    Calculate distance range based on visual estimate.

    Args:
        visual_estimate_cm: Human visual estimate (in cm)
        uncertainty_pct: Uncertainty percentage (default ±15%)

    Returns:
        (d_min, d_center, d_max)
    """
    d_center = visual_estimate_cm
    uncertainty_factor = uncertainty_pct / 100.0
    d_min = d_center * (1 - uncertainty_factor)
    d_max = d_center * (1 + uncertainty_factor)

    return d_min, d_center, d_max

def draw_scope_overlay(image_path, target_x, target_y,
                       dist_min_cm, dist_center_cm, dist_max_cm,
                       output_suffix="_result.jpg"):
    """Draw tactical scope with visual distance estimate."""

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Cannot read {image_path}")
        return None

    # Format display text
    distance_str = f"{dist_min_cm:.0f}–{dist_max_cm:.0f} cm"
    subtitle_str = "visual estimate"
    text_color = (255, 255, 0)  # Cyan

    # --- SCOPE RETICLE ---
    outline_color = (0, 0, 0)
    outline_thick = 8
    main_color = (255, 255, 0)  # Cyan
    main_thick = 4

    # Concentric circles
    for r in [40, 70]:
        cv2.circle(img, (target_x, target_y), r, outline_color, outline_thick)
        cv2.circle(img, (target_x, target_y), r, main_color, main_thick)

    # Center dot
    cv2.circle(img, (target_x, target_y), 6, main_color, -1)

    # Crosshairs
    crosshair_coords = [
        ((target_x - 100, target_y), (target_x - 40, target_y)),
        ((target_x + 40, target_y), (target_x + 100, target_y)),
        ((target_x, target_y - 100), (target_x, target_y - 40)),
        ((target_x, target_y + 40), (target_x, target_y + 100))
    ]

    for p1, p2 in crosshair_coords:
        cv2.line(img, p1, p2, outline_color, outline_thick)
        cv2.line(img, p1, p2, main_color, main_thick)

    # Corner brackets
    bracket_size = 25
    bracket_coords = [
        (target_x - 70, target_y - 70, 1, 1),
        (target_x + 70, target_y - 70, -1, 1),
        (target_x - 70, target_y + 70, 1, -1),
        (target_x + 70, target_y + 70, -1, -1)
    ]

    for cx, cy, dx, dy in bracket_coords:
        cv2.line(img, (cx, cy), (cx + dx * bracket_size, cy), outline_color, outline_thick)
        cv2.line(img, (cx, cy), (cx, cy + dy * bracket_size), outline_color, outline_thick)
        cv2.line(img, (cx, cy), (cx + dx * bracket_size, cy), main_color, main_thick)
        cv2.line(img, (cx, cy), (cx, cy + dy * bracket_size), main_color, main_thick)

    # --- TEXT LABEL ---
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Main distance
    (tw, th), baseline = cv2.getTextSize(distance_str, font, 1.6, 3)
    (sw, sh), sub_base = cv2.getTextSize(subtitle_str, font, 0.8, 2)

    main_x = target_x - tw // 2
    main_y = target_y - 110
    sub_x = target_x - sw // 2
    sub_y = main_y + th + sh + 5

    # Background box
    pad = 15
    box_x1 = min(main_x, sub_x) - pad
    box_y1 = main_y - th - baseline - pad
    box_x2 = max(main_x + tw, sub_x + sw) + pad
    box_y2 = sub_y + sub_base + pad

    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
    cv2.rectangle(img, (box_x1, box_y1), (box_x2, box_y2), text_color, 3)

    # Text
    cv2.putText(img, distance_str, (main_x, main_y), font, 1.6, text_color, 3, cv2.LINE_AA)
    cv2.putText(img, subtitle_str, (sub_x, sub_y), font, 0.8, (200, 200, 200), 2, cv2.LINE_AA)

    output_path = image_path.replace(".jpg", output_suffix).replace(".png", output_suffix)
    cv2.imwrite(output_path, img)
    print(f"✓ Result saved: {output_path}\n")

    return output_path

# --- Main Execution ---
if __name__ == "__main__":

    # Upload image
    IMAGE_FILE = None
    try:
        from google.colab import files
        import shutil
        print("=" * 70)
        print("📸 UPLOAD IMAGE")
        print("=" * 70)
        uploaded = files.upload()
        original_name = list(uploaded.keys())[0]
        IMAGE_FILE = "input_image.jpg"
        shutil.copy(original_name, IMAGE_FILE)
        print(f"✓ Uploaded: {original_name}\n")
    except:
        import glob
        images = glob.glob("*.jpg") + glob.glob("*.png") + glob.glob("*.jpeg")
        images = [f for f in images if not any(x in f for x in ['depth', 'result', 'vis'])]
        if images:
            IMAGE_FILE = images[0]
            print(f"✓ Using: {IMAGE_FILE}\n")
        else:
            print("Error: No image found.")
            exit()

    # Initialize
    estimator = DepthReferenceEstimator()

    # Generate depth map
    print("=" * 70)
    print("STEP 1: DEPTH MAP GENERATION")
    print("=" * 70 + "\n")
    depth_map, vis_path = estimator.estimate_depth(IMAGE_FILE)

    # Get center reference point from actual image dimensions
    img = cv2.imread(IMAGE_FILE)
    h, w = img.shape[:2]
    ref_x = w // 2
    ref_y = h // 2

    # Visual distance estimation
    print("=" * 45)
    print("STEP 2: DISTANCE ESTIMATION")
    print("=" * 45 + "\n")
    print(" Distance is based on VISUAL ESTIMATION")
    print("   The scope marks the IMAGE CENTER as reference\n")

    # --- IMAGE-SPECIFIC VISUAL ESTIMATES ---
    # Each image gets its own distance estimate based on visual assessment
    # These are HUMAN ASSUMPTIONS, not automated measurements
    VISUAL_ESTIMATES = {
        "plant_image_1.jpg": 35.0,  # Example: close-up shot
        "plant_image_2.jpg": 55.0,  # Example: medium distance
        "plant_image_3.jpg": 70.0,  # Example: farther shot
        # Add more images as needed
    }

    # Get base filename for lookup
    base_filename = os.path.basename(IMAGE_FILE)

    # Try to get image-specific estimate, otherwise prompt user
    if base_filename in VISUAL_ESTIMATES:
        VISUAL_ESTIMATE_CM = VISUAL_ESTIMATES[base_filename]
        print(f"Using pre-configured visual estimate for '{base_filename}': {VISUAL_ESTIMATE_CM:.0f} cm")
        print("(From VISUAL_ESTIMATES dictionary)\n")
    else:
        print(f"No pre-configured estimate found for '{base_filename}'")
        print("\nHow to estimate distance:")
        print("  • Look at the CENTER of your image")
        print("  • Estimate the distance to the plant at that point")
        print("  • Enter your best visual estimate in centimeters\n")

        try:
            user_input = input("Enter visual distance estimate (cm): ")
            VISUAL_ESTIMATE_CM = float(user_input)
            print(f"\n✓ Using visual estimate: {VISUAL_ESTIMATE_CM:.0f} cm\n")
        except:
            # Fallback if input fails (e.g., in non-interactive environments)
            VISUAL_ESTIMATE_CM = 40.0
            print(f"\n  Input unavailable. Using default: {VISUAL_ESTIMATE_CM:.0f} cm")
            print("   Add this image to VISUAL_ESTIMATES dictionary for batch processing\n")

    # Calculate range
    d_min, d_center, d_max = calculate_distance_range(
        visual_estimate_cm=VISUAL_ESTIMATE_CM,
        uncertainty_pct=15
    )

    print(f"📍 Scope reference point: ({ref_x}, {ref_y})")
    print(f"   (Image center - consistent with your visual estimate)")
    print(f"\n📏 Distance range: {d_min:.0f}–{d_max:.0f} cm")
    print(f"   Based on visual estimate: {d_center:.0f} cm ±15%")
    print(f"\n⚠️  INTERPRETATION:")
    print(f"   • Scope marks the image center")
    print(f"   • Distance is a VISUAL ESTIMATE (human assumption)")
    print(f"   • Each image can have a different estimate")
    print(f"   • NOT an automated measurement")
    print(f"   • Model provides relative depth only, not metric scale")
    print(f"   • Treat as approximate reference only\n")

    # Draw scope overlay
    draw_scope_overlay(IMAGE_FILE, ref_x, ref_y, d_min, d_center, d_max)

    print("=" * 70)
    print("✓ COMPLETE")
    print("=" * 70)
