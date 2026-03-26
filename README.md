# Agrifly – Flappy Drone AI Vision

## Project Context

The **Flappy project** is an initiative that develops small flapping-wing drones to monitor greenhouse crops. The drones fly between plant rows, capture leaf images, and use AI to detect pests and diseases early — reducing the need for chemical pesticides and manual scouting.

This repository contains my personal contributions to the project, developed as part of a university AI minor.

---

## My Role & Contributions

I contributed across all four project phases (Climbs), working on depth estimation, model experimentation, dataset work, and workflow research. My primary technical focus was on building a **camera-to-plant distance estimation pipeline** using the Depth-Anything model, validated on real drone footage.

---

## Table of Contents

- [Depth Estimation Pipeline](#depth-estimation-pipeline)
- [Key Features](#key-features)
- [Technologies](#technologies)
- [Results & Findings](#results--findings)
- [Challenges](#challenges)
- [Visuals](#visuals)
- [Future Work](#future-work)
- [Credits](#credits)
- [License](#license)

---

## Depth Estimation Pipeline

The core of my work was developing a **demo-ready distance estimation tool** that works on both images and video (including real drone footage). The pipeline uses [Depth-Anything](https://github.com/LiheYoung/Depth-Anything) to estimate the relative distance between the drone camera and a plant leaf.

### Pipeline Overview

```
Input (Image / Video)
        ↓
Depth-Anything Model (relative depth map)
        ↓
Manual Calibration (user provides reference distance in cm)
        ↓
Vegetation Locking (HSV green mask)
        ↓
Leaf Tracker (IoU + centre distance + size consistency)
        ↓
Confidence Gating + Hold Logic
        ↓
Scope Overlay Output (distance displayed in-frame)
```

### Development Steps

**Step 1 – Proof of concept**  
Started with a single plant image from the project dataset to verify that Depth-Anything could run on greenhouse imagery end-to-end.

**Step 2 – Depth visualisation**  
Generated a purple/yellow depth colormap where bright/yellow = closer, dark/purple = farther. This made it easy to visually validate depth quality without reading raw numbers.

**Step 3 – Identifying the relative depth limitation**  
Confirmed that Depth-Anything produces *relative* depth only — no trustworthy real-world distance without calibration. This became a hard constraint for the roadmap.

**Step 4 – Manual calibration**  
Implemented a user-input calibration step: the user estimates the distance to the centre plant in centimetres, which is used as the scaling reference. This kept the pipeline transparent and prevented misleading metric outputs.

**Step 5 – Scope overlay & upload flow**  
Added a visual scope overlay that displays the estimated distance directly inside the image/video frame. Rewrote the input flow into a button-based upload system to reduce file path errors in Google Colab.

**Step 6 – Image to video: framewise vs temporal modes**  
Extended the pipeline from single-image to video processing. Tested both framewise (independent per frame) and temporal (smoothed across frames) modes on a YouTube greenhouse video.

**Step 7 – Real drone footage validation**  
Switched to the project owner's actual drone footage with fast movement, blur, and shake — much closer to real Flappy conditions. This made instability clearly visible and drove all subsequent improvements.

**Step 8 – Vegetation locking (HSV)**  
Added an HSV-based green mask to restrict the scope to plant regions and ignore greenhouse structures, shadows, and background clutter.

**Step 9 – Hold logic**  
Implemented hold logic so that during unreliable frames (motion blur, brief occlusions), the system retains the last valid distance rather than outputting noise. Distance resumes updating when frame quality stabilises.

**Step 10 – Leaf Tracker integration**  
Integrated a trajectory-style Leaf Tracker using association signals (IoU, centre distance, size consistency) to maintain tracking of the same leaf over time, preventing target switching when other leaves enter frame.

**Step 11 – Confidence gating**  
Updated the system so distance values only update when both tracking confidence and frame quality are sufficiently high — significantly reducing distance "jumps" on shaky drone footage.

**Step 12 – Debug output & packaging**  
Added an optional depth debug video output (purple/yellow colormap) to make the depth signal visible during reviews. Packaged the full notebook into a clean, reproducible demo workflow with consistent inputs, toggles, and output naming.

---

## Key Features

- Works on both **images and videos** including real drone footage
- **Manual calibration** for transparent, user-controlled distance scaling
- **Scope overlay** displaying estimated distance in-frame
- **Vegetation locking** via HSV green masking to ignore background
- **Hold logic** for stability during blur and occlusion
- **Leaf Tracker** for consistent target tracking across frames
- **Confidence gating** to suppress noisy distance updates
- **Debug mode** for visualising the raw depth map alongside output
- Runs fully in **Google Colab** with button-based file upload

---

## Technologies

- Python
- [Depth-Anything V2](https://github.com/LiheYoung/Depth-Anything) (monocular depth estimation)
- OpenCV
- PyTorch
- Google Colab
- Roboflow (dataset & model management, used across project)
- YOLOv12 / RF-DETR (project-wide pest detection models)
- Zapier + Supabase (automated workflow research)

---

## Results & Findings

- Successfully built a **complete image and video depth estimation pipeline** validated on real greenhouse drone footage
- Demonstrated that Depth-Anything produces a logical depth signal on plant imagery, with stable relative rankings between close and far regions
- Close-range metric values required calibration to be meaningful — changing the environment setting (outdoor → indoor) had no effect; improvement came from better depth scaling logic
- The **vegetation locking, hold logic, tracker, and confidence gating** together produced noticeably more stable distance output on shaky drone footage
- The prototype is **demo-ready** as a steppingstone baseline, but further benchmarking against ground-truth measurements is needed before deployment

---

## Challenges

- **Relative depth only** — Depth-Anything cannot produce trustworthy metric distances without a known reference, requiring manual calibration
- **Motion blur & shake** on real drone footage caused distance flicker and target switching, requiring multiple stability layers
- **HSV vegetation locking** can fail under certain lighting conditions; it improved robustness but did not fully eliminate false locks
- **Extreme blur** could still confuse the Leaf Tracker despite hold logic
- No GPS or ground-truth distance data was available in the dataset for objective validation

---

## Visuals

### Step 1: Raw Image
Original raw image captured for depth estimation testing.

<img src="https://github.com/user-attachments/assets/2ae0a55b-582a-409c-bb2c-2920e361b84c" alt="Raw Image" width="600"/>


### Step 2: Depth Estimation Output
Applied AnythingDepth on the raw image. Purple indicates far objects, yellow indicates objects close to the camera.

<img src="https://github.com/user-attachments/assets/9e29aad0-c788-4dda-abb2-1b6380984144" alt="Depth Estimation" width="600" style="transform: rotate(90deg);"/>


### Step 3: Distance Estimation
Custom code used to estimate distance from the camera to the mid-point of the plant leaf. This step helped quantify plant proximity.

<div style="display: flex; gap: 10px; flex-wrap: wrap;">

  <img src="https://github.com/user-attachments/assets/f8bae0c2-cacd-467f-905c-c4a97ebe1e4e" 
       alt="Raw Image" 
       width="400" 
       style="height: auto;"/>
  
  <img src="https://github.com/user-attachments/assets/68dd7548-8548-4ce2-bd28-4c0e4d2c293b" 
       alt="Distance Map" 
       width="400" 
       style="height: auto;"/>
  
</div>


### Step 4: YouTube Video Test
Tested temporal mode vs framewise mode on YouTube drone footage. This video was used for experimentation but was not suitable for the final project.

<img width="1173" height="851" alt="image" src="https://github.com/user-attachments/assets/1a0c09be-b3a2-47c5-8d80-9ea37cf2b86a" />


### Step 5: Drone Footage
Raw footage received from the project owner. This became the primary input for the project.

<img width="1417" height="705" alt="image" src="https://github.com/user-attachments/assets/78d6046a-315a-4730-82e8-e4aa1fb9006c" />

### Step 6: Drone Depth Visualization
Applied AnythingDepth to real drone footage. Leaves close to the camera appear yellow, farther ones are purple.

<img width="1491" height="670" alt="image" src="https://github.com/user-attachments/assets/f987e1f2-5e4c-4416-90ac-f95a84c4f8a6" />

### Step 7: Normal Drone Footage
Optional code to view the drone footage without the AnythingDepth overlay. This version attempts to focus on the closest leaf using a green bounding box and an estimated distance.

This step was the most challenging due to:
- Motion blur from the drone and flapping leaves  
- Leaves not being clearly visible in some frames  
- High movement and dynamics in the environment  

<img width="934" height="1069" alt="image" src="https://github.com/user-attachments/assets/bab75123-6ffc-4988-8018-eac9bae886af" />

---

## Future Work

- Benchmark calibrated distance against real measured distances across multiple plants, angles, and lighting conditions
- Improve tracker recovery under heavy blur and occlusion
- Fine-tune HSV sensitivity for more reliable vegetation locking across greenhouse lighting conditions
- Tune confidence gating thresholds based on more drone footage
- Integrate distance estimation with the pest detection pipeline so the drone knows when it is close enough for reliable leaf imaging
- Explore automatic calibration using known physical references in the greenhouse

---

## Credits

This project was developed as part of a team collaboration on the Flappy Drone project.

The Leaf Tracker logic and the depth estimation pipeline in this repository represent my own independent work. A teammate separately developed their own depth estimation baseline (Roadmap Variation 1) as a parallel approach within the same project.

---

## License

This project is licensed under the MIT License.
