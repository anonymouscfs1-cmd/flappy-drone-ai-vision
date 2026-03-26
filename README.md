# Flappy Drone AI Vision

## Description

This project explores the use of AI and computer vision to analyze drone footage in a greenhouse environment.

The motivation behind this project was to understand how visual data from drones can be transformed into meaningful information for real-world applications, such as monitoring environments or enabling autonomous navigation.

The project focuses on processing raw video data and enhancing it using depth estimation techniques, allowing for a better understanding of spatial structure in complex environments.

Through this project, I learned how important data preprocessing, model selection, and real-world testing conditions are in building effective AI systems.

---

## Table of Contents
- [Usage](#usage)
- [Technologies](#technologies)
- [Results](#results)
- [Credits](#credits)
- [Future Work](#future-work)
- [Visuals](#visuals)
- [License](#license)

---

## Usage

This project processes drone footage recorded in a greenhouse environment and applies depth estimation techniques to extract additional spatial information.

Example workflow:
1. Input: raw drone video footage  
2. Process: apply depth estimation model (e.g., AnythingDepth)  
3. Output: enhanced frames with depth information  

You can extend this pipeline for:
- object detection  
- obstacle avoidance  
- environmental analysis  

---

## Technologies

- Python  
- Computer Vision  
- Roboflow  
- Depth Estimation Models (e.g., AnythingDepth)

---

## Results

- Converted drone video into structured data  
- Applied depth estimation successfully to visual data  
- Identified challenges such as:
  - motion blur  
  - lighting variation  
  - dynamic environments  

---

## Credits

This project was developed as part of a team collaboration.

My contributions included:
- Processing drone video footage  
- Applying depth estimation models  
- Preparing datasets for analysis  
- Experimenting with models using Roboflow  

---

## Future Work

- Implement object detection (plants, pests, obstacles)  
- Integrate AI for autonomous drone navigation  
- Improve dataset quality and model accuracy  
- Combine depth estimation with real-time decision making  

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


## License

This project is licensed under the MIT License.
