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
![Raw Image](assets/images/step1-raw.png)<img width="1200" height="1600" alt="image" src="https://github.com/user-attachments/assets/2ae0a55b-582a-409c-bb2c-2920e361b84c" />


### Step 2: Depth Estimation
Applied AnythingDepth on the raw image. Purple indicates far objects, yellow indicates objects close to the camera.
![AnythingDepth Output](assets/images/step2-anythingdepth.png)

### Step 3: Distance Estimation
Custom code used to estimate distance from the camera to the mid-point of the plant leaf. This step helped quantify plant proximity.
![Distance Estimation](assets/images/step3-distance-estimation.png)

### Step 4: YouTube Video Test
Tested temporal mode vs framewise mode on YouTube drone footage. This video was used for experimentation but was not suitable for the final project.
![YouTube Test](assets/images/step4-youtube-test.png)

### Step 5: Drone Footage
Raw footage received from the project owner. This became the primary input for the project.
![Drone Footage](assets/images/step5-drone-footage.png)

### Step 6: Drone Depth Visualization
Applied AnythingDepth to real drone footage. Leaves close to the camera appear yellow, farther ones are purple.
![Drone Depth](assets/images/step6-drone-depth.png)

### Step 7: Normal Drone Footage
Optional code to view the drone footage without depth overlay, showing the original visuals.
![Normal Drone](assets/images/step7-drone-normal.png)

---

## License

This project is licensed under the MIT License.
