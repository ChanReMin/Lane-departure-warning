# Lane Departure Warning System

🚗⚠️ **AI-powered Lane Departure Warning for Dashcam with Raspberry Pi + Coral Edge TPU**

<p align="center">
  <img src="assets/demo.gif">
</p>

## Overview

This project implements a **real-time Lane Departure Warning System** using semantic segmentation.  
The model detects road lanes from dashcam footage, processes the results on **Raspberry Pi**, and triggers a **buzzer + LED** to warn the driver if the car is drifting out of its lane.  

- **Lightweight & Fast**: Optimized to run on **Coral Edge TPU** for low-latency inference.  
- **Portable**: Can run directly from a powerbank and integrate with a car as a dashcam device.  
- **Reliable**: Works in real-world scenarios including low visibility conditions.  

---

## Models
- Lane finder model from this repository: https://github.com/xadrianzetx/lanefinder
- Backbone: **MobileNetV2 + U-Net decoder**  
- Deployment: Converted to TensorFlow Lite with **full integer quantization** for Edge TPU acceleration.  

## Features

- **Semantic Segmentation** of road lanes in real-time.  
- **Lane Departure Detection**: Calculates lane alignment and determines if the car is drifting.  
- **Warning System**:  
  - 🔊 **Buzzer** when lane departure is detected.  
  - 💡 **LED light** alert for visual feedback.  

---

## Performance

- **Camera feed mode**: ~30 FPS on Raspberry Pi + Coral Edge TPU.  
- **Video playback mode**: ~10 FPS (used only for testing/debugging).  

---

## Prototype

<p align="center">
  <img src="assets/prototype.gif">
</p>

---

## Hardware Requirements

- Raspberry Pi 5
- Coral USB Accelerator (Edge TPU)
- Camera attached to Raspberry Pi
- Buzzer + LED for warnings  
- Powerbank ≥ 2.4A (if testing in car)  

---

## Installation & Run

1. Enable the camera interface:  
   ```bash
   sudo raspi-config
