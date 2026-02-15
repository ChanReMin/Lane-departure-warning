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
  <img src="assets/2d77da0a-3ff8-4da8-b72f-f2531b8a461c.jpg">
</p>

---

## Hardware Requirements

- Raspberry Pi 5
- Coral USB Accelerator (Edge TPU)
- Camera attached to Raspberry Pi
- Buzzer + LED for warnings  
- Powerbank ≥ 2.4A (if testing in car)  

---
# Example how to connect to your pi through ssh without pi GUI
By this way, you'll never need a monitor or keyboard to use Raspberry Pi
## Check your IP range on Window by running
**IMPORTANT** 
Make sure that you've already setup Pi to connect to same wifi with your PC when using Pi Imager to create boot image
```
ipconfig
```
## Scap the IP and MAC Address of your Pi
After getting your IP, eg 192.168.1.5, use nmap to scan all the IP in 192.168.1.x
installing nmap
```
winget install nmap
nmap -sn 192.168.1.0/24
```
Example output
```
Nmap scan report for gpon.net (192.168.1.1)
Host is up (0.00s latency).
MAC Address: A0:09:2E:18:D8:8E (Unknown)
Nmap scan report for 192.168.1.2 (192.168.1.2)
Host is up (0.063s latency).
MAC Address: 00:D2:79:B4:02:D6 (Unknown)
Nmap scan report for 192.168.1.3 (192.168.1.3)
Host is up (0.00s latency).
MAC Address: 3C:06:A7:1F:35:72 (Unknown)
Nmap scan report for 192.168.1.4 (192.168.1.4)
Host is up (0.016s latency).
MAC Address: 1C:90:FF:35:A9:3C (Unknown)
Nmap scan report for 192.168.1.8 (192.168.1.8)
Host is up (0.016s latency).
MAC Address: 4C:24:CE:13:B8:E1 (Unknown)
Nmap scan report for 192.168.1.9 (192.168.1.9)
Host is up (0.00s latency).
MAC Address: B0:BE:83:2B:55:2F (Unknown)
Nmap scan report for 192.168.1.10 (192.168.1.10)
Host is up (0.016s latency).
MAC Address: 4C:24:CE:13:B8:E1 (Unknown)
Nmap scan report for 192.168.1.12 (192.168.1.12)
Host is up (0.016s latency).
MAC Address: 2C:CF:67:38:FE:6C (Unknown)
Nmap scan report for 192.168.1.6 (192.168.1.6)
Host is up.
Nmap scan report for 192.168.1.11 (192.168.1.11)
Host is up.
Nmap done: 256 IP addresses (10 hosts up) scanned in 5.27 seconds
```
## Try connect to your Pi for the first time
For first time finding, try ssh to all these IP with this command
```
ssh <raspberrypi host>@<IP>
```
if prompt out the password entering,then it is the right Pi IP, you can now have the IP of your raspberry pi now
save the Mac address below that IP, eg:
```
Nmap scan report for 192.168.1.12 (192.168.1.12)
Host is up (0.016s latency).
MAC Address: **2C:CF:67:38:FE:6C** (Unknown)
```
Next time, whenever you want to connect to Pi, use nmap to scan and choose the IP which the exact MAC Address like before if Pi IP change

