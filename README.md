
  
# Improving Checkout Experience in Retail Stores - Automated Shopping Experience

To use Computer vision to improve the customer experience in a retail stores. We plan to track the customers inside the store, detect their activities and automatically bill them towards the end. This results in a cashier-less retail store reducing run-time expenses for the store owners.

## Approach
In this project, we plan to tackle this problem in three phases.
- Person Identification and tracking
- Object Detection
- Activity Recognition

As soon as the person enters the store, he is identified by the cameras by analyzing the store’s database and is tracked until he/she checks out. The objects and the joint locations of the customers inside the store are
continuously monitored with the help of object detection and pose estimation algorithms. So the actions of a person such as picking up or dropping an object can be recognized by looking at the relative distance between the object and the customer or by running an activity detection algorithms. The objects are appended to the cart of the appropriate customer and price of the cart will be deducted from the customer’s

##  Modules
### Object Detection
**Research:** 
- Single Shot Detector. (**Chosen**)
- Faster RCNN.
- YOLO.

**Results:** Object detection using SSD algorithm was implemented with 8 different types of object within a single environment. An mAP score of 87.36% was obtained.

### Person Detection and Tracking
**Research:** 
- Realtime Multi-Person 2D Pose Estimation using Part Affinity Field. (**Chosen**)
- Alpha Pose Multi-scale Deep Learning Architectures for Person Re-identification.
- One-Shot Video-Based Person Re-Identification by Stepwise Learning
- Face recognition
- Real-time Human Detection in Computer Vision.

**Results:** Check the Final Report.

###  Activity Recognition
**Research:** 
- A Closer Look at Spatiotemporal Convolutions for Action Recognition.
- Non-local Neural Networks.
- Long Short Term Memory Networks. (**Chosen**)
- Support Vector Machines.
- Pure Algorithms.

**Results:** Accuracy for customer actions: 76.979%.

## Fun Video
Youtube: https://www.youtube.com/watch?v=sIdVjU_iGqA
