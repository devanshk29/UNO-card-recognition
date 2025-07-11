# UNO Card Detection using Pattern Matching

**Team Uno Reverse**  
*Amey Karan (2022111026), Devansh Kantesaria (2022112003)*  
*IIIT Hyderabad*

---

##  Project Overview

This project presents a computer vision system that automatically detects and recognizes UNO cards based on their **number** and **color (suit)** using classical **pattern matching techniques** in OpenCV.

The system accepts an image of a UNO card as input and outputs the card’s **number** and **color**. To support this, a custom dataset of over **5,000+ UNO card images** in different orientations and conditions was created.

---

##  Key Features

- **Pattern Matching-Based Detection**  
  Recognizes card numbers via template matching and identifies card color through color segmentation.

- **Fully Automated Pipeline**  
  Performs preprocessing, edge detection, perspective correction, number and color classification.

- **Custom Dataset**  
  Built from scratch using three different UNO decks, enhancing generalizability across card styles.

---

##  Methodology

1. **Image Reading**
   - Load and rotate images for uniform orientation.
   - Apply Gaussian blur and Canny edge detection.

2. **Card Segmentation & Corner Detection**
   - Use morphological operations to isolate the card.
   - Detect edges and compute intersections to identify corners.

3. **Perspective Transformation**
   - Warp the image to a standard orientation using corner coordinates.

4. **Skeletonization & Template Matching**
   - Focus on the center of the card and detect number contours.
   - Match with multiple templates to classify the card number.

5. **Color Detection**
   - Use color region analysis to classify card suit (Red, Green, Blue, Yellow) with **100% accuracy**.

---

##  Results

- **Overall Accuracy:** 77%  
- **Color Detection:** 100% accuracy  
- **Challenges:** Cards like `1 ↔ 7` and `3 ↔ 8` showed confusion due to design similarity.

---

##  Challenges & Limitations

- **Perspective distortion** and **orientation variation** required advanced preprocessing.
- **Design variability** across decks impacted template matching.
- **Highly symmetric** designs hindered standard feature detectors like SIFT.

---

##  Dataset

- Total: **~5,700 images**
- Cards: Only **number cards** across **4 colors** (no wild/action cards included)
- Images captured from **3 distinct decks** with varied orientations and lighting

---

##  Future Work

- Include **wild** and **action** cards in classification
- Use deep learning for better **generalization**
- Improve robustness to **real-world distortions**

---

##  Tech Stack

- **Language**: Python  
- **Libraries**: OpenCV, NumPy  
- **Concepts Used**: Image Processing, Morphological Operations, Template Matching, Color Segmentation

---

##  References

- Lowe, D. G. “Distinctive image features from scale-invariant keypoints.” *IJCV*, 2004.  
- Snyder, D. "Playing card detection and identification." *Stanford EE368*, 2019.  
- Pimentel, J. & Bernardino, A. “A comparison of methods for detection and recognition of playing cards.”

---
