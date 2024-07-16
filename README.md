# Emotion Analyzer

## 01 Repo Structure

```
    ├── docs                                   # Contains documents  
    ├── research                               # Contains pre-research. 
    ├── source                                 # Contains project source code.
    │   ├── main.php                               # Main app file.
    └── README.MD                              # Readme Content.
    └── LICENSE.MD                             # LICENSE of the project.
```


## 02 Introduction

The Emotion Analyzer is a Python-based application designed to detect and analyze human emotions from facial expressions in images or video streams. This system leverages the power of machine learning and computer vision, utilizing pre-trained models to recognize a variety of emotions with high accuracy.

### 2.1 Features

- Emotion Detection: Detects emotions such as happy, sad, angry, surprised, and more.
- Real-time Analysis: Capable of analyzing emotions in real-time using webcam input.
- Image and Video Support: Supports both image files and video streams for emotion analysis.

![Video](docs/media/images/0-banner-image.png)

## 03 Tech Stack

- Python 3.8
-  

## 04 Setup


- **Step 01:** Install Python

  ```
    https://python.org/
  ```

- **Step 02:** Navigate to docs folder


  ```
   cd docs
  ```

- **Step 03:** Install the requirements.txt

  ```
    pip install -r requirements.txt
  ```

- **Step 04 (Optional | Anaconda Env Only):** Install the environment.yml (If you are using Anaconda environment)

  ```
    conda env create -f environment.yml
  ```

## 05 Usage

### 1. Demo

![Video](docs/media/videos/smile.mp4)

### 2. Demo

![Video](docs/media/videos/surprise.webm)

## Requirements

- ```tensorflow==2.9.1```
- ```imutils==0.5.4```
- ```numpy==1.22.4```
- ```opencv-python==4.6.0.66```
- ```deepface==0.0.75```


## Files

- deploy.prototxt
- res10_300x300_ssd_iter_140000.caffemodel

## Installation

1. Clone the repository:

      ```
        git clone https://github.com/yourusername/emotion-analyzer.git
      ```
      ```
        cd emotion-analyzer
      ```
2. Install the required libraries:

      ```
        pip install -r requirements.txt
      ```

## Usage

- Running the emotion analyzer on a video stream:

    ```
      python run.py
    ```

- Running the emotion analyzer on a video file:

    ```
      python video-file-method.py
    ```

## Acknowledgements

1. ```Python = 3.8```
2. ```tensorflow==2.9.1```
3. ```imutils==0.5.4```
4. ```numpy==1.22.4```
5. ```opencv-python==4.6.0.66```
6. ```deepface==0.0.75```


## Contact

- [Website](https://www.gunarakulan.info/)