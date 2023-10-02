# Sample Codes

## CREATE Lab

I had the opportunity to be an intern at EPFL's [CREATE Lab](https://www.epfl.ch/labs/create/) for two months, under the expert guidance of [Prof. Josie Hughes](https://people.epfl.ch/josie.hughes?lang=en).

### Project 1: Sensor Glove
At the lab, a distinctive glove embedded with multiple sensors was developed by the members. These electric-based sensors measured the electric resistance from a predetermined point to a source. The measured parameters changed as a result of changes in the glove's shape caused by its electric properties. Our goal was to use the sensor data to reconstruct the hand position. To achieve this, we decided on a machine-learning strategy.

#### Data Preparation:
The crucial task was gathering precise data. It was easy to gather sensor data for different hand positions, but connecting this data to the hand position was challenging. My solution was to synchronise the video capturing the hand movement with sensor data recording, then determine the hand position by analysing the frames. For this, I employed Google’s open-source [Mediapipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) Python package, extracting 3D coordinates of the key hand points from the frames.

<div align="center">

| Glove Device | Mediapipe Sample Result |
|     :---:      |          :---: |
| <img src="https://user-images.githubusercontent.com/53098142/185445328-2f47635e-60fb-4535-bde8-b0229bb623db.jpg" alt="20210902_144346 - Copy" height="200"/>     | <img src="https://user-images.githubusercontent.com/53098142/185445430-d6d34def-ee4c-42bf-a984-d125f38a9e1a.png" alt="1_elevation 20" height="200"/>   |

</div>


#### Model Development:
Multiple machine learning models were explored, including deep neural networks and several classical methods. However, since the input data from the sensors was noisy and didn’t have clear information, getting accurate results was challenging, and a scientific paper could not materialise from this effort.
### Project 2: Silicon-Casted Trunk
This project was an experiment in soft robotics with the goal of coming up with a way to identify the shape of the silicon-casted trunk. Initially, a camera was placed inside the trunk, and a machine-learning method was to be created to determine the shape of the trunk from the images that were captured.

#### Strategy and Model Training:
The lack of training data I faced could be overcome by adding an extra external camera. We were able to collect enough image pairs as a result to this setup to provide major training data. I divided the task into two phases to start the problem-solving process. In the first stage, a model was created that could anticipate the second camera's image based on the information from the first. The second stage involved analysing the predicted image to extract the relevant features of the trunk. Before deploying the actual dataset, the model was pre-trained on simulated pairs to improve performance. An auto-encoder architecture was chosen due to the nature of the task.

<div align="center">

|  | Input Image | Output Image |
| :---:         |     :---:      |          :---: |
| Simulated Data   | <img src="https://user-images.githubusercontent.com/53098142/185436057-86b22cfe-35b6-409b-a1e0-8822065ca788.png" alt="52 input" width="200"/>     | <img src="https://user-images.githubusercontent.com/53098142/185436044-4f697037-8783-4cc3-8c31-4eed0a4f9bb6.png" alt="52 output" width="200"/>   |
| Real Data     | <img src="https://user-images.githubusercontent.com/53098142/185436048-8c811bba-5c8f-4e58-a25d-5f6ff25b65fe.png" alt="245 input" width="200"/>       | <img src="https://user-images.githubusercontent.com/53098142/185436051-25ad575e-c0bb-429f-b876-8bf284f607c0.png" alt="245 output" width="200"/>     |

</div>

#### Overcoming Challenges:
The first camera experienced occlusions, obscuring the view of the trunk's last part. Despite this, the model successfully predicted the initial part of the trunk. To counter the occlusion issue, we thought about including a pressure sensor beside the camera to aid decision-making. However, due to time limitations, this solution was not implemented.
### Conclusion:
My time at the CREATE Lab was full of learning, exploring, and solving problems in soft robotics and using machine learning. Even though we had some roadblocks, like time limits and not enough data, I learned so much. This experience has given me a strong starting point for more work in the application of artificial intelligence in robotics in the future.



## Master Thesis: Acute Lymphoblastic Leukemia Diagnosis

My Master's thesis involved developing a model to diagnose Acute Lymphoblastic Leukemia (ALL), a prevalent childhood blood cancer, from blood smear images. Early diagnosis is crucial to initiate timely treatment, saving patients’ lives, hence the significance of this project. The task was to enable the model to mirror hematologists' diagnostic process, examining blood smear images, a task traditionally performed by expert doctors.

### Problem Definition and Challenges:

In medical diagnostics, having limited training data is a common hurdle. This project also faced such constraints, necessitating strategies at every stage to counter the limited dataset sizes. The reliability of the model is paramount, with a focus on ensuring the model makes meaningful and accurate decisions. The necessity arose from the observation that high diagnostic accuracy alone could lead to models taking shortcuts due to the small size of medical training datasets.

### Solution and Methodology:

I introduced a novel pipeline that mimics the process used by hematologists, constraining the model to follow a pipeline inspired by experts' work and redefining the problem as a multiple-instance learning problem. This was pivotal to achieving practical and reliable results, as judgments based on only one image proved insufficient. The figure below illustrates the final architecture used for this project.

<div align="center">
  
| Final Architecture of the Model |
| :---: |
| <img src="https://github.com/askari1375/Sample-Codes/assets/53098142/d463370a-419d-4e4e-b325-fcf78a854019" alt="Architecture of Model" width="600"/> |
  
</div>




A multitude of deep-learning techniques were employed and tested during this project, such as:
1. Object detection with Faster RCNN
2. Segmentation with U-Net
3. LSTM models
4. Self-Supervised Learning
5. Vision transformers

### Results and Evaluation:

The model achieved an accuracy of 96.15%, an F1-score of 94.24%, a sensitivity of 97.56%, and a specificity of 90.91% on ALL IDB 1. It demonstrated resilience and acceptable performance even on an out-of-distribution dataset, serving as a challenging test. The results are documented in detail in my paper available on [Arxiv](https://arxiv.org/abs/2307.04014).

<div align="center">

| Sample Input Image | Sample Cell Detection Result |
|     :---:      |          :---: |
| <img src="https://user-images.githubusercontent.com/53098142/185441050-24c06613-2d2b-4fc8-a25b-59bdbe3a807e.jpg" alt="Im006_1" height="200"/>     | <img src="https://user-images.githubusercontent.com/53098142/185441792-0155359f-ff5a-43f5-9dc9-c82cdf59afb5.jpg" alt="Im012_1" height="200"/>   |

</div>

### Conclusion:

This project not only advanced the field of medical diagnostics with its novel approach and methodological advancements but also underscored the potential of employing advanced machine learning models in critical healthcare domains, even with limited data availability. It set a precedent for employing a multiple-instance learning setup to solve intricate medical diagnosis problems, emphasizing meaningful decision-making processes aligned with expert practices.






## Master Thesis

Medical data processing for disease diagnosis is one of the uses of machine learning and artificial intelligence. In this study, blood cell images, a sort of histopathology imaging, were the main emphasis, and the model was designed to be able to identify a specific type of blood cancer called acute lymphoblastic leukemia (ALL) from those photos.

Lack of training data and short dataset sizes are common issues that frequently arise in medical settings. This project was not an exception, and this restriction was taken into account at every turn. The appropriate strategy was used at each stage to overcome this obstacle.

In medical cases, the model's reliability is also critical. In this study, a novel pipeline was created that can make decisions using relevant biomarkers and do so with high accuracy. The study that will be published soon will provide further information.

Several deep-learning techniques were tested in this project, and some of them were used. Here are a few of them as an illustration:

1. object detection with Faster RCNN
2. Segmentation with U-Net
3. LSTM models
4. Self-Supervised Learning
5. Vision transformers


<div align="center">

| Sample Input Image | Sample Cell Detection Result |
|     :---:      |          :---: |
| <img src="https://user-images.githubusercontent.com/53098142/185441050-24c06613-2d2b-4fc8-a25b-59bdbe3a807e.jpg" alt="Im006_1" height="200"/>     | <img src="https://user-images.githubusercontent.com/53098142/185441792-0155359f-ff5a-43f5-9dc9-c82cdf59afb5.jpg" alt="Im012_1" height="200"/>   |

</div>



## Web Crawler

Python code that can crawl [this](https://searchingfortruth.ir/) website was written for this project. This code's objective is to create a Latex file from articles on this website. Running this code will produce a ".tex" file that can be used by a Latex compiler. To achieve this, the following tasks had been completed:

1. Using the [Scrapy framework](https://scrapy.org/), a crawler was created to read website articles.

2. The outcome of the preceding step needs to be processed because it is not clean enough.

3. The entire text should be combined in the proper Latex format.
