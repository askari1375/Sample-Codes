# Sample Codes

## CREATE Lab

I participate as an intern for two months in [CREATE Lab](https://www.epfl.ch/labs/create/) at **EPFL** under the supervision of [Prof. Josie Hughes](https://people.epfl.ch/josie.hughes?lang=en).

### Glove

The lab members created a special glove with multiple sensors. these sensors measure some electric resistance between the point that they were connected and a certain source. The measured parameters changed when the shape of the glove changed because of the glove's electric characteristics. This project's goal was to use sensor data to recreate hand positions. A machine-learning strategy was chosen to solve this issue.

For this problem, labeled data, sensor data for input, and hand position for output are required while training a machine learning method. So, it is important to determine hand position. The [Mediapipe](https://google.github.io/mediapipe/solutions/hands.html) python package was used to extract hand key points from the camera-captured image to determine the hand position.

Several machine learning techniques, including deep neural networks, were examined for model development. Those models didn't yield accurate results since the glove data was noisy and of low quality.


<div align="center">

| Glove Device | Mediapipe Sample Result |
|     :---:      |          :---: |
| <img src="https://user-images.githubusercontent.com/53098142/185445328-2f47635e-60fb-4535-bde8-b0229bb623db.jpg" alt="20210902_144346 - Copy" height="200"/>     | <img src="https://user-images.githubusercontent.com/53098142/185445430-d6d34def-ee4c-42bf-a984-d125f38a9e1a.png" alt="1_elevation 20" height="200"/>   |

</div>




### Trunk

This project was in the field of soft robotics. The project's goal was to discover a suitable solution to the issue of determining the shape of the trunk at each instant for a silicon-cast trunk. The initial suggestion was to use a camera to keep an eye on the trunk. The absence of labeled data for this topic posed a significant challenge, so another camera was employed to collect the trunk's position from outside of that.

Some pretraining on simulated data has been conducted to enhance the results. The network's architecture resembled that of auto-encoders, which take an image as input and output. Due to occlusions, there was no information on the end section of the trunk in the acquired image. As expected, the model was unable to predict the position of the end part very well. However, this network was able to predict the position of the initial part of the trunk very well.

The use of a pressure sensor was suggested as a possible solution to this problem, but it wasn't put to the test because of time constraints.

<div align="center">

|  | Input Image | Output Image |
| :---:         |     :---:      |          :---: |
| Simulated Data   | <img src="https://user-images.githubusercontent.com/53098142/185436057-86b22cfe-35b6-409b-a1e0-8822065ca788.png" alt="52 input" width="200"/>     | <img src="https://user-images.githubusercontent.com/53098142/185436044-4f697037-8783-4cc3-8c31-4eed0a4f9bb6.png" alt="52 output" width="200"/>   |
| Real Data     | <img src="https://user-images.githubusercontent.com/53098142/185436048-8c811bba-5c8f-4e58-a25d-5f6ff25b65fe.png" alt="245 input" width="200"/>       | <img src="https://user-images.githubusercontent.com/53098142/185436051-25ad575e-c0bb-429f-b876-8bf284f607c0.png" alt="245 output" width="200"/>     |

</div>


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



## Crawler

Python code that can crawl [this](https://searchingfortruth.ir/) website was written for this project. This code's objective is to create a Latex file from articles on this website. Running this code will produce a ".tex" file that can be used by a Latex compiler. To achieve this, the following tasks had been completed:

1. Using the [Scrapy framework](https://scrapy.org/), a crawler was created to read website articles.

2. The outcome of the preceding step needs to be processed because it is not clean enough.

3. The entire text should be combined in the proper Latex format.
