# Sample Codes

## CREATE Lab

### Glove


### Trunk

This project was in the field of soft robotics. The project's goal was to discover a suitable solution to the issue of determining the shape of the trunk at each instant for a silicon-cast trunk. The initial suggestion was to use a camera to keep an eye on the trunk. The absence of labeled data for this topic posed a significant challenge, so another camera was employed to collect the trunk's position from outside of that.

Some pretraining on simulated data has been conducted to enhance the results. The network's architecture resembled that of auto-encoders, which take an image as input and output. Due to occlusions, there was no information on the end section of the trunk in the acquired image. As expected, the model was unable to predict the position of the end part very well. However, this network was able to predict the position of the initial part of the trunk very well.

The use of a pressure sensor was suggested as a possible solution to this problem, but it wasn't put to the test because of time constraints.


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


## Crawler

Python code that can crawl [this](https://searchingfortruth.ir/) website was written for this project. This code's objective is to create a Latex file from articles on this website. Running this code will produce a ".tex" file that can be used by a Latex compiler. To achieve this, the following tasks had been completed:

1. Using the [Scrapy framework](https://scrapy.org/), a crawler was created to read website articles.

2. The outcome of the preceding step needs to be processed because it is not clean enough.

3. The entire text should be combined in the proper Latex format.
