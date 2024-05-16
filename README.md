
# Satellite Image Segmentation using U-Net Deep Learning Architecture
<p align="center"> <b>Worked on by klaypso</b> </p>

# TL;DR
This project contains code for a U-Net segmentation model that can be used to segment roads and water bodies in high-resolution satellite images. If interested in understanding each process, please read through the (optional) explanations.

# Getting Started
Use **Conda** to clone the isro.yml file to setup my environment for this project. 

# Project Dataset
The data used comes from the DSTL's Kaggle competition "Can you train an eye in the sky?". You will need to get the data from the competition's [page](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data), after accepting all terms and conditions. 

# Project Structure
In this project, individual binary-prediction problems are solved. For example: Road and Not-Road, Water and Not-Water.

# U-Net Model
A U-Net model is used for segmenting roads. 

# Using Pretrained Weights
You can use the pretrained weights downloaded from [here](https://drive.google.com/drive/folders/1YGMZVCRn5UVlQL3V-MTwT4QHtI7HJJSX?usp=sharing) and place them in `Parameters/Road_tar` folder. Remember to modify the first line of the `checkpoint` file to your local project path. 

Run the below command to test the model:
```python test_roads.py
```

Alternatively, if you want to train the model from scratch, you will need to follow the instructions provided in the original readme content.

# Results
For tarred roads, we achieve a high Jaccard score of 0.6.

# Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please make sure to update tests as appropriate.