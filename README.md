# YouTube Channel Analysis and Growth Prediction Project
This repository contains the code and data used for a comprehensive analysis of the top 1000 YouTube channels and the development of a machine learning model to predict a channel's growth based on certain features. The project is split into two main parts: data visualization and exploration, and building a machine learning model.

## Repository Structure
The repository is organized into the following directories and files:

- analysis: This folder contains the two Jupyter notebooks used in the project.
  - **youtube_visualization.ipynb:** Data visualization and exploration of the top 1000 YouTube channels.
  - **youtube_models.ipynb:** Building a machine learning model to predict subscriber growth based on video views, video count, and channel age.
 
- data: This folder contains the dataset used in the project, topSubscribed.csv.

## Project Overview
### Data Visualization and Exploration
The first part of the project focuses on exploring the dataset, which includes the following columns:
- Rank
- YouTube Channel
- Subscribers
- Video Views
- Video Count
- Category
- Started (year the channel was created)

The analysis of the data led to several findings, such as:

1. "Music" and "Entertainment" videos receive the most views, making them attractive categories for new content creators.
2. Older channels are more likely to have more combined video views.
3. There is a positive correlation between the number of videos, views, and subscribers. This is supported by the Rank column, which is based on subscriber count.
4. Fewer top YouTube channels are being created each year, according to the Top 1000 dataset.

### Machine Learning Model
The second part of the project involves building a neural network regression model to predict subscriber growth based on video views, video count, and channel age. The model was built using TensorFlow and Keras, and it was trained on a dataset that was preprocessed and scaled. 

The neural network consists of 3 input neurons, 2 hidden layers, comprised of 64 and 32 neurons respectively, and 1 output neuron. 
The hidden layers use a Rectified Linear Unit activation function, and the model was trained using Adaptive Moment Estimation

The model achieved a mean squared error (MSE) of 45.37.

## Usage
To use the Jupyter notebooks in this project, you can open them in Google Colab or another Jupyter notebook-compatible environment.

Clone the repository to your local machine or open the repository on GitHub.
Navigate to the analysis folder and open the desired notebook (e.g., youtube_visualization.ipynb or youtube_models.ipynb).
Run the cells in the notebook to visualize the data or train the machine learning model.
Please note that the dataset used in this project only represents the top 1000 channels on YouTube and may not accurately reflect all YouTube channels.

## Dependencies
The following libraries were used in this project:
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- tensorflow
