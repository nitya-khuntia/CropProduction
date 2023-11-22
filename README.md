# Crop Production Prediction Project

Welcome to the Crop Production Prediction Project! This repository houses a machine learning model and a web application that together offer insights into crop production in India based on historical data. The primary goal of this project is to serve as an educational resource for learning about data science and machine learning applications in agriculture.

## Overview

This project utilizes a dataset containing historical crop production data from India. A machine learning model has been trained to predict the production of various crops considering factors such as state, district, crop type, season, and area sown. These predictions are showcased through a Flask web application, demonstrating how machine learning can be applied in agricultural contexts.

## Features

- Predictive model trained on historical crop production data from India.
- Flask web application for demonstrating model predictions.
- Data preprocessing and encoding to effectively handle categorical variables.
- Analysis of crop production trends to understand agricultural patterns.

## Project Structure

- `crop-production.ipynb`: Jupyter notebook for data preprocessing and model training.
- `app.py`: Flask application for demonstrating the model's predictions.
- `crop_prediction_frontend.html`: Frontend HTML file for interacting with the model.
- `columns.pkl` and `encoder.pkl`: Serialized files for data preprocessing in the Flask app.
- `crop_production.csv`: Dataset used for model training.

## Getting Started

To explore this project:

1. Clone the repository to your local machine.
2. Ensure you have Python installed, along with libraries such as Flask, Pandas, and scikit-learn.
3. Run the Flask application (`app.py`) to start the web service.
4. Open `crop_prediction_frontend.html` in a web browser to interact with the model and see predictions.

## Usage

Input details about the crop, including state, district, season, crop type, and area, into the web interface. The model will then provide a production prediction based on these parameters.

## Contributing

While this project is primarily educational, contributions are still welcome. You can contribute in different ways:

- Enhancing the model's accuracy and performance.
- Extending the dataset with more diverse data for broader learning.
- Improving the frontend for a more interactive learning experience.

For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgments

- Data source: [(https://www.kaggle.com/datasets/abhinand05/crop-production-in-india)]
- This project was inspired by the potential applications of data science in the field of agriculture, particularly in understanding and predicting crop production patterns.

 
