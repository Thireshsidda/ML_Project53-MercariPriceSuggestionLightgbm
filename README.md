# ML_Project53-MercariPriceSuggestionLightgbm

# Mercari Price Suggestion with LightGBM

## Overview
This project focuses on predicting the prices of items listed on the Mercari e-commerce platform using the LightGBM gradient boosting framework. By analyzing various features such as item name, category, brand, item condition, and shipping information, the goal is to build a regression model that accurately predicts item prices.

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- scipy
- scikit-learn
- lightgbm

## The Data
The dataset consists of item listings from Mercari, split into training and test sets. The training data is primarily used for exploratory data analysis (EDA) and model training, while the test data is used for evaluation.

## Exploratory Data Analysis (EDA)
- Explore the distribution of item prices and apply log-transformation for better visualization.
- Analyze the relationship between shipping type and item price.
- Investigate the distribution of item categories and brands.
- Examine the impact of item condition on item price.

## LightGBM Model
### Data Preprocessing
- Handle missing values in the dataset by imputing or replacing them.
- Convert categorical features (category name, brand name, item condition) into numerical representations.
- Vectorize text data (item name, item description) using techniques like CountVectorizer and TF-IDF Vectorizer.

### Feature Engineering
- Create dummy variables for item condition and shipping columns.
- Concatenate sparse matrices representing different features into a single sparse merge matrix.
- Remove features with low document frequency to reduce dimensionality.

### Model Training
- Define the LightGBM model parameters such as learning rate, maximum depth, and number of leaves.
- Train the LightGBM model using the training data with the specified parameters.
- Tune hyperparameters and optimize the model performance.

### Model Evaluation
- Make predictions on the test data using the trained model.
- Evaluate the model's performance using root mean squared error (RMSE) metric.
- Analyze the prediction results and identify areas for improvement.

## Results
The trained LightGBM model achieves an RMSE of approximately 0.46 on the test data, indicating good performance in predicting item prices.

## Usage
1. Ensure that the dataset file 'train.tsv' is available.
2. Run the provided code in a Python environment with the required dependencies installed.
3. Analyze the EDA visualizations to gain insights into the dataset.
4. Train the LightGBM model using the provided code.
5. Evaluate the model's performance using the RMSE metric on the test data.
6. Experiment with different preprocessing techniques and model parameters to improve performance if needed.

## Future Improvements
- Experiment with additional feature engineering techniques such as word embeddings and text summarization.
- Explore ensemble methods and model stacking for further performance improvement.
- Incorporate external data sources or additional features to enhance model accuracy.

## Acknowledgments
- Acknowledge any additional resources, libraries, or datasets used in the project.
