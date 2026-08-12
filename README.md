# us-accident-severity-prediction

Predicting the severity of traffic accidents in the United States using supervised machine learning algorithms. (my Data-Mining course project)

## Step 1: Understanding the Dataset

The original dataset was provided as `group_31_train.csv`. The feature columns did not have meaningful names, so the data was mostly unknown and unclear at the beginning.

Our first goal was to explore the dataset and understand its general structure, feature types, and possible meaning. This process was done in:

`01_data_understanding.ipynb`

After completing the analyses and investigations described in the notebook, we gained a better understanding of the dataset. We renamed the columns whose meanings could be identified and created a new file named `initial_data_renamed.csv`.

In the next step, we will continue working with this renamed dataset.

## Step 2: Data Processing

The data cleaning and preprocessing steps were performed in:

`02_data_processing.ipynb`

At the beginning of this step, the dataset was divided into **train, validation, and test sets** while preserving the target distribution. This was done before the main preprocessing steps to prevent data leakage. All statistics required for validation and test transformations were learned only from the training data.

Missing values were handled based on the characteristics of each feature. For environmental and wind-related features, location-based hierarchical imputation using city, state, and global training statistics was applied.

Some features were also transformed or replaced with more useful representations. For example, `Lat` and `Lng` were replaced by `Location_Available`, wind direction was converted into cyclical features, and `Weather_Time` was replaced by `Record_Delay_Time`.

Noise and outliers were handled using geographical limits and IQR-based boundaries calculated from the training data. Categorical, Boolean, textual, and numerical features were then encoded or standardized where necessary.

Finally, duplicated records within and across the train, validation, and test sets were removed.

The processed datasets were saved as:

* `train_processed1.csv`
* `validation_processed1.csv`
* `test_processed1.csv`

## Step 3: Feature Engineering and Feature Selection

Feature engineering and final feature selection were performed in:

`03_feature_engineering.ipynb`

One important constraint of this project was that the final dataset could contain **at most 25 features**. Therefore, feature engineering was designed to preserve useful information while keeping the number of features limited.

`Accident_Time` was replaced with a binary `is_holiday` feature that represents weekends and U.S. holidays.

For the `Description` column, important severity-related words were identified from the training data. Instead of using a high-dimensional text representation, the descriptions were compressed into a single binary feature called `has_sensitive_word`.

The location columns `Street`, `City`, `County`, and `State` contained too many unique values for One-Hot Encoding. They were therefore compressed into two numerical features:

* `Address_Freq`: represents how frequently a location appears.
* `Address_Mean`: represents accident severity patterns using hierarchical smoothed target encoding.

Out-of-Fold Target Encoding was used for the training data to prevent target leakage, while validation and test statistics were calculated only from the training set.

Low-information features, where more than `99.5%` of the values were the same, were removed. Highly redundant features were also examined using correlation analysis, which resulted in removing `Location_Available` because of its strong relationship with `Source`.

After feature engineering and selection, the dataset was reduced to the required **25 final features** and prepared for modeling.

## Step 4: Modeling and Model Selection

Model training, tuning, comparison, and final evaluation were performed in:

`04_Modeling.ipynb`

Because the target classes were highly imbalanced, a `most_frequent` Dummy Classifier was first used as a baseline to show why Accuracy alone was not a sufficient evaluation metric.

Three main supervised learning algorithms were then investigated:

* Multinomial Logistic Regression
* Random Forest
* XGBoost

For each model family, different versions such as baseline, class-balanced/weighted, and hyperparameter-tuned models were evaluated on the validation set.

The main metric used for model comparison was **Macro F1**, because it gives equal importance to all severity classes and is more suitable for this imbalanced classification problem. Weighted F1, Balanced Accuracy, Accuracy, and class-level Precision and Recall were also considered.

After comparing all experiments, the **Tuned XGBoost** model with balanced sample weights was selected as the final model.

The selected model was then retrained using the combined **train + validation datasets**, and the untouched test set was used only for the final evaluation.

### Final Test Results

| Metric            |  Score |
| ----------------- | -----: |
| Macro F1          | 0.5943 |
| Weighted F1       | 0.8439 |
| Balanced Accuracy | 0.6558 |
| Accuracy          | 0.8352 |

The final classification results showed that although the dataset was strongly imbalanced, the tuned and weighted XGBoost model provided a considerably better balance between the majority and minority severity classes compared with the simpler baseline approaches.
