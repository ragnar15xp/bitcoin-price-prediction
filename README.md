======================================================================
README - Bitcoin Price Prediction using Machine Learning
======================================================================

Project Description:
---------------------
This project demonstrates how machine learning can be used to predict whether buying Bitcoin at the end of the day will result in profit the next day. We use historical OHLC (Open, High, Low, Close) price data over an 8-year period and apply several ML models to predict price movement signals.

The goal is to train models that classify whether the next day's closing price will be higher than today's, providing a BUY or HOLD signal.

======================================================================

Project Features:
------------------
✅ Uses 8 years of historical Bitcoin data (2014–2022)  
✅ Performs Exploratory Data Analysis (EDA)  
✅ Conducts feature engineering for meaningful inputs  
✅ Applies multiple machine learning models:
   - Logistic Regression
   - Support Vector Classifier (SVC) with polynomial kernel
   - XGBoost Classifier

✅ Evaluates models using ROC-AUC scores  
✅ Visualizes results using plots (line plots, distributions, boxplots, heatmaps, confusion matrix)

======================================================================

Requirements:
--------------
Python 3.x

Required Python libraries:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost

You can install them using:
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost

======================================================================

Dataset:
---------
The dataset used is `bitcoin.csv`  
Columns:
- Date (string, format YYYY-MM-DD)
- Open
- High
- Low
- Close
- Adj Close (identical to Close, dropped in preprocessing)
- Volume

After loading, we:
- Drop 'Adj Close' (redundant)
- Convert 'Date' to datetime format
- Create new features:
   - 'year', 'month', 'day' from Date
   - 'open-close' (Open - Close)
   - 'low-high' (Low - High)
   - 'is_quarter_end' (1 if end of quarter, else 0)
- Define the target:
   - 1 if next day's Close > today's Close (BUY signal)
   - 0 if next day's Close ≤ today's Close (HOLD signal)

======================================================================

Exploratory Data Analysis (EDA):
---------------------------------
We use various plots to understand the data:
- Line plot of Bitcoin's closing price over time
- Distribution plots of OHLC data to observe spread
- Boxplots to detect outliers
- Barplots of mean yearly prices
- Heatmap of feature correlations
- Pie chart of target class balance

These visualizations help understand the data trends and variability.

======================================================================

Modeling Approach:
--------------------
Selected Features:
- open-close
- low-high
- is_quarter_end

Steps:
1. Normalize features using StandardScaler
2. Split data into:
   - Training (first ~70%)
   - Validation (last ~30%)

Models Trained:
1. Logistic Regression
2. Support Vector Classifier (SVC) with polynomial kernel
3. XGBoost Classifier

Evaluation Metric:
- ROC-AUC Score (measures model’s ability to assign correct probabilities)

======================================================================

Model Results:
---------------
| Model                   | Training ROC-AUC | Validation ROC-AUC |
|-------------------------|------------------|--------------------|
| Logistic Regression     | ~0.52            | ~0.52              |
| SVC (poly kernel)       | ~0.48            | ~0.53              |
| XGBoost Classifier      | ~0.92            | ~0.46              |

Observations:
- Logistic Regression shows balanced but low performance (~50%).
- XGBoost shows overfitting: high training accuracy but poor validation performance.
- Overall, simple models on this dataset do not outperform random guessing.

======================================================================

Confusion Matrix:
------------------
We visualize the confusion matrix to understand:
- True Positives (correct BUY predictions)
- True Negatives (correct HOLD predictions)
- False Positives / False Negatives (misclassifications)

======================================================================

How to Run:
------------
1. Clone or download this repository:
    git clone https://github.com/yourusername/bitcoin-price-prediction.git
    cd bitcoin-price-prediction

2. Install required packages:
    pip install -r requirements.txt

3. Place the `bitcoin.csv` dataset in the project folder.

4. Run the script:
    python bitcoin_prediction.py

5. Check the terminal for printed evaluation metrics and view the generated plots.

======================================================================

Folder Structure:
------------------
/bitcoin-price-prediction
    ├── bitcoin.csv              → Dataset file
    ├── bitcoin_prediction.py    → Python script with code
    ├── README.txt               → This README file
    └── requirements.txt         → Required Python packages

======================================================================

Possible Improvements:
-----------------------
- Include more technical indicators (e.g., RSI, MACD, moving averages)
- Incorporate market sentiment data (e.g., Twitter, news headlines)
- Use deep learning models (LSTM, GRU) for time-series forecasting
- Hyperparameter tuning for better model performance
- Evaluate using additional metrics (precision, recall, F1-score)

======================================================================

License:
---------
This project is licensed under the MIT License.
You are free to use, modify, and share the code with attribution.

======================================================================

Acknowledgments:
-----------------
Original code adapted and modified by Susobhan Akhuli.

======================================================================

Contact:
---------
For any questions or contributions, feel free to reach out!

