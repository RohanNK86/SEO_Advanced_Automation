import pandas as pd
import matplotlib.pyplot as pt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('./Datasets/content_refresh_anonymized (1).csv')

num_cols = [
    'search_volume', 'competition', 'cpc',
    'word_count', 'char_count', 'trend_pct',
    'ctr_gap_score', 'volume_score', 'score'
]

# Impute missing values with the column mean using Pandas .mean()
df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

# For categorical columns like 'competition_level', fill missing values with the mode (most frequent value)
df['competition_level'] = df['competition_level'].fillna(df['competition_level'].mode()[0])


X = df[['search_volume', 'competition', 'cpc', 'word_count', 'char_count', 'trend_pct', 'staleness_score', 'ctr_gap_score', 'volume_score']]
y = df['score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
lm = LinearRegression()
lm.fit(X_train, y_train)

#Preditions
pred = lm.predict(X_test)

pt.scatter(y_test, pred)
pt.plot(y_test, y_test, 'r')
pt.xlabel('Actual Values')
pt.ylabel('predicted Values')
pt.show()

#Errors Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
print('MAE : ',mean_absolute_error(y_test, pred))
print('MSE : ',mean_squared_error(y_test, pred))