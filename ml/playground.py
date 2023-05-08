import matplotlib.pyplot as plt
import numpy as np
import awswrangler as wr
import pandas as pd

import os
os.environ['AWS_PROFILE'] = "personal"


def read_and_parse_fitness_data(file):
    df_r = (pd
            .read_csv(file, parse_dates=["x"], dayfirst=True)
            .rename({" y": "fitness_score",
                   "x": "timestamp"}, axis=1)
            .astype({"fitness_score": float, "timestamp": "datetime64[ns]"})
            .assign(date=lambda x: x['timestamp'].dt.date.astype("str"),
                    source=lambda x: file.split("/")[-1])
            .sort_values("date")
          )
    return df_r


fitness_data_files = ["./data/20230101-20230331_fitness.csv",
                      "./data/20230401-20230507_fitness.csv"]

df_fitness = (pd
              .concat([read_and_parse_fitness_data(i) for i in fitness_data_files])
              .sort_values("date")
              .drop_duplicates(subset="date", keep="last")
              .reset_index(drop=True)
              )

df_activities = (wr
                 .athena
                 .read_sql_query("""SELECT * from strava.activities
                                    where start_timestamp >= date('2023-01-01')
                                    and start_timestamp <= date('2023-05-08')
                                    """,
                                 "strava")
                 .assign(date=lambda x: x['start_timestamp'].dt.date.astype("str"))  # Take date for the activity
                 .sort_values("date")
                 )

df_merged = (pd
             .merge(df_fitness, df_activities,
                     on="date",how="outer")
             .astype({"fitness_score": float, "date": "datetime64[ns]",
                      "id": float, "suffer_score": float})
             )


plt.plot(df_merged['date'].astype("datetime64"), df_merged['fitness_score'], label="Fitness date")
plt.scatter(df_merged['start_timestamp'].dt.date.astype("datetime64"),
            df_merged['fitness_score'], c='r',
            label="Act date")
plt.legend()
plt.xticks(rotation=90);


df_rest = (df_merged
          .get(['fitness_score','id', 'date'])
          .assign(days_diff=lambda x: (x['date'].shift(-1) - x['date']).dt.days,
                  fitness_score_final=lambda x: x['fitness_score'].shift(-1),
                  activity_in_between=lambda x: np.logical_not(x['id'].shift(-1).isna()))
          .query("activity_in_between == False")
          .dropna(subset=['days_diff', "fitness_score_final"])
          # .drop(["activity_in_between", "id", "date"], axis=1)
        )

plt.scatter(df_rest['days_diff'], df_rest['fitness_score_final']-df_rest['fitness_score'])
plt.xlim(0,10)
plt.ylim(-4,1)


def decay_model(ini, t, params):
    """Exponential decay base model"""
    b = params[0]
    c = params[1]
    return ini * np.exp(-b * t) + c


def exponential_decay_model(x, b, c):
    """Exponential decay model function to fit"""
    ini = x[:, 0]
    t = x[:, 1]
    return decay_model(ini, t, [b, c])


from scipy.optimize import curve_fit

x = np.c_[df_rest['fitness_score'], df_rest['days_diff']]
y = df_rest['fitness_score_final']

popt, pcov = curve_fit(exponential_decay_model, x, y)


t = np.arange(0,35)
plt.plot(t, decay_model(6.2, t, popt), '--', label='Fitted model')
plt.scatter([0, 34], [6.2, 2.3], label='real data', s=20, alpha=0.8, c='yellow')
t = np.arange(0,8)
plt.plot(t, decay_model(7.08, t, popt), '--', label='Fitted model')
plt.scatter([0, 7], [7.08, 5.93], label='real data', s=20, alpha=0.8, c='yellow')
plt.xlabel('Days')
plt.ylabel('Fitness score')
plt.legend()
plt.show()


plt.scatter(x[:,0], y, label='real data', s=50, alpha=1, c='yellow')
plt.plot(x[:,0], decay_model(x[:,0], x[:,1], popt), "s", label='Fitted model', alpha=0.8)
plt.xlabel('Initial fitness score')
plt.ylabel('Final fitness score')
plt.title("Prediction performance")
plt.legend()
plt.show()

df_plot = (df_merged
           .assign(fitness_score_initial=lambda x: x['fitness_score'].shift(1),
                   score_diff=lambda x: x['fitness_score']-x['fitness_score_initial'])
           # .dropna(subset=['suffer_score'])
           .get(['suffer_score', 'score_diff', 'fitness_score_initial', 'fitness_score', 'date'])
          )

plt.scatter(df_plot['suffer_score'], df_plot['score_diff'])
plt.show()


from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(),
                    SGDRegressor(loss='squared_error',
                     penalty='l2',
                     alpha=0.01,
                     max_iter=100,
                     tol=1e-3,
                     random_state=42,
                     eta0=0.1,
                     verbose=1)
                      )

PREDICTORS = ['suffer_score', 'fitness_score_initial']
TARGET = 'fitness_score'

df = (df_merged
      .assign(fitness_score_initial=lambda x: x['fitness_score'].shift(1),
              score_diff=lambda x: x['fitness_score']-x['fitness_score_initial'])
      .get(['suffer_score', 'score_diff', 'fitness_score_initial', 'fitness_score', 'date'])
      .dropna(subset=[TARGET, *PREDICTORS])
      )

X = df[PREDICTORS].values
y = df[TARGET].values

model.fit(X, y)

plt.scatter(y, model.predict(X))
plt.plot([0, 10], [0, 10], '--', c='r')
plt.xlabel('Real fitness score')
plt.ylabel('Predicted fitness score')
plt.title("Prediction performance")
plt.show()


