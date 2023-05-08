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

df_merged = pd.merge(df_fitness, df_activities,
                     on="date",how="left")


plt.plot(df_merged['date'].astype("datetime64"), df_merged['fitness_score'], label="Fitness date")
plt.scatter(df_merged['start_timestamp'].dt.date.astype("datetime64"),
            df_merged['fitness_score'], c='r',
            label="Act date")
plt.legend()
plt.xticks(rotation=90);

