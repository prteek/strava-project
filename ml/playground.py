import matplotlib.pyplot as plt
import numpy as np
import awswrangler as wr


df_activities = wr.athena.read_sql_query("SELECT * FROM activities", database="strava")

df_activities_ = (
    df_activities.astype({"moving_time": float})
    .dropna(subset=["average_heartrate", "moving_time", "suffer_score"])
    .assign(moving_time_mins=lambda x: x.moving_time / 60)
)

marker_dict = {"Workout": "s", "Run": "o", "WeightTraining": "+", "Walk": "x"}
for marker, d in df_activities_.groupby("type"):
    plt.scatter(
        d["average_heartrate"],
        np.log(d["suffer_score"] + 0.5),
        c=d["moving_time_mins"],
        marker=marker_dict[marker],
        label=marker,
    )
    plt.legend()


plt.colorbar()
plt.title("Avg. Heart rate vs Log transformed suffer score")
plt.xlabel("Avg. Heart rate")
plt.ylabel("Log transformed suffer score")
plt.grid()
