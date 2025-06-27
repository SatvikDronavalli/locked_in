import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("HC100_SelfPace.csv")
df = df[df["GeneralEvent"] != 'Standing']
idx = 0
for i in df["GeneralEvent"]:
    if i == 'Turn':
        break
    idx += 1
df = df.iloc[1:idx]
cases = []
while True:
    stepStarted = False
    bad_curve = False
    started_idx = 0
    ended_idx = 0
    for i in range(0, len(df["RTotalForce"])):
        if not stepStarted and df.iloc[i]["RTotalForce"] < 20 and df.iloc[i+1]["RTotalForce"] >= 20:
            stepStarted = True
            started_idx = i
            ended_idx = started_idx
        elif stepStarted and df.iloc[i]["RTotalForce"] < 20 and df.iloc[i+1]["RTotalForce"] < 20:
            break
        else:
            ended_idx += 1
    new_df = df.iloc[started_idx:ended_idx+2]
    bad_indexes = list(new_df[new_df["RTotalForce"].isna()].index)
    if not bad_indexes:
        cases.append((started_idx,ended_idx+2))
    plt.plot(new_df["RTotalForce"])
    plt.show()
    df = df.iloc[ended_idx+2:]
    if len(df) <= 100:
        break
print("completed extraction")

#df.to_csv("test.csv")
