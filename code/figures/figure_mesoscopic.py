# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter
import pandas as pd
import os




# METHODS
def loadModelDetectorCounts(model_folder):   
    files = os.listdir(model_folder+"loops/")
    detector_counts = []
    for file in files:
        f = open(model_folder+"loops/"+file, "r")
        content = f.read()
        f.close()
        lines = [line.strip() for line in content.split("\n")]
        lines = [line for line in lines if line.startswith("<instantOut")]
        for line in lines:
            time = float(line.split("time=\"")[1].split("\"")[0])
            detector = line.split("id=\"")[1].split("\"")[0]
            detector_counts.append([time, detector])
    detector_counts = pd.DataFrame(detector_counts, columns=["time", "detector"])
    detector_counts['time_interval'] = (detector_counts['time'] // INTERVAL) * INTERVAL
    detector_counts = detector_counts.groupby(['time_interval', 'detector']).size().reset_index(name='count')
    return detector_counts

# My Measure - Every 300s MAE
def evaluateMAE(df_original, df_method, label):
    df_eval = df_original.merge(df_method, on=["time_interval", "detector"], how="left")
    df_eval['count_y'] = df_eval['count_y'].fillna(0)
    df_eval["mae"] = abs(df_eval["count_x"]-df_eval["count_y"])
    df_eval = df_eval[df_eval["time_interval"]<45900]
    average_mae = np.mean(df_eval["mae"])
    print(label, "average_mae", average_mae)
    return average_mae

def evaluateMAEC(df_original, df_method, label):
    df_eval = df_original.merge(df_method, on=["time_interval", "detector"], how="left")
    df_eval['count_y'] = df_eval['count_y'].fillna(0)
    df_eval = df_eval.groupby(['detector']).agg({
        'count_x': 'sum',
        'count_y': 'sum'
    }).reset_index()
    df_eval["mae"] = abs(df_eval["count_x"]-df_eval["count_y"])
    # df_eval = df_eval[df_eval["time_interval"]<45900]
    average_mae = np.mean(df_eval["mae"])
    print(label, "average_mae_c", average_mae)
    return average_mae

# GEH Measure from Sumo dfrouter paper
def evaluateGEH(df_original, df_method, label):
    detector_sums_1 = df_original.groupby('detector')['count'].sum().sort_values(ascending=False).reset_index()
    detector_sums_2 = df_method.groupby('detector')['count'].sum().sort_values(ascending=False).reset_index()
    df_eval = detector_sums_1.merge(detector_sums_2, on="detector", how="left")
    df_eval["geh"] = np.sqrt(2*(df_eval["count_x"]-df_eval["count_y"])*(df_eval["count_x"]-df_eval["count_y"])/(df_eval["count_x"]+df_eval["count_y"]))
    average_geh = np.mean(df_eval["geh"])
    print(label, "average_geh", average_geh)
    return average_geh




# MAIN ANALYSIS
INTERVAL = 300
df_original = loadModelDetectorCounts("../logs/original/")

df_dfrouter = loadModelDetectorCounts("../logs/logs_dfrouter/")
df_gravity  = loadModelDetectorCounts("../logs/run_gravity_model/")
df_frankwo  = loadModelDetectorCounts("../logs/run_frank_wolfe/")
df_entropy  = loadModelDetectorCounts("../logs/run_entropy_maxim/")
df_triprec  = loadModelDetectorCounts("../logs/logs_trip_reconstruction/")
df_triprecP = loadModelDetectorCounts("../logs/logs_trip_reconstruction_perfect/")

print("MAE")
evaluateMAE(df_original, df_dfrouter, "Dfrouter")
evaluateMAE(df_original, df_gravity, "Gravity")
evaluateMAE(df_original, df_frankwo, "FrankWolfe")
evaluateMAE(df_original, df_entropy, "Entropy")
evaluateMAE(df_original, df_triprec, "TripReconstruction")
evaluateMAE(df_original, df_triprecP, "TripReconstructionPerf")

print("\nMAE_c")
evaluateMAEC(df_original, df_dfrouter, "Dfrouter")
evaluateMAEC(df_original, df_gravity, "Gravity")
evaluateMAEC(df_original, df_frankwo, "FrankWolfe")
evaluateMAEC(df_original, df_entropy, "Entropy")
evaluateMAEC(df_original, df_triprec, "TripReconstruction")
evaluateMAEC(df_original, df_triprecP, "TripReconstructionPerf")

print("\nGEH")
evaluateGEH(df_original, df_dfrouter, "Dfrouter")
evaluateGEH(df_original, df_gravity, "Gravity")
evaluateGEH(df_original, df_frankwo, "FrankWolfe")
evaluateGEH(df_original, df_entropy, "Entropy")
evaluateGEH(df_original, df_triprec, "TripReconstruction")
evaluateGEH(df_original, df_triprecP, "TripReconstructionPerf")

df_tot_stat = df_original.copy()
df_tot_stat = df_tot_stat.groupby(['detector']).agg({
        'count': 'sum',
    }).reset_index()
n_intevals = 12*13+9
print("\nAverage Detector Count", np.mean(df_tot_stat["count"]), np.std(df_tot_stat["count"]))
print("\nAverage Detector Count", np.mean(df_tot_stat["count"])/n_intevals, np.std(df_tot_stat["count"])/n_intevals)




plt.figure(figsize=(12,2.5))  

plt.suptitle("(B) Mesoscopic Assessment", fontweight="bold", y=0.97)

# Define a color map for each model
color_map = {
    'dfrouter': 'lightsalmon',
    'gravity': 'lightpink',
    'frank-wolfe': 'wheat',
    'entropy-maxim': 'lightgreen',
    'triprec': 'lightblue',
    'triprec*': 'lavender'
}

plt.subplot(1,3,1)
all_errors = []
avg_errors = []
for df_method in [df_dfrouter, df_triprec, df_gravity, df_frankwo, df_entropy, df_triprecP]:
    df_eval = df_original.merge(df_method, on=["time_interval", "detector"], how="left")
    df_eval['count_y'] = df_eval['count_y'].fillna(0)
    df_eval["mae"] = abs(df_eval["count_x"]-df_eval["count_y"])
    df_eval = df_eval[df_eval["time_interval"]<45900]
    average_mae = np.mean(df_eval["mae"])
    all_errors.append(df_eval["mae"].tolist())
    avg_errors.append(average_mae)
new_labels = [f"{label} [{sum_value:.2f}]" for label, sum_value in zip(list(color_map.keys()), avg_errors)]
bp = plt.gca().boxplot(all_errors, vert=False, patch_artist=True)
models = list(color_map.keys())
for box, median, ctr in zip(bp['boxes'], bp['medians'], range(0,len(bp['boxes']))):
    box.set(facecolor=color_map[models[ctr]])
    median.set(color='black', linewidth=1.5)
plt.gca().set_yticklabels(new_labels)
plt.yticks(ha='right')
plt.xlabel("Absolute Count Error (Interval)", fontweight="bold")
plt.xlim(0,400)

fig = plt.gcf()
pos1 = plt.gca().get_position()

plt.subplot(1,3,2)
all_errors = []
avg_errors = []
for df_method in [df_dfrouter, df_triprec, df_gravity, df_frankwo, df_entropy, df_triprecP]:
    df_eval = df_original.merge(df_method, on=["time_interval", "detector"], how="left")
    df_eval['count_y'] = df_eval['count_y'].fillna(0)
    df_eval = df_eval[df_eval["time_interval"]<45900]
    df_eval = df_eval.groupby(['detector']).agg({'count_x': 'sum', 'count_y': 'sum'}).reset_index()
    df_eval["mae"] = abs(df_eval["count_x"]-df_eval["count_y"])
    average_mae = np.mean(df_eval["mae"])
    all_errors.append(df_eval["mae"].tolist())
    avg_errors.append(average_mae)
new_labels = [f"[{sum_value:.2f}]" for sum_value in avg_errors]
bp = plt.gca().boxplot(all_errors, vert=False, patch_artist=True)
models = list(color_map.keys())
for box, median, ctr in zip(bp['boxes'], bp['medians'], range(0,len(bp['boxes']))):
    box.set(facecolor=color_map[models[ctr]])
    median.set(color='black', linewidth=1.5)
plt.gca().set_yticklabels(new_labels)
plt.yticks(ha='right')
plt.xlabel("Absolute Count Error (Total)", fontweight="bold")
# plt.xlim(0,400)
plt.gca().xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.gca().xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
plt.gca().xaxis.set_major_locator(plt.MaxNLocator(2))

plt.subplot(1,3,3)
all_errors = []
avg_errors = []
for df_method in [df_dfrouter, df_triprec, df_gravity, df_entropy, df_frankwo, df_triprecP]:
    detector_sums_1 = df_original.groupby('detector')['count'].sum().sort_values(ascending=False).reset_index()
    detector_sums_2 = df_method.groupby('detector')['count'].sum().sort_values(ascending=False).reset_index()
    df_eval = detector_sums_1.merge(detector_sums_2, on="detector", how="left")
    df_eval["geh"] = np.sqrt(2*(df_eval["count_x"]-df_eval["count_y"])*(df_eval["count_x"]-df_eval["count_y"])/(df_eval["count_x"]+df_eval["count_y"]))
    average_geh = np.mean(df_eval["geh"])
    all_errors.append(df_eval["geh"].tolist())
    avg_errors.append(average_geh)
new_labels = [f"[{sum_value:.2f}]" for sum_value in avg_errors]
bp = plt.gca().boxplot(all_errors, vert=False, patch_artist=True)
models = list(color_map.keys())
for box, median, ctr in zip(bp['boxes'], bp['medians'], range(0,len(bp['boxes']))):
    box.set(facecolor=color_map[models[ctr]])
    median.set(color='black', linewidth=1.5)
plt.gca().set_yticklabels(new_labels)
plt.yticks(ha='right')
plt.xlabel("Geoffrey-E-Havers Score (GEH)", fontweight="bold")
plt.xlim(0, 110)

text_x = pos1.x0 + 0.03
text_y = pos1.y0 - 0.02
fig.text(text_x, text_y, "Detector-Averaged", rotation=0, va='center', ha='right', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 1.05])  
plt.show()