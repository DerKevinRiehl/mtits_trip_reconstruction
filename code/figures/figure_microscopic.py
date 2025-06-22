# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import entropy




# METHODS
def loadTripInfos(model_folder):
    f = open(model_folder+"TripInfos.xml", "r")
    content = f.read()
    f.close()
    lines = [line.strip() for line in content.split("\n")]
    lines = [line for line in lines if line.startswith("<tripinfo ")]
    data = []
    for line in lines:
        timeLoss = float(line.split("timeLoss=\"")[1].split("\"")[0])
        distance = float(line.split("routeLength=\"")[1].split("\"")[0])
        waitTime = float(line.split("waitingTime=\"")[1].split("\"")[0])
        duration = float(line.split("duration=\"")[1].split("\"")[0])
        data.append([timeLoss, distance, waitTime, duration])
    data = pd.DataFrame(data, columns=["delay", "distance", "waitTime", "totTravelTime"])
    return data

def getCumulativeDistribution(data, metric, binZ, norm=None):
    bins = np.arange(binZ[0], binZ[1], binZ[2])
    delay_binned = pd.cut(data[metric], bins=bins, include_lowest=True, right=False)
    delay_distribution = delay_binned.value_counts().sort_index()
    cumulative_distribution = delay_distribution.cumsum()
    if norm is not None:
        cumulative_distribution = cumulative_distribution/norm.iloc[-1]
    return cumulative_distribution

def evaluateDistributionError(data_original, data_method, metric):
    if metric=="delay":
        binZ = [1, 200, 2]
    elif metric=="distance":
        binZ = [70, 400, 20]
    elif metric=="waitTime":
        binZ = [0, 150, 5]
    else:
        binZ = [0, 250, 10]
    cdf_distribution_original  = getCumulativeDistribution(data_original, metric, binZ)
    cdf_distribution_method    = getCumulativeDistribution(data_method,  metric, binZ, norm=cdf_distribution_original)
    cdf_distribution_original /= cdf_distribution_original.iloc[-1]
    # Calculate MAE
    mae_error = abs(cdf_distribution_original - cdf_distribution_method)
    # Calculate summed error
    summed = sum(abs(cdf_distribution_original - cdf_distribution_method)) / len(cdf_distribution_original)*100
    # Calculate KL divergence
    pdf_original = np.diff(cdf_distribution_original, prepend=0)
    pdf_method = np.diff(cdf_distribution_method, prepend=0)
    epsilon = 1e-10 # Add a small epsilon to avoid division by zero
    pdf_original = pdf_original + epsilon
    pdf_method = pdf_method + epsilon
    pdf_original = pdf_original / np.sum(pdf_original)
    pdf_method = pdf_method / np.sum(pdf_method)
    kl_divergence = entropy(pdf_original, pdf_method)
    return mae_error, summed, kl_divergence




# MAIN ANALYSIS
INTERVAL = 300

data_original = loadTripInfos("../logs/original/")
data_dfroute  = loadTripInfos("../logs/logs_dfrouter/")
data_gravmod  = loadTripInfos("../logs/run_gravity_model/")
data_frankwo  = loadTripInfos("../logs/run_frank_wolfe/")
data_entrmax  = loadTripInfos("../logs/run_entropy_maxim/")
data_triprec  = loadTripInfos("../logs/logs_trip_reconstruction/")
data_perfect  = loadTripInfos("../logs/logs_trip_reconstruction_perfect/")

model_labels = ["dfrouter", "gravity", "frank-wolfe", "entropy", "triprec", "perfect",]

def calculateErrors(metric):
    error_dfroute, sum_dfroute, kdl_dfroute = evaluateDistributionError(data_original, data_dfroute, metric)
    error_gravmod, sum_gravmod, kdl_gravmod = evaluateDistributionError(data_original, data_gravmod, metric)
    error_frankwo, sum_frankwo, kdl_frankwo = evaluateDistributionError(data_original, data_frankwo, metric)
    error_entrmax, sum_entrmax, kdl_entrmax = evaluateDistributionError(data_original, data_entrmax, metric)
    error_triprec, sum_triprec, kdl_triprec = evaluateDistributionError(data_original, data_triprec, metric)
    error_perfect, sum_perfect, kdl_perfect = evaluateDistributionError(data_original, data_perfect, metric)
    all_errors = [error_dfroute, error_gravmod, error_frankwo, error_entrmax, error_triprec, error_perfect]
    all_sums = [sum_dfroute, sum_gravmod, sum_frankwo, sum_entrmax, sum_triprec, sum_perfect]
    all_kdl = [kdl_dfroute, kdl_gravmod, kdl_frankwo, kdl_entrmax, kdl_triprec, kdl_perfect]
    print(metric, all_kdl)
    return all_errors, all_sums, all_kdl



plt.figure(figsize=(12, 2.5))

plt.suptitle("(C) Microscopic Assessment", fontweight="bold", y=0.97)

# Define a color map for each model
color_map = {
    'dfrouter': 'lightsalmon',
    'gravity': 'lightpink',
    'frank-wolfe': 'wheat',
    'entropy-maxim': 'lightgreen',
    'triprec': 'lightblue',
    'triprec*': 'lavender'
}

metrics = ["distance", "totTravelTime", "delay", "waitTime"]
titles = ["Distance", "Travel Time", "Delay Time", "Waiting Time"]
# Calculate overall min and max
all_errors_combined = []
for metric in metrics:
    all_errors, _, _ = calculateErrors(metric)
    all_errors_combined.extend(all_errors)

xmin = np.min([np.min(errors) for errors in all_errors_combined])
xmax = np.max([np.max(errors) for errors in all_errors_combined])
fig = plt.gcf()

for i, metric in enumerate(metrics, 1):
    plt.subplot(1, 4, i)
    all_errors, all_sums, _ = calculateErrors(metric)
    if i==1:
        new_labels = [f"{label} [{sum_value:.2f}]" for label, sum_value in zip(list(color_map.keys()), all_sums)]
    else:
        new_labels = [f"[{sum_value:.2f}]" for sum_value in all_sums]
    bp = plt.gca().boxplot(all_errors, vert=False, patch_artist=True)
    models = list(color_map.keys())
    for box, median, ctr in zip(bp['boxes'], bp['medians'], range(0,len(bp['boxes']))):
        box.set(facecolor=color_map[models[ctr]])
        median.set(color='black', linewidth=1.5)
    plt.gca().set_yticklabels(new_labels)
    plt.yticks(ha='right')
    plt.xscale("log")
    plt.xlim(xmin, xmax)
    plt.xlabel(titles[i-1], fontweight="bold")
    if i==1:
        pos1 = plt.gca().get_position()
        
text_x = pos1.x0 + 0.03  
text_y = pos1.y0 - 0.02  
fig.text(text_x, text_y, "Integrated Absolute\nDistribution Error [%]", rotation=0, va='center', ha='right', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 1.05]) 
plt.show()




"""
labels = delay_distribution_original.index.tolist()
labels = [str(s).replace("Interval(", "").replace(", closed='left')", "") for s in labels]

plt.plot(labels, delay_distribution_original, label="original")
plt.plot(labels, delay_distribution_perfect, label="perfect")
plt.plot(labels, delay_distribution_gravmod, label="gravity model")
plt.plot(labels, delay_distribution_entrmax, label="entropy maximization")
plt.plot(labels, delay_distribution_frankwo, label="frank wolfe")
plt.plot(labels, delay_distribution_triprec, label="trip reconstruction")
plt.legend()
"""



