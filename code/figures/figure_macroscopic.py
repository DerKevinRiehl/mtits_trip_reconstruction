# IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import pandas as pd




# METHODS
def process_summary(file_path, interval=300):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    last_ended = 0
    for stepinfo in root.findall('step'):
        av_traveltime =  float(stepinfo.get('meanTravelTime'))
        n_cars = float(stepinfo.get('running'))
        n_ended =  float(stepinfo.get('ended'))
        n_flow = (n_ended-last_ended)/interval*3600
        ttt = n_cars * av_traveltime
        av_speed = float(stepinfo.get('meanSpeed'))*3.6
        time = float(stepinfo.get('time'))
        if time>45900:
            continue
        if av_speed != -1:
            data.append([time, n_cars, ttt, av_speed, n_flow])
        last_ended = n_ended
    return data

def loadMethodMFD(file, INTERVAL):
    mfd = process_summary(file, INTERVAL)    
    mfd = np.asarray(mfd)
    if mfd[0][0]==0:
        mfd = mfd[1:]
    return mfd

def loadSeededDate(file, INTERVAL):
    lst_density = []
    lst_flow = []
    lst_speed = []
    for seed in [1,2,3,4,5,6,7,8,9,10]:
        # mdf loading
        mfd_method = loadMethodMFD(file, INTERVAL)    
        # calculating of mae
        mae_density = np.abs(mfd_original[:,1]-mfd_method[:,1])
        mae_flow    = np.abs(mfd_original[:,4]-mfd_method[:,4])
        mae_speed   = np.abs(mfd_original[:,3]-mfd_method[:,3])
        lst_density += mae_density.tolist()
        lst_flow    += mae_flow.tolist()
        lst_speed   += mae_speed.tolist()
    return lst_density, lst_flow, lst_speed

def loadAllSeeds(file, INTERVAL):
    lst_density = []
    lst_flow = []
    lst_speed = []
    for seed in [1,2,3,4,5,6,7,8,9,10]:
        a, b, c = loadSeededDate(file+'_'+str(seed)+'/Log_summary.xml', INTERVAL)
        lst_density += a
        lst_flow    += b
        lst_speed   += c
    return [np.median(lst_density), np.std(lst_density), np.median(lst_flow), 
            np.std(lst_flow), np.median(lst_speed), np.std(lst_speed)]




# MAIN ANALYSIS
INTERVAL = 300
mfd_original = loadMethodMFD('../logs/original/Log_summary.xml', INTERVAL)    

# LOAD BENCHMARK
data = []
for method in ["entropy_maxim", "frank_wolfe", "gravity_model"]:
    for od_time in ["15m", "30m", "60m"]:
        record = loadAllSeeds('../logs/many_seed_logs/logs_run_'+method+'_'+od_time, INTERVAL)
        data.append([method, od_time, *record])
record = loadAllSeeds('../logs/many_seed_logs/logs_dfrouter', INTERVAL)
data.append(["dfrouter", "", *record])

# LOAD METHOD
mfd_method = loadMethodMFD("../logs/logs_trip_reconstruction/Log_summary.xml", INTERVAL)
lst_density = np.abs(mfd_original[:,1]-mfd_method[:,1]).tolist()
lst_flow = np.abs(mfd_original[:,4]-mfd_method[:,4]).tolist()
lst_speed = np.abs(mfd_original[:,3]-mfd_method[:,3]).tolist()
data.append(["trip_reconstruction", "15m",  
             np.median(lst_density), np.std(lst_density), 
             np.median(lst_flow), np.std(lst_flow), 
             np.median(lst_speed), np.std(lst_speed)])

# LOAD THEORETICAL PERFECT
mfd_method = loadMethodMFD("../logs/logs_trip_reconstruction_perfect/Log_summary.xml", INTERVAL)
lst_density = np.abs(mfd_original[:,1]-mfd_method[:,1]).tolist()
lst_flow = np.abs(mfd_original[:,4]-mfd_method[:,4]).tolist()
lst_speed = np.abs(mfd_original[:,3]-mfd_method[:,3]).tolist()
data.append(["trip_reconstructiontP", "15m",  
             np.median(lst_density), np.std(lst_density), 
             np.median(lst_flow), np.std(lst_flow), 
             np.median(lst_speed), np.std(lst_speed)])

# PREPARE TABULAR DATASET
data = pd.DataFrame(data, columns=["Method", "OD_Time", "MAE_density", "std1", "MAE_flow", "std2", "MAE_speed", "std3"])


# PLOT
mfd_original = loadMethodMFD('../logs/original/Log_summary.xml', INTERVAL) 

benchmark_methods = {
    "dfrouter": '../logs/logs_dfrouter/Log_summary.xml',
    "gravity": '../logs/run_gravity_model/Log_summary.xml',
    "frank-wolfe": '../logs/run_frank_wolfe/Log_summary.xml',
    "entropy-maxim": '../logs/run_entropy_maxim/Log_summary.xml',
    "triprec": "../logs/logs_trip_reconstruction/Log_summary.xml",
    "triprec*": "../logs/logs_trip_reconstruction_perfect/Log_summary.xml"
}
   
errors_density = {}
errors_flow = {}
errors_speed = {}
for key in benchmark_methods:
    mfd_method = loadMethodMFD(benchmark_methods[key], INTERVAL)
    errors_density[key] = np.abs(mfd_original[:,1]-mfd_method[:,1])
    errors_flow[key]    = np.abs(mfd_original[:,4]-mfd_method[:,4])
    errors_speed[key]   = np.abs(mfd_original[:,3]-mfd_method[:,3])

plt.figure(figsize=(12,2.5))  # Adjusted figure size for better visibility

plt.suptitle("(A) Macroscopic Assessment", fontweight="bold", y=0.97)

# Define a color map for each model
color_map = {
    'dfrouter': 'lightsalmon',
    'gravity': 'lightpink',
    'frank-wolfe': 'wheat',
    'entropy-maxim': 'lightgreen',
    'triprec': 'lightblue',
    'triprec*': 'lavender'
}

def plot_horizontal_boxplot_with_avg(ax, errors, labelsShow=False):
    avg_errors = {k: np.mean(v) for k, v in errors.items()}
    bp = ax.boxplot(errors.values(), vert=False, patch_artist=True)
    
    # Set the facecolor of the boxes based on the model and make median line black
    for box, median, (model, _) in zip(bp['boxes'], bp['medians'], errors.items()):
        box.set(facecolor=color_map[model])
        median.set(color='black', linewidth=1.5)
    
    if labelsShow:
        labels = [f"{k} [{avg_errors[k]:.2f}]" for k in errors.keys()]
        ax.set_yticklabels(labels)
    else:
        labels = [f"[{avg_errors[k]:.2f}]" for k in errors.keys()]
        ax.set_yticklabels(labels)

plt.subplot(1,3,1)
plot_horizontal_boxplot_with_avg(plt.gca(), errors_speed, labelsShow=True)
plt.gca().set_xlabel("Speed [km/h]", fontweight="bold")

fig = plt.gcf()
pos1 = plt.gca().get_position()

plt.subplot(1,3,2)
plot_horizontal_boxplot_with_avg(plt.gca(), errors_flow)
plt.gca().set_xlabel("Flow [veh/h]", fontweight="bold")

plt.subplot(1,3,3)
plot_horizontal_boxplot_with_avg(plt.gca(), errors_density)
plt.gca().set_xlabel("Density [#Vehicles]", fontweight="bold")

text_x = pos1.x0 + 0.03 
text_y = pos1.y0 - 0.02  
fig.text(text_x, text_y, "Absolute Error", rotation=0, va='center', ha='right', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 1.05]) 
plt.show()