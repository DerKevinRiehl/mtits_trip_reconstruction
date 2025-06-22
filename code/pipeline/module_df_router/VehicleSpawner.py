# #############################################################################
# Imports
# #############################################################################
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")


for seed in [1,2,3,4,5,6,7,8,9,10]:
    np.random.seed(42)  # Set seed for reproducibility
    
    folder = "../../model_logs/original/loops_dfrouter/"
    simulation_time_start = "2024-03-04 09:15:00"
    simulation_time_end = "2024-03-04 23:00:00"
    
    f = open("df_router_emitters.xml", "r")
    content = f.read()
    f.close()
    parts = content.split("</routeDistribution>")[:-1]
    parts = [part.split("\n") for part in parts]
    parts = [[line.strip() for line in part] for part in parts]
    
    spawn_dict = {}
    for part in parts:
        for line in part:
            if line.startswith("<routeDistribution"):
                detectorId = line.split("id=\"")[1].split("\"")[0]
                break
        spawns = {}
        for line in part:
            if line.startswith("<route "):
                routeName = line.split("refId=\"")[1].split("\"")[0]
                p = line.split("probability=\"")[1].split("\"")[0]
                spawns[routeName] = p
        spawn_dict[detectorId] = spawns
    
    spawn_dict2 = {}
    for detector in spawn_dict:
        options = list(spawn_dict[detector].keys())
        values = [float(v) for v in list(spawn_dict[detector].values())]
        values = [v/sum(values) for v in values]
        spawn_dict2[detector] = {
            "options": options,
            "probabilities": values,
        }
    
    
    
    def exponential_time_values(n, begin, end, rate=1.0):
        # Generate n exponentially distributed interarrival times
        inter_arrivals = np.random.exponential(scale=1/rate, size=n)
        # Normalize to fit within [begin, end]
        times = np.cumsum(inter_arrivals)
        times = begin + (times - times.min()) / (times.max() - times.min()) * (end - begin)
        return np.sort(times)
    
    vehicle_spawns = []
    for detector in spawn_dict2:
        f = open(folder+detector+".xml", "r")
        content = f.read()
        f.close()
        content = content.split("\n")
        content = [line.strip() for line in content]
        content = [line for line in content if line.startswith("<interval")]
        
        for line in content:
            begin = float(line.split("begin=\"")[1].split("\"")[0])
            end = float(line.split("end=\"")[1].split("\"")[0])
            n = int(line.split("nVehContrib=\"")[1].split("\"")[0])
            options = spawn_dict2[detector]["options"]
            values = spawn_dict2[detector]["probabilities"]
            
            if n>0:
                if n>=2:
                    times = exponential_time_values(n, begin, end)
                else:
                    times = [(begin+end)/2]
                for x in range(n):
                    # Choose route based on probabilities
                    route = str(np.random.choice(options, p=values))
                    vehicle_spawns.append([times[x], route])                       
    
    dfspawn = []
    ctr = 0
    for sp in vehicle_spawns:
        ctr += 1
        time = (datetime.strptime(simulation_time_start, "%Y-%m-%d %H:%M:%S") + timedelta(seconds=sp[0])).strftime("%Y-%m-%d %H:%M:%S")
        dfspawn.append([
            # ctr, 
            time, 
            1.0,
            "?",
            sp[1],
            0,
            time
        ])
    dfspawn = pd.DataFrame(dfspawn, columns=["Datetime", "n_spawn", "entrance", "route", "spawn_delay", "Adjusted_Datetime"])
    dfspawn.to_csv("Spawn_Cars_dfrouter_"+str(seed)+".csv")

