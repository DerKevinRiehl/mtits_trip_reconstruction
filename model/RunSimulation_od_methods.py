# #############################################################################
# ## IMPORTS
# #############################################################################
import os
import sys
if 'SUMO_HOME' in os.environ:
    sys.path.append(os.path.join(os.environ['SUMO_HOME'], 'tools'))
import traci
import pandas as pd
import numpy as np
import shutil




# #############################################################################
# ## METHODS
# #############################################################################
def loadSignalsFromFile(file):
    df = pd.read_csv(file)
    del df['Unnamed: 0']
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    return df

def loadSignals(traffic_light_controller):
    signal_data = {}
    for tflid in traffic_light_controller:
        file = "signals/TF_Signals_"+str(tflid)+".csv"
        signal_data[tflid] = loadSignalsFromFile(file)
    return signal_data

def loadDemandFromFile(file):
    df_car_spawn = pd.read_csv(file)
    df_car_spawn = df_car_spawn.rename(columns={"Unnamed: 0": "veh_ctr"})
    return df_car_spawn

def loadAllSignals():
    # Load
    traffic_light_controller = list(trafficlight_map.keys())
    signal_data_dict = loadSignals(traffic_light_controller)
    signal_data = None
    for key in traffic_light_controller:
        if signal_data is None:
            signal_data = signal_data_dict[key].copy()
        else:
            signal_data = signal_data.merge(signal_data_dict[key], on="Datetime", how="left")
    # DETERMINE TIMES
    df = signal_data[["Datetime"]]
    df = df[df["Datetime"]>=simulation_time_start]
    df = df[df["Datetime"]<=simulation_time_end]
    times = df["Datetime"].astype(str).tolist()
    return traffic_light_controller, signal_data, times

def launchSUMO():
    sumoBinary = "C:/Users/kriehl/AppData/Local/sumo-1.19.0/bin/sumo-gui.exe"
    # sumoBinary = "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe"
    sumoConfigFile = "Configuration.sumocfg" 
    sumoCmd = [sumoBinary, "-c", sumoConfigFile, "--start", "--quit-on-end", "--time-to-teleport", "-1"]
    traci.start(sumoCmd)




# #############################################################################
# ## PARAMETERS
# #############################################################################
simulation_steps_per_second = 4
simulation_time_start = "2024-03-04 09:15:00"
# simulation_time_end = "2024-03-04 10:15:00"
simulation_time_end = "2024-03-04 23:00:00"
trafficlight_number = {
    "cluster_278801067_282372132": 0,
    "cluster_6537970447_6537970448_6537970449_6537970451": 1,
    "cluster_1011603764_10814792332_252503857_458759462_#2more": 2,
    "cluster_10814792326_29027230": 3,
    "J1": 4,
    "cluster_10947641377_12271808954_12271808955_29027248_#4more": 5,
    "cluster_1934095630_30895401": 6, 
    "10828119451": 7,
    "1165953435": 8,
    "J2": 9, 
    "J3": 10, 
    "J4": 11,
    "J5": 12,
    "J6": 13,    
}
trafficlight_map = {value: key for key, value in trafficlight_number.items()}




# #############################################################################
# ## MAIN CODE
# #############################################################################

    # LOAD DATA
traffic_light_controller, signal_data, times = loadAllSignals()

for time in ["15m", "30m", "60m"]:
    for method in ["run_entropy_maxim", "run_frank_wolfe", "run_gravity_model"]:
        for seed in [1,2,3,4,5,6,7,8,9,10]:
            df_car_spawn = loadDemandFromFile("demand/Spawn_Cars_"+method+"_"+time+"_"+str(seed)+".csv")
            # LAUNCH SUMO
            launchSUMO()
            # RUN SIMULATION
            for current_time in times:
                # SET TRAFFIC LIGHTS
                signalStrings = signal_data[signal_data["Datetime"]==current_time]
                for key in traffic_light_controller:
                    signalString = signalStrings[str(key)].iloc[0]
                    tflid = trafficlight_map[key]
                    traci.trafficlight.setRedYellowGreenState(tflid, signalString)
                # SPAWN CARS
                for idx, row in df_car_spawn[df_car_spawn["Adjusted_Datetime"]==current_time].iterrows():
                    for x in range(0, int(np.ceil(row["n_spawn"]))):
                        traci.vehicle.add("Car_"+str(row["veh_ctr"]), row["route"], typeID="DEFAULT_CAR")
                # RUN SIMULATION FOR ONE SECOND
                for n in range(0,simulation_steps_per_second):
                    traci.simulationStep()
                print("\t", current_time)
            # STOP SUMO
            traci.close()
            # copy all files
            os.rename("logs", "logs_"+method+"_"+time+"_"+str(seed))
            # reset file structure
            os.mkdir("logs")
            os.mkdir("logs/loops")