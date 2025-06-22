# #############################################################################
# Imports
# #############################################################################
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")




# #############################################################################
# Methods
# #############################################################################
def loadTrafficFlows(folder, entrance_list, aggregation, start_time, end_time):
    df_entrance_data = None
    for entrance in entrance_list:
        print("Load Data...", entrance)
        # Load Table
        df = pd.read_csv(folder+entrance+".csv", sep=";")
        del df["Unnamed: 0"]
        colname = df.columns[1]
        df2 = df[df[colname]!=0]
        df2["entrance"] = colname
        df2 = df2.rename(columns={colname:"n_spawn"})
        # Merge
        if df_entrance_data is None:
            df_entrance_data = df2.copy()
        else:
            df_entrance_data = pd.concat((df_entrance_data, df2))
    return df_entrance_data




# #############################################################################
# Paths & Parameters
# #############################################################################

aggregation = "30T"
aggregation2 = "30min"
start_time = pd.to_datetime("2024-03-04 09:15:00")
end_time   = pd.to_datetime("2024-03-10 23:00:00")

folder_input_vector  = "../module_1_od_estimation/vectors/vector_input/"
folder_od            = "../module_1_od_estimation/od_matrix_30m/"
data_path_target = "spawn_vehicles/"

for od_type in ["run_frank_wolfe", "run_gravity_model", "run_entropy_maxim"]:
    for SEED in [1,2,3,4,5,6,7,8,9,10]:
        input_list = {
        	"E1":  "26249185#1", 
        	"E2":  "26249185#1",
        	"E3":  "921020464#0",
        	"E4":  "921020464#0",
        	"E5":  "183049957#0",
        	"E6":  "-1169441386",
        	"E7":  "-1169441386",
        	"E8":  "-60430429#1",
        	"E9":  "-25973410#1",
        	"E10": "758088375#0",
        	"E11": "758088375#0",
        	"E12": "-E4",
        	"E13": "-E4",
        	"E14": "25497525",
        	"E15": "-25576697#0",
        	"E16": "89290458#1",
        	"E17": "23320502",
        	"E18": "-23320456#1",
        	"E19": "E0",
        	"E20": "-394114218#1",
        	"E21": "1162834479#1",
        	"E22": "1162834479#1",
        	"E23": "-208691154#0",
        	"E24": "E13",
        }
        output_list = {
        	"A1":  "183419043#1",
        	"A2":  "694116204#2",
        	"A3":  "331752492#2",
        	"A4":  "60430429#1",
        	"A5":  "25973410#1",
        	"A6":  "-758088375#0",
        	"A7":  "E4",
        	"A8":  "-E2",
        	"A9":  "-25497525",
        	"A10": "25576697#0",
        	"A11": "-89290458#1",
        	"A12": "-23320502",
        	"A13": "23320456#1",
        	"A14": "-E0",
        	"A15": "394114218#1",
        	"A16": "-1162834479#1",
        	"A17": "208691154#0",
        	"A18": "-E13",
        }
        
        
        
        
        # #############################################################################
        # Main Code
        # #############################################################################
        np.random.seed(SEED)
        
        # ## LOAD DATA
        df_input_data  = loadTrafficFlows(folder_input_vector,  input_list,  aggregation, start_time, end_time)
        df_input_data["Datetime"] = pd.to_datetime(df_input_data["Datetime"])
        df_input_data['slot'] = df_input_data['Datetime'].dt.floor(aggregation2)
        df_input_data2 = df_input_data.sort_values(by='slot', ascending=True)
        df_input_data = df_input_data2.copy()
        # Filter relevant time frame
        df_input_data = df_input_data[df_input_data["Datetime"]>=start_time]
        df_input_data = df_input_data[df_input_data["Datetime"]<=end_time]
        # drop based on random chance when flow below 1.0
        df_input_data['not_drop'] = (df_input_data['n_spawn'] > 1.0) | (np.random.random(len(df_input_data)) < df_input_data['n_spawn'])
        df_input_data = df_input_data[df_input_data["not_drop"]==True]
        del df_input_data['not_drop']
        # Expand rows when flow greater than 1.0 
        def expand_rows(row):
            if row['n_spawn'] == 1.0:
                return [row]
            else:
                num_rows = int(np.ceil(row['n_spawn']))
                expanded_rows = [row.copy() for _ in range(num_rows)]
                for expanded_row in expanded_rows:
                    expanded_row['n_spawn'] = 1.0
                return expanded_rows
        
        expanded_df = pd.DataFrame(
            [expanded_row for _, row in df_input_data.iterrows() for expanded_row in expand_rows(row)]
        )
        expanded_df = expanded_df.reset_index(drop=True)
        df_input_data = expanded_df.copy()
        
        # Randomly determine Route with OD Matrix
        lastslot = None
        odmatrix = None
        routes = []
        ctr = 0
        for idx, row in df_input_data.iterrows():
            ctr+=1
            print(ctr, len(df_input_data))
            slot = str(row["slot"])
            # load new od matrix if necessary
            if lastslot is None or lastslot!=slot:
                lastslot = slot
                od_file = folder_od+od_type+"_"+slot.replace(":","_").replace(" ","_")+".csv"
                odmatrix = np.loadtxt(od_file, delimiter=',')
            idx = list(input_list.keys()).index(row["entrance"])
            probs = odmatrix[idx]
            selected_exit = np.random.choice(len(probs), p=probs)
            exit_label = list(output_list.keys())[selected_exit]
            selected_route = "route_"+row["entrance"]+"_"+exit_label
            routes.append(selected_route)
        df_input_data["route"] = routes
        
        df_input_data2 = df_input_data.sort_values(by='Datetime', ascending=True)
        df_input_data = df_input_data2.copy()
        del df_input_data["slot"]
        
        # Earlier Spawning
        spawn_delay = {
            "E1":	2,
            "E2":	0,
            "E3":	3,
            "E4":	3,
            "E5":	4,
            "E6":	2,
            "E7":	2,
            "E8":	14,
            "E9":	3,
            "E10":	1,
            "E11":	2,
            "E12":  3,
            "E13":  4,
            "E14":  10,
            "E15":	1,
            "E16":	15,
            "E17":	1,
            "E18":	9,
            "E19":	0,
            "E20":	4,
            "E21":	0,
            "E22":	0,
            "E23":	0,
            "E24":	9,
            "E25":  9,
        }
        
        df_input_data['spawn_delay'] = df_input_data['entrance'].map(spawn_delay)
        df_input_data['Adjusted_Datetime'] = df_input_data['Datetime'] - pd.to_timedelta(df_input_data['spawn_delay'], unit='seconds')
        df_input_data2 = df_input_data.sort_values(by='Adjusted_Datetime', ascending=True)
        df_input_data = df_input_data2.copy()
        
        df_input_data.to_csv(data_path_target+"Spawn_Cars_"+od_type+"_"+str(aggregation.replace("T", "m"))+"_"+str(SEED)+".csv")