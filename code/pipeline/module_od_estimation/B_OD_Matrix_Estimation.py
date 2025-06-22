# #############################################################################
# Imports
# #############################################################################
import numpy as np
from scipy.optimize import minimize_scalar
import pandas as pd
from OD_Matrix_Estimation_tools import OD_MatrixEstimator, normalize_OD_Matrix
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
        # Aggregate to 15 minutes
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
        df_aggregated = df.resample(aggregation)[entrance].sum()
        df_aggregated = pd.DataFrame(df_aggregated).reset_index()
        date_range = pd.date_range(start=start_time.floor(aggregation), 
                                   end=end_time.ceil(aggregation), 
                                   freq=aggregation)
        df_complete = pd.DataFrame({'Datetime': date_range})
        df_final = pd.merge(df_complete, df_aggregated, on='Datetime', how='left').fillna(0)
        # merge to complete table
        if df_entrance_data is None:
            df_entrance_data = df_final.copy()
        else:
            df_entrance_data = df_entrance_data.merge(df_final, on="Datetime", how="left")
    return df_entrance_data




# #############################################################################
# Paths & Parameters
# #############################################################################

aggregation = "60T"
start_time = pd.to_datetime("2024-03-04 09:15:00")
end_time   = pd.to_datetime("2024-03-04 23:00:00")

folder_input_vector  = "vectors/vector_input/"
folder_output_vector = "vectors/vector_output/"
folder_target_od     = "od_matrix_60m/"

cost_matrix_path = "resources/cost_matrix.csv"

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

# Defining possible Routes
possible_routes_list = {
	"E1":  ["A4", "A7", "A9", "A10", "A11", "A14", "A15", "A16", "A17", "A18"],
	"E2":  ["A3"],
	"E3":  ["A1"],
	"E4":  ["A4", "A7", "A9", "A10", "A11", "A14", "A15", "A16", "A17", "A18"],
	"E5":  ["A3"],
	"E6":  ["A2"],
	"E7":  ["A1", "A4", "A7", "A9", "A10", "A11", "A14", "A15", "A16", "A17", "A18"],
	"E8":  ["A1", "A2", "A7", "A9", "A10", "A11", "A14", "A15", "A16", "A17", "A18"],
	"E9":  ["A1", "A2", "A4", "A6"],
	"E10": ["A5"],
	"E11": ["A7", "A9", "A10", "A11", "A14", "A15", "A16", "A17", "A18"],
	"E12": ["A8", "A9", "A10", "A11", "A14", "A15", "A16", "A17", "A18"],
	"E13": ["A1", "A2", "A4", "A5", "A6"],
	"E14": ["A1", "A2", "A4", "A5", "A6", "A7", "A10", "A11", "A14", "A15", "A16", "A17", "A18"],
	"E15": ["A1", "A2", "A4", "A5", "A6", "A7", "A9", "A11", "A14", "A15", "A16", "A17", "A18"],
	"E16": ["A1", "A2", "A4", "A5", "A6", "A7", "A9", "A10", "A14", "A15", "A16", "A17", "A18"],
	"E17": ["A1", "A2", "A4", "A5", "A6", "A7", "A9", "A10", "A11", "A14", "A15", "A16", "A17", "A18"],
	"E18": ["A1", "A2", "A4", "A5", "A6", "A7", "A9", "A10", "A11", "A14", "A15", "A16", "A17", "A18"],
	"E19": ["A1", "A2", "A4", "A5", "A6", "A7", "A9", "A10", "A11", "A15", "A16", "A17", "A18"],
	"E20": ["A1", "A2", "A4", "A5", "A6", "A7", "A9", "A10", "A11", "A14", "A16", "A17", "A18"],
	"E21": ["A17", "A18"],
	"E22": ["A1", "A2", "A4", "A5", "A6", "A7", "A9", "A10", "A11", "A14", "A15"],
	"E23": ["A1", "A2", "A4", "A5", "A6", "A7", "A9", "A10", "A11", "A14", "A15", "A16", "A18"],
	"E24": ["A1", "A2", "A4", "A5", "A6", "A7", "A9", "A10", "A11", "A14", "A15", "A16", "A17"],
}


# #############################################################################
# Main Code
# #############################################################################

# ## DETERMINE MASK MATRIX
mask_matrix = np.zeros((len(input_list), len(output_list)))
for entrance in possible_routes_list:
    for exitance in possible_routes_list[entrance]:
        idx1 = list(input_list.keys()).index(entrance)
        idx2 = list(output_list.keys()).index(exitance)
        mask_matrix[idx1][idx2] = 1

# ## LOAD DATA
df_input_data  = loadTrafficFlows(folder_input_vector,  input_list,  aggregation, start_time, end_time)
df_output_data = loadTrafficFlows(folder_output_vector, output_list, aggregation, start_time, end_time)
df_input_data['total']  = df_input_data.loc[:, 'E1':'E24'].sum(axis=1)
df_output_data['total'] = df_output_data.loc[:, 'A1':'A18'].sum(axis=1)
time_slots = df_input_data["Datetime"].astype(str).tolist()

# ## LOAD COST MATRIX
cost_matrix = np.loadtxt(cost_matrix_path, delimiter=',')

# ## DETERMINE CORRECTION FACTOR
df_diff = df_input_data[["Datetime", "total"]]
df_diff = df_diff.merge(df_output_data[["Datetime", "total"]], on="Datetime", how="left")
df_diff["difference"] = df_diff["total_x"] - df_diff["total_y"]
df_diff = df_diff.rename(columns={"total_x": "total_in", "total_y": "total_out"})
df_diff['correction_factor'] = df_diff['total_out'] / df_diff['total_in']

# ## DETERMINE OD-MATRIX
for time_slot in time_slots:
    print(time_slot)
    correction_factor = df_diff[df_diff["Datetime"]==time_slot]["correction_factor"].iloc[0]
    input_vector = list(df_input_data[df_input_data["Datetime"]==time_slot].values)[0][1:-1].tolist()
    input_vector = [v*correction_factor for v in input_vector]
    output_vector = list(df_output_data[df_output_data["Datetime"]==time_slot].values)[0][1:-1].tolist()
    
    # FRANK-WOLFE
    try:
        estimator = OD_MatrixEstimator(input_vector, output_vector, mask_matrix, algorithm="Frank-Wolfe")
        best_matrix = estimator.run_estimation()
        best_matrix = normalize_OD_Matrix(best_matrix)
        np.savetxt(folder_target_od+"run_frank_wolfe_"+time_slot.replace(" ", "_").replace(":", "_")+".csv", best_matrix, delimiter=',', fmt='%.10f')
        print("\tsuccess FRANK-WOLFE")
    except:
        print("\tfailed FRANK-WOLFE")
        
    # ENTROPY_MAXIMIZATION
    try:
        estimator2 = OD_MatrixEstimator(input_vector, output_vector, mask_matrix, algorithm="Entropy-Maximization")
        best_matrix2 = estimator.run_estimation()
        best_matrix2 = normalize_OD_Matrix(best_matrix2)
        np.savetxt(folder_target_od+"run_entropy_maxim_"+time_slot.replace(" ", "_").replace(":", "_")+".csv", best_matrix2, delimiter=',', fmt='%.10f')
        print("\tsuccess ENTROPY_MAXIMIZATION")
    except:
        print("\tfailed ENTROPY_MAXIMIZATION")

    # GRAVITY_MODEL
    try:
        estimator3 = OD_MatrixEstimator(input_vector, output_vector, mask_matrix, algorithm="Gravity-Model", cost_matrix=cost_matrix)
        best_matrix3 = estimator.run_estimation()
        best_matrix3 = normalize_OD_Matrix(best_matrix3)
        np.savetxt(folder_target_od+"run_gravity_model_"+time_slot.replace(" ", "_").replace(":", "_")+".csv", best_matrix3, delimiter=',', fmt='%.10f')
        print("\tsuccess GRAVITY_MODEL")
    except:
        print("\tfailed GRAVITY_MODEL")
