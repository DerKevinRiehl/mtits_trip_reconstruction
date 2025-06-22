# #############################################################################
# Imports
# #############################################################################
import pandas as pd
import zipfile
import warnings
import os
warnings.filterwarnings("ignore")




# #############################################################################
# PARAMETERS
# #############################################################################
resource_path = "resources/"
path_eingang_files = "vectors/vector_input/"
path_ausgang_files = "vectors/vector_output/"
path_rest_files = "vectors/vector_internal/"
start_time = pd.to_datetime("2024-03-04 09:15:00")
end_time   = pd.to_datetime("2024-03-10 23:00:00")

detector_log_files = "../../model_logs/original/loops/"





# #############################################################################
# LOAD MAPS
# #############################################################################
EntriesDH = pd.read_excel(resource_path + 'VerteilungenImDarkHole.xlsx', sheet_name='Entry')
ExitsDH = pd.read_excel(resource_path + 'VerteilungenImDarkHole.xlsx', sheet_name='Exit')
Entries = pd.read_excel(resource_path + 'CorrespondingDetectors.xlsx', sheet_name='Entry(E)')
Exits = pd.read_excel(resource_path + 'CorrespondingDetectors.xlsx', sheet_name='Exit(A)')
Inter_I = pd.read_excel(resource_path + 'CorrespondingDetectors.xlsx', sheet_name='InterEntry(I)')
Inter_L = pd.read_excel(resource_path + 'CorrespondingDetectors.xlsx', sheet_name='InterExit(L)')
MovementDistributionVISSIM = pd.read_excel(resource_path + 'MovementDistribution.xlsx', sheet_name='VISSIM')
MovementDistributionAssumptions = pd.read_excel(resource_path + 'MovementDistribution.xlsx', sheet_name='Assumptions')
MovementCalculations = pd.read_excel(resource_path + 'ComputedVectors.xlsx', sheet_name='VectorCalculations')
checks_L_I = pd.read_excel(resource_path + 'Differences_L_I.xlsx', sheet_name='Checks')
VectorCalculations_2 = pd.read_excel(resource_path + 'Differences_L_I.xlsx', sheet_name='Differences_L_I')
list_maps = [Entries, Exits, Inter_I, Inter_L]

list_all_vectors = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20', 'E21', 'E22', 'E23', 'E24', 'E25', 
                'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I13', 'I14', 'I15', 'I16', 'I17', 'I18',
                'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18',
                'L1', 'L2_3', 'L4', 'L5', 'L6', 'L7', 'L8', 'L9', 'L10', 'L11', 'L12', 'L13']
list_vector_out_of_detectors = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E9', 'E10', 'E11', 'E12', 'E13', 'E19', 'E21', 'E22', 'E23', 'E24', 'E25', 
                'I1', 'I2', 'I3', 'I4', 'I5', 'I6', 'I7', 'I8', 'I9', 'I10', 'I11', 'I12', 'I15', 'I16', 'I17', 'I18',
                'L11', 'L12', 'A18']
movementVectors = ['E7', 'I1', 'E9', 'I4', 'E12', 'I10', 'E19', 'I15', 'I16', 'E21', 'E23', 'E24', 'I18']
list_compute_vector_1 = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18',
                         'L1', 'L2_3', 'L4', 'L5', 'L6', 'L13']
list_compute_vector_2 =['E20', 'A15']
list_compute_vector_3 = checks_L_I['Vector']
list_compute_vector_4 = ['A4', 'E8', 'A9', 'E14']
compute_4 = ["E8_A4_1", "E8_A4_2", "E14_A9_1", "E14_A9_2", "DH_1", "DH_2"]

list_L_I = ["E8_A4_1", "E8_A4_2", "Check1_1", "Check1_2", "E14_A9_1", "E14_A9_2", "DH_1", "DH_2", "Check2_1", "Check2_2"]
movementVectors_2 = ['E15', 'E17', 'E18', 'E20']





# #############################################################################
# Methods
# #############################################################################
def load_dataframe_from_zipped_csv(zip_file, csv_file, headerSet=True):
    with zipfile.ZipFile(zip_file) as z:
        with z.open(csv_file) as f:
            if headerSet:
                df = pd.read_csv(f, sep=";", header=1)
            else:
                df = pd.read_csv(f, sep=";")
    return df

def movement_two_or_three(vector, VISSIM = True):
    if VISSIM:
        df = MovementDistributionVISSIM
    else:
        df = MovementDistributionAssumptions
    if vector in df.iloc[:, 0].values:
    # Get the corresponding value from the second column
        if pd.isna(df.loc[df.iloc[:, 0] == vector].iloc[0, 5]):
            return(True) # True = 2 Movements
        else:
            return(False) # False = 3 Movements
    
def find_percentage2(vector, VISSIM = True):
    if VISSIM:
        df = MovementDistributionVISSIM
    else:
        df = MovementDistributionAssumptions
    if vector in df.iloc[:, 0].values:
    # Get the corresponding value from the second column
        value = df.loc[df.iloc[:, 0] == vector].iloc[0, 3]
    return(value)

def find_percentage3(vector, VISSIM = True):
    if VISSIM:
        df = MovementDistributionVISSIM
    else:
        df = MovementDistributionAssumptions
    if vector in df.iloc[:, 0].values:
    # Get the corresponding value from the second column
        value2 = df.loc[df.iloc[:, 0] == vector].iloc[0, 3]
        value3 = df.loc[df.iloc[:, 0] == vector].iloc[0, 5]
    return(value2, value3)

def find_detectors(vector):
    if vector.startswith('E'):
        df = Entries
    elif vector.startswith('I'):
        df = Inter_I
    elif vector.startswith('L'):
        df = Inter_L
    elif vector.startswith('A'):
        df = Exits
    if vector in df.iloc[:, 0].values:
    # Get the corresponding value from the second column
        value = df.loc[df.iloc[:, 0] == vector].iloc[0, 1]
    # Return the value if it is not NaN
    if not pd.isna(value):
        return value
    # Return None if entry is not found or value is NaN
    return None

def lanesplit_2(dataframe, column, percentage):
    # Percentage of the Cars will turn into Movement 1
    movement1 = []
    movement2 = []
    for value in dataframe[column]:
        movement1.append(value*(1-percentage))
        movement2.append(value*(percentage))
        
        # if value == 1:
        #     # If it's 1, add it to list with a probability
        #     if np.random.rand() <= percentage:
        #         movement1.append(0.0)
        #         movement2.append(value)
        #     else:
        #         movement1.append(value)
        #         movement2.append(0.0)
        # elif value == 0:
        #     # If it's not 1, always add to both lists
        #     movement1.append(value)
        #     movement2.append(value)
        # else:
        #     print("There is an Error in " + column)
    return movement1, movement2

def lanesplit_3(dataframe, column, percentage2, percentage3):
    # Percentage of Cars will turn into Movement 1
    # Percentage of Cars will turn into Movement 2
    movement1 = []
    movement2 = []
    movement3 = []
    for value in dataframe[column]:
        movement3.append(value*(percentage3))
        movement2.append(value*(percentage2))
        movement1.append(value*(1-percentage2-percentage3))
        
        # if value == 1:
        #     rand = np.random.rand()
        #     # If it's 1, add it to list with a probability
        #     if rand <= percentage3:
        #         movement1.append(0.0)
        #         movement2.append(0.0)
        #         movement3.append(value)
        #     elif percentage2 < rand <= (percentage2 + percentage3):
        #         movement1.append(0.0)
        #         movement2.append(value)
        #         movement3.append(0.0)
        #     else:
        #         movement1.append(value)
        #         movement2.append(0.0)
        #         movement3.append(0.0)
        # elif value == 0:
        #     # If it's not 1, always add to both lists
        #     movement1.append(value)
        #     movement2.append(value)
        #     movement3.append(value)
        # else:
        #     print("There is an Error in " + column)
    return movement1, movement2, movement3

def prepare_single_csv(zip_file, csv_file, start_time, end_time):
    # LOAD CSV FROM ZIP
    data = load_dataframe_from_zipped_csv(zip_file, csv_file)
    # DATETIME COLUMN
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    del data["Date"]
    del data["Time"]
    # REPLACE CHARACTERS BY NUMBERS
    lst_columns = list(data.columns)
    lst_columns.remove("Datetime")
    for information in lst_columns:
        data[information] = data[information].replace('.', '0')
        data[information] = data[information].replace('x', '0')
        data[information] = data[information].replace('|', '0')
        data[information] = data[information].astype(int)
    # MAKE SURE EVERY SECOND EXISTS; AGGREGATE ON SECOND LEVEL
    data.set_index('Datetime', inplace=True)
    df_aggregated = data.resample(aggregation)[lst_columns].sum()
    df_aggregated = pd.DataFrame(df_aggregated).reset_index()
    # Make sure that missing 1 second slots are added
        # Create a complete date range with0 second intervals
    date_range = pd.date_range(start=start_time.floor(aggregation), 
                               end=end_time.ceil(aggregation), 
                               freq=aggregation)
        # Create a new DataFrame with all possible 15-minute intervals
    df_complete = pd.DataFrame({'Datetime': date_range})
        # Merge with original data, filling NaN with 0
    df_final = pd.merge(df_complete, df_aggregated, on='Datetime', how='left').fillna(0)
        # Ensure detector is of integer type
    for information in lst_columns:
        df_final[information] = df_final[information].astype(int)
        # Sort by Datetime
    df_final = df_final.sort_values('Datetime')
        # Reset index
    df_final = df_final.reset_index(drop=True)
    data = df_final.copy()
    # SELECT COLUMNS
    lst_selected_columns = [col for col in lst_columns if col.startswith("D")]
    lst_selected_columns.append("Datetime")
    data = data[[*lst_selected_columns]]
    # RENAME COLUMNS
    lst_columns = list(data.columns)
    lst_columns.remove("Datetime")
    intersection_name = csv_file.split("/")[1].replace(".csv","_")
    for column in lst_columns:
        data = data.rename(columns={column: intersection_name+column})
    return data

def prepare_single_csv_only_off(zip_file, csv_file, start_time, end_time):
    # LOAD CSV FROM ZIP
    data = load_dataframe_from_zipped_csv(zip_file, csv_file)
    # DATETIME COLUMN
    data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'])
    del data["Date"]
    del data["Time"]
    # REPLACE CHARACTERS BY NUMBERS
    lst_columns = list(data.columns)
    lst_columns.remove("Datetime")
    for information in lst_columns:
        data[information] = data[information].replace('.', '0')
        data[information] = data[information].replace('x', '10')
        data[information] = data[information].replace('|', '0')
        data[information] = data[information].astype(int)
        data.loc[data[information] < 10, information] = 0
        data.loc[data[information] == 10, information] = 1
    # MAKE SURE EVERY SECOND EXISTS; AGGREGATE ON SECOND LEVEL
    data.set_index('Datetime', inplace=True)
    df_aggregated = data.resample(aggregation)[lst_columns].sum()
    df_aggregated = pd.DataFrame(df_aggregated).reset_index()
    # Make sure that missing 1 second slots are added
        # Create a complete date range with0 second intervals
    date_range = pd.date_range(start=start_time.floor(aggregation), 
                               end=end_time.ceil(aggregation), 
                               freq=aggregation)
        # Create a new DataFrame with all possible 15-minute intervals
    df_complete = pd.DataFrame({'Datetime': date_range})
        # Merge with original data, filling NaN with 0
    df_final = pd.merge(df_complete, df_aggregated, on='Datetime', how='left').fillna(0)
        # Ensure detector is of integer type
    for information in lst_columns:
        df_final[information] = df_final[information].astype(int)
        # Sort by Datetime
    df_final = df_final.sort_values('Datetime')
        # Reset index
    df_final = df_final.reset_index(drop=True)
    data = df_final.copy()
    # SELECT COLUMNS
    lst_selected_columns = [col for col in lst_columns if col.startswith("D")]
    lst_selected_columns.append("Datetime")
    data = data[[*lst_selected_columns]]
    # RENAME COLUMNS
    lst_columns = list(data.columns)
    lst_columns.remove("Datetime")
    intersection_name = csv_file.split("/")[1].replace(".csv","_")
    for column in lst_columns:
        data = data.rename(columns={column: intersection_name+column})
    return data

def loadLoopDetectorFiles(key, value):
    if value=="EMPTY":
        df = pd.DataFrame({
            'Datetime': pd.date_range(start=start_time, end=end_time, freq='S'),
            key: 0
        })
        return df
    else:
        with open(detector_log_files+value, "r") as f:
            content = f.read()
        lines = [line.strip() for line in content.split("\n") if line.strip().startswith("<instantOut") and "leave" in line]
        loop_events = []
        for line in lines:
            time = float(line.split("time=\"")[1].split("\"")[0])
            loop_events.append([time, 1])
        df_loop_events = pd.DataFrame(loop_events, columns=["time", key])
        df_loop_events['Datetime'] = start_time + pd.to_timedelta(df_loop_events['time'], unit='s')
        df_loop_events = df_loop_events.drop(columns=['time'])
        df_loop_events = df_loop_events[['Datetime'] + [col for col in df_loop_events.columns if col != 'Datetime']]
        df_loop_events['Datetime'] = df_loop_events['Datetime'].dt.round('S')
        df_loop_events = df_loop_events.groupby('Datetime').sum().reset_index()
        return df_loop_events
    
# ###########################################################################
# LOAD BIG TABLE
# ###########################################################################
available_detector_files = os.listdir(detector_log_files)

files_to_load = {
    "ES217_D81": "217_D81.xml",
    "ES217_D62": "217_D61.xml",
    "ES217_D41": "217_D41.xml",
    "ES217_D21": "217_D21.xml",
    "ES217_D11": "217_D11.xml",
    "ES217_D151": "217_D151.xml",
    "ES217_D131": "217_D131.xml",
    "ES216_D61a": "216_D61.xml",
    "ES216_D22": "216_D21.xml",
    "ES216_D11": "216_D11.xml",
    "ES215_D12": "215_D12.xml",
    "ES215_D31a": "215_D31.xml",
    "ES212_D1": "212_D1.xml",
    "ES235_D81": "235_D81.xml",
    "ES235_D61": "235_D61.xml",
    "ES235_D11": "235_D11.xml",
    "ES235_D101.1": "235_D101.xml",
    "ES235_D101.2": "EMPTY",
    "ES217_D101": "217_D101.xml",
    "ES217_D104": "217_D103.xml",
    "ES216_D41": "216_D41.xml",
    "ES216_D81": "216_D81.xml",
    "ES216_D101a": "216_D101.xml",
    "ES215_D111a": "215_D111.xml",
    "ES215_D91": "215_D91.xml",
    "ES215_D51": "215_D51.xml",
    "ES215_D71": "215_D71.xml",
    "ES211_DO15": "211_D015.xml",
    "ES211_DO35": "211_D035.xml",
    "ES211_D611": "EMPTY",
    "ES213_D42": "213_D42.xml",
    "ES213_D52": "213_D52.xml",
    "ES235_D41a": "235_D42.xml",
    "ES235_DO25": "235_D025.xml",
    "ES213_D15b": "EMPTY",
    "ES213_D35a": "EMPTY",
    "ES235_D125b": "235_D125.xml",
    "ES235_D104": "235_D104.xml"
}

big = pd.DataFrame({
    'Datetime': pd.date_range(start=start_time, end=end_time, freq='S'),
})

for key in files_to_load:
    df_loop_events = loadLoopDetectorFiles(key, files_to_load[key])
    big = big.merge(df_loop_events, on="Datetime", how="left").fillna(0)

aggregation = "1S"

# ###########################################################################
# Main Code
# ###########################################################################
# Step 1: Vectors out of Detectors
list_relevant_detectors = []
list_to_compute = []
list_vector_out_of_detectors_2 = list_vector_out_of_detectors.copy()
list_vector_out_of_detectors_2.remove('I9')
list_vector_out_of_detectors_2.remove('E25')
for vector in list_vector_out_of_detectors_2:
    detector = find_detectors(vector)
    # if detector not in list_relevant_detectors:
    #     list_relevant_detectors.append(detector)
    if not pd.isna(detector):
        list_relevant_detectors.append(detector)
    else:
        list_to_compute.append(vector)
big_day = big[["Datetime", *list_relevant_detectors]]
# Rename the Columns
for vector in list_vector_out_of_detectors_2:
    detector = find_detectors(vector)
    big_day = big_day.rename({detector : vector}, axis='columns')
    

# Step 2: Compute Vectors: 'A1', 'A2', 'A3', 'A5', 'A6', 'A7', 'A8', 'A10', 'A16', 'A17', 'L1', 'L2_3', 'L4', 'L5', 'L6', 'L13'
# Create DataFrame
datetime_column = big_day['Datetime']
movements_1 = pd.DataFrame({'Datetime': datetime_column})
movements = {}
movements['Datetime'] = datetime_column

for vector in movementVectors:
    if vector in MovementDistributionVISSIM['Movement'].values:
        df = MovementDistributionVISSIM
        VISSIM = True
    else:
        df = MovementDistributionAssumptions
        VISSIM = False
        
    if vector in df.iloc[:, 0].values:
    # Get the corresponding value from the second column
        if movement_two_or_three(vector, VISSIM):
            value1 = df.loc[df.iloc[:, 0] == vector].iloc[0, 2]
            value2 = df.loc[df.iloc[:, 0] == vector].iloc[0, 4]
            movements[f"{vector}_{value1}"], movements[f"{vector}_{value2}"] = lanesplit_2(big_day, vector, find_percentage2(vector, VISSIM))
        else:
            value1 = df.loc[df.iloc[:, 0] == vector].iloc[0, 2]
            value2 = df.loc[df.iloc[:, 0] == vector].iloc[0, 4]
            value3 = df.loc[df.iloc[:, 0] == vector].iloc[0, 6]
            percentage2, percentage3 = find_percentage3(vector, VISSIM)
            movements[f"{vector}_{value1}"], movements[f"{vector}_{value2}"], movements[f"{vector}_{value3}"] = lanesplit_3(big_day, vector, percentage2, percentage3)
movements['E24_left'] = big['ES235_D104'].tolist()

MovementCalculations_2 = MovementCalculations.copy()
MovementCalculations = MovementCalculations.drop(MovementCalculations.index[20])



# Calculate new Vectors
for vector in list_compute_vector_1:
    computed_movement = [0.0] * big_day.shape[0]
    if vector in MovementCalculations.iloc[:, 0].values:
        row = MovementCalculations[MovementCalculations['Vector'] == vector]
        for value in row.iloc[0, 1:]:
            if pd.isna(value):
                break
            else:
                if value in list(movements.keys()):
                    computed_movement = [a+b for a,b, in zip(computed_movement, movements[value])]
                else:
                    computed_movement = [a+b for a,b, in zip(computed_movement, big_day[value])]
        movements_1[vector] = computed_movement
big_day_step_2 = pd.merge(big_day, movements_1, on='Datetime', how='inner')

# Step 3: Compute 'E20' and 'A15'
difference_west = big_day_step_2['I16']-big_day_step_2['L11']
difference_east = big_day_step_2['I15']-big_day_step_2['L12']
E20_1 = []
E20_2 = []
A15_1 = []
A15_2 = []
for value in difference_east:
    if value == 0:
        E20_1.append(value)
        A15_1.append(value)
    elif value > 0:
        E20_1.append(0)
        A15_1.append(value)
    elif value < 0:
        E20_1.append(-1*value)
        A15_1.append(0)

for value in difference_west:
    if value == 0:
        E20_2.append(value)
        A15_2.append(value)
    elif value > 0:
        E20_2.append(0)
        A15_2.append(value)
    elif value < 0:
        E20_2.append(-1*value)
        A15_2.append(0)       

E20 = [a+b for a,b, in zip(E20_1, E20_2)]
A15 = [a+b for a,b, in zip(A15_1, A15_2)]

big_day_step_2['E20'] = E20
big_day_step_2['A15'] = A15

# Step 4: Check Differences between I and L
# START QuickFix

L5 = [a+b for a,b, in zip(big_day_step_2['I5'], big_day_step_2['I4'])]
big_day_step_2['L5'] = L5
# # END QuickFix
# check_L = pd.DataFrame(datetime_column)
# check_L_entry = pd.DataFrame(datetime_column)
# for vector in list_compute_vector_3:
#     computed_movement = [0.0] * big_day.shape[0]
#     if vector in checks_L_I.iloc[:, 0].values:
#         row = checks_L_I[checks_L_I['Vector'] == vector]
#         for value in row.iloc[0, 1:]:
#             if pd.isna(value):
#                 print("NaN for "+vector)
#             else:
#                 if value.startswith('I'):
#                     # print(value)
#                     computed_movement = [a+b for a,b, in zip(computed_movement, big_day_step_2[value])]
#                 elif value.startswith('L'):
#                     check_L_entry[vector] = computed_movement
#                     # print(value)
#                     computed_movement = [a-b for a,b, in zip(big_day_step_2[value], computed_movement)]
#         check_L[vector] = computed_movement

# check_L = check_L.set_index("Datetime")
# check_L_entry = check_L_entry.set_index("Datetime")
# column_sums_entry = check_L_entry.sum().tolist()
# column_sums = check_L.sum().tolist()
# result = [a / b for a, b in zip(column_sums, column_sums_entry)]

# Step 5: Compute Vectors: 'A4', 'E8', 'A9', 'E14'
I9 = [a - b for a, b in zip(big_day_step_2['L5'], big_day_step_2['E13'])]
big_day_step_2['I9'] = I9
vector = big_day_step_2['I10']
big_day_step_2['L8_2'] = vector
vector = big_day_step_2['I11']
big_day_step_2['L7'] = vector

compute4_df = pd.DataFrame(datetime_column)
for vector in compute_4:
    computed_movement = [0.0] * big_day.shape[0]
    if vector in VectorCalculations_2.iloc[:, 0].values:
        row = VectorCalculations_2[VectorCalculations_2['Vector'] == vector]
        for value in row.iloc[0, 1:]:
            if pd.isna(value):
                print("NaN for "+vector)
            else:
                if value.startswith('I'):
                    # print(value)
                    computed_movement = [a+b for a,b, in zip(computed_movement, big_day_step_2[value])]
                elif value.startswith('L'):
                    # print(value)
                    computed_movement = [a-b for a,b, in zip(big_day_step_2[value], computed_movement)]
        compute4_df[vector] = computed_movement

# Compute E8, A4
difference_west = compute4_df['E8_A4_1']
difference_east = compute4_df['E8_A4_2']
E8_1 = []
E8_2 = []
A4_1 = []
A4_2 = []
for value in difference_east:
    if value == 0:
        E8_1.append(value)
        A4_1.append(value)
    elif value > 0:
        E8_1.append(0)
        A4_1.append(value)
    elif value < 0:
        E8_1.append(-1*value)
        A4_1.append(0)

for value in difference_west:
    if value == 0:
        E8_2.append(value)
        A4_2.append(value)
    elif value > 0:
        E8_2.append(0)
        A4_2.append(value)
    elif value < 0:
        E8_2.append(-1*value)
        A4_2.append(0)       

E8 = [a+b for a,b, in zip(E8_1, E8_2)]
A4 = [a+b for a,b, in zip(A4_1, A4_2)]

big_day_step_2['E8'] = E8
big_day_step_2['A4'] = A4

difference_west = compute4_df['E14_A9_1']
difference_east = compute4_df['E14_A9_1']
E14_1 = []
E14_2 = []
A9_1 = []
A9_2 = []
for value in difference_east:
    if value == 0:
        E14_1.append(value)
        A9_1.append(value)
    elif value > 0:
        E14_1.append(0)
        A9_1.append(value)
    elif value < 0:
        E14_1.append(-1*value)
        A9_1.append(0)

for value in difference_west:
    if value == 0:
        E14_2.append(value)
        A9_2.append(value)
    elif value > 0:
        E14_2.append(0)
        A9_2.append(value)
    elif value < 0:
        E14_2.append(-1*value)
        A9_2.append(0)       

E14 = [a+b for a,b, in zip(E14_1, E14_2)]
A9 = [a+b for a,b, in zip(A9_1, A9_2)]

big_day_step_2['E14'] = E14
big_day_step_2['A9'] = A9

# Step 5: Compute Vectors 'Dark Hole': 'E15', 'E16', 'E17', 'E18', 'A10', 'A11', 'A12', 'A13', 'A14'
difference_west = compute4_df['DH_1']
difference_east = compute4_df['DH_2']
list_entries_DH_1 = {
    'E15_1': [],
    'E16_1': [],
    'E17_1': [],
    'E18_1': []
}
list_entries_DH_2 = {
    'E15_2': [],
    'E16_2': [],
    'E17_2': [],
    'E18_2': []
}
list_exits_DH_1 = {
    'A10_1': [],
    'A11_1': [],
    'A12_1': [],
    'A13_1': [],
    'A14_1': []
}
list_exits_DH_2 = {
    'A10_2': [],
    'A11_2': [],
    'A12_2': [],
    'A13_2': [],
    'A14_2': []
}
for value in difference_east:
    if value == 0:
        for vector in list_entries_DH_1:
            list_entries_DH_1[vector].append(0)
        for vector in list_exits_DH_1:
            list_exits_DH_1[vector].append(0)
    elif value > 0:
        for vector in list_entries_DH_1:
            list_entries_DH_1[vector].append(0)
        for vector in list_exits_DH_1:
            prob = ExitsDH.loc[ExitsDH['Vector'] == vector, 'Probability'].values
            list_exits_DH_1[vector].append(float(value*prob))
    elif value < 0:
        for vector in list_entries_DH_1:
            prob = EntriesDH.loc[EntriesDH['Vector'] == vector, 'Probability'].values
            list_entries_DH_1[vector].append(float(-1*value*prob))
        for vector in list_exits_DH_1:
            list_exits_DH_1[vector].append(0)

for value in difference_west:
    if value == 0:
        for vector in list_entries_DH_2:
            list_entries_DH_2[vector].append(0)
        for vector in list_exits_DH_2:
            list_exits_DH_2[vector].append(0)
    elif value > 0:
        for vector in list_entries_DH_2:
            list_entries_DH_2[vector].append(0)
        for vector in list_exits_DH_2:
            prob = ExitsDH.loc[ExitsDH['Vector'] == vector, 'Probability'].values
            list_exits_DH_2[vector].append(float(value*prob))
    elif value < 0:
        for vector in list_entries_DH_2:
            prob = EntriesDH.loc[EntriesDH['Vector'] == vector, 'Probability'].values
            list_entries_DH_2[vector].append(float(-1*value*prob))
        for vector in list_exits_DH_2:
            list_exits_DH_2[vector].append(0)    

E15 = [a + b for a, b in zip(list_entries_DH_1['E15_1'], list_entries_DH_2['E15_2'])]
E16 = [a + b for a, b in zip(list_entries_DH_1['E16_1'], list_entries_DH_2['E16_2'])]
E17 = [a + b for a, b in zip(list_entries_DH_1['E17_1'], list_entries_DH_2['E17_2'])]
E18 = [a + b for a, b in zip(list_entries_DH_1['E18_1'], list_entries_DH_2['E18_2'])]
A10 = [a + b for a, b in zip(list_exits_DH_1['A10_1'], list_exits_DH_2['A10_2'])]
A11 = [a + b for a, b in zip(list_exits_DH_1['A11_1'], list_exits_DH_2['A11_2'])]
A12 = [a + b for a, b in zip(list_exits_DH_1['A12_1'], list_exits_DH_2['A12_2'])]
A13 = [a + b for a, b in zip(list_exits_DH_1['A13_1'], list_exits_DH_2['A13_2'])]
A14 = [a + b for a, b in zip(list_exits_DH_1['A14_1'], list_exits_DH_2['A14_2'])]

big_day_step_2['E15'] = E15
big_day_step_2['E16'] = E16
big_day_step_2['E17'] = E17
big_day_step_2['E18'] = E18
big_day_step_2['A10'] = A10
big_day_step_2['A11'] = A11
big_day_step_2['A12'] = A12
big_day_step_2['A13'] = A13
big_day_step_2['A14'] = A14


big_day_step_2 = big_day_step_2.sort_index(axis=1)


# Final Check
# list_all_exit = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18']
# list_all_entry = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20', 'E21', 'E22', 'E23', 'E24']

# # Check Sums Total
# exit_total = 0
# for vector in list_all_exit:
#     exit_total += sum(big_day_step_2[vector])
# entry_total = 0

# for vector in list_all_entry:
#     entry_total += sum(big_day_step_2[vector])
    
# diff_copy = big_day_step_2.copy()

# # Sum columns starting with 'A'
# diff_copy['A_total'] = diff_copy.filter(regex='^A').sum(axis=1)

# # Sum columns starting with 'E'
# diff_copy['E_total'] = diff_copy.filter(regex='^E').sum(axis=1)

# # Select columns
# diff_copy = diff_copy[["Datetime", "A_total", "E_total"]]

# # Aggregate by 15 minutes
# aggregation = "15T"
# diff_copy.set_index('Datetime', inplace=True)
# lst_columns = list(diff_copy.columns)
# df_aggregated = diff_copy.resample(aggregation)[lst_columns].sum()
# df_aggregated = pd.DataFrame(df_aggregated).reset_index()
# # Make sure that missing 1 second slots are added
#     # Create a complete date range with0 second intervals
# date_range = pd.date_range(start=start_time.floor(aggregation), 
#                            end=end_time.ceil(aggregation), 
#                            freq=aggregation)
#     # Create a new DataFrame with all possible 15-minute intervals
# df_complete = pd.DataFrame({'Datetime': date_range})
#     # Merge with original data, filling NaN with 0
# df_final = pd.merge(df_complete, df_aggregated, on='Datetime', how='left').fillna(0)
#     # Ensure detector is of integer type
# for information in lst_columns:
#     df_final[information] = df_final[information].astype(int)
#     # Sort by Datetime
# df_final = df_final.sort_values('Datetime')
#     # Reset index
# df_final = df_final.reset_index(drop=True)
# diff_copy = df_final.copy()

# # Calculate difference
# diff_copy["diff_abs"] = diff_copy["E_total"]- diff_copy["A_total"]
# diff_copy["diff_rel"] = (diff_copy["E_total"]- diff_copy["A_total"])/(diff_copy["E_total"])


# Manual Overwrite
#Backup
big_day_step_2_backup = big_day_step_2.copy()

big_day_step_2['E8'] =  0.003*big_day_step_2_backup['E8']
big_day_step_2['A4'] = 0.003* big_day_step_2_backup['A4']

big_day_step_2['E14'] = 0.004*big_day_step_2_backup['E14']
big_day_step_2['A9'] = 0.005* big_day_step_2_backup['A9']

big_day_step_2['E15'] = 0.015*big_day_step_2_backup['E15']
big_day_step_2['A10'] = 0.015*big_day_step_2_backup['A10']

big_day_step_2['E16'] = 0.04* big_day_step_2_backup['E16']
big_day_step_2['A11'] = 0.04* big_day_step_2_backup['A11']

big_day_step_2['E17'] = 0.025*big_day_step_2_backup['E17']
big_day_step_2['A12'] = 0.025*big_day_step_2_backup['A12']

big_day_step_2['E18'] = 0.06* big_day_step_2_backup['E18']
big_day_step_2['A13'] = 0.06* big_day_step_2_backup['A13']

big_day_step_2['A14'] = 0.04* big_day_step_2_backup['A14']

big_day_step_2['E20'] = 0.047* big_day_step_2_backup['E20']
big_day_step_2['A15'] = 0.047*big_day_step_2_backup['A15']

big_day_step_2['A16'] = 0.89*big_day_step_2_backup['A16']
big_day_step_2['A17'] = 0.047*big_day_step_2_backup['A17']
# big_day_step_2['A18'] = 1.084*big_day_step_2_backup['A18']

# diff_E8 = sum(big_day_step_2['E8'])-sum(big_day_step_2_backup['E8'])
# diff_E14 = sum(big_day_step_2['E14'])-sum(big_day_step_2_backup['E14'])
# diff_E15 = sum(big_day_step_2['E15'])-sum(big_day_step_2_backup['E15'])
# diff_E16 = sum(big_day_step_2['E16'])-sum(big_day_step_2_backup['E16'])
# diff_E17 = sum(big_day_step_2['E17'])-sum(big_day_step_2_backup['E17'])
# diff_E18 = sum(big_day_step_2['E18'])-sum(big_day_step_2_backup['E18'])
# diff_E20 = sum(big_day_step_2['E20'])-sum(big_day_step_2_backup['E20'])

# factor_A4 = 1+(diff_E8/sum(big_day_step_2['A4']))
# factor_A9 = 1+(diff_E14/sum(big_day_step_2['A9']))
# factor_A10 = 1+(diff_E15/sum(big_day_step_2['A10']))
# factor_A11 = 1+(diff_E16/sum(big_day_step_2['A11']))
# factor_A12 = 1+(diff_E17/sum(big_day_step_2['A12']))
# factor_A13 = 1+(diff_E18/sum(big_day_step_2['A13']))
# factor_A15 = 1+(diff_E20/sum(big_day_step_2['A15']))

# big_day_step_2['A4'] = factor_A4*big_day_step_2['A4']
# big_day_step_2['A9'] = factor_A9*big_day_step_2['A9']
# big_day_step_2['A10'] = factor_A10*big_day_step_2['A10']
# big_day_step_2['A11'] = factor_A11*big_day_step_2['A11']
# big_day_step_2['A12'] = factor_A12*big_day_step_2['A12']
# big_day_step_2['A13'] = factor_A13*big_day_step_2['A13']
# big_day_step_2['A15'] = factor_A15*big_day_step_2['A15']

# Reset
# big_day_step_2['E8'] =  big_day_step_2_backup['E8']
# big_day_step_2['A4'] = big_day_step_2_backup['A4']

# big_day_step_2['E14'] = big_day_step_2_backup['E14']
# big_day_step_2['A9'] = big_day_step_2_backup['A9']

# big_day_step_2['E15'] = big_day_step_2_backup['E15']
# big_day_step_2['A10'] = big_day_step_2_backup['A10']

# big_day_step_2['E16'] = big_day_step_2_backup['E16']
# big_day_step_2['A11'] = big_day_step_2_backup['A11']

# big_day_step_2['E17'] = big_day_step_2_backup['E17']
# big_day_step_2['A12'] = big_day_step_2_backup['A12']

# big_day_step_2['E18'] = big_day_step_2_backup['E18']
# big_day_step_2['A13'] = big_day_step_2_backup['A13']

# big_day_step_2['A14'] = big_day_step_2_backup['A14']

# big_day_step_2['E20'] = big_day_step_2_backup['E20']
# big_day_step_2['A15'] = big_day_step_2_backup['A15']

# big_day_step_2['A17'] = big_day_step_2_backup['A17']

# Check
# list_intersections = ['Whole', 'ES217', 'ES216', 'ES215', 'DarkHole', 'ES213', 'ES235', 'Outside']
list_entry_vectors = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', 'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20', 'E21', 'E22', 'E23', 'E24']
list_exit_vectors = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16', 'A17', 'A18']

# list_entry_217 = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'I1', 'I2']
# list_exit_217 = ['A1', 'A2', 'A3', 'L1']

# list_entry_216 = ['E9', 'E10', 'E11', 'I3', 'I4', 'I5']
# list_exit_216 = ['A5', 'A6', 'L2_3', 'L4']

# list_entry_215 = ['E12', 'E13', 'I6', 'I7', 'I8', 'I9']
# list_exit_215 = ['A7', 'A8', 'L5', 'L6']

# list_entry_DH = ['I10', 'E15', 'E16', 'E17', 'E18', 'E19', 'L11']
# list_exit_DH = ['L7', 'A10', 'A11', 'A12', 'A13', 'A14', 'I15']

# list_entry_213 = ['E20', 'I15', 'I16']
# list_exit_213 = ['L11', 'A15', 'L12']

# list_entry_235 = ['E21', 'E22', 'E23', 'E24', 'I17', 'I18']
# list_exit_235 = ['A16', 'A17', 'A18', 'L13']

# list_entry_out = ['E8', 'E14', 'E16']
# list_exit_out = ['A4', 'A9', 'A11']

# list_entries_list =[list_entry_vectors, list_entry_217, list_entry_216, list_entry_215, list_entry_DH, list_entry_213, list_entry_235, list_entry_out]
# list_exits_list =[list_exit_vectors, list_exit_217, list_exit_216, list_exit_215, list_exit_DH, list_exit_213, list_exit_235, list_exit_out]

# total_total_sum = 0
# for name, entry_list, exit_list in zip(list_intersections, list_entries_list, list_exits_list):
#     total_check_entry = 0
#     total_check_exit = 0
#     total_check_sum = 0
#     for vector in entry_list:
#         total_check_entry += sum(big_day_step_2[vector])
#     for vector in exit_list:
#         total_check_exit += sum(big_day_step_2[vector])
#     total_check_sum = total_check_entry-total_check_exit
#     print(name + ": " + str(total_check_sum))
#     if name != 'Whole':
#         total_total_sum += total_check_sum



total_entry = 0
for vector in list_entry_vectors:
    print(vector + ": "+ str(sum(big_day_step_2[vector])))
    total_entry += sum(big_day_step_2[vector])
total_exit = 0
for vector in list_exit_vectors:
    print(vector + ": "+ str(sum(big_day_step_2[vector])))
    total_exit += sum(big_day_step_2[vector])

print("Entry: "+str(total_entry))
print("Exit: "+str(total_exit))    
print("Rate based on Entry: "+str((total_entry-total_exit)/total_entry))
    
    
# #%% Check Percentage
# column_names = big_day_step_2.columns[1:].tolist()
# e_entries = [entry for entry in column_names if entry.startswith("E")]
# # List of entries starting with 'A'
# a_entries = [entry for entry in column_names if entry.startswith("A")]
# entry_comparison_df = pd.DataFrame({'Entry Vectors': e_entries})
# exit_comparison_df = pd.DataFrame({'Exit Vectors': a_entries})

# temp_list = []
# for col in e_entries:
#     temp_list.append(sum(big_day_step_2[col]))
# entry_comparison_df['Amount of Cars'] = temp_list
# max_score_index = entry_comparison_df['Amount of Cars'].idxmax()
# name_with_max_score = entry_comparison_df.loc[max_score_index, 'Entry Vectors']
# temp_list = []
# for col in e_entries:
#     temp_list.append(sum(big_day_step_2[col])/sum(big_day_step_2[name_with_max_score]))
# entry_comparison_df['% with Base:'+f'{name_with_max_score}'] = temp_list


# temp_list = []
# for col in a_entries:
#     temp_list.append(sum(big_day_step_2[col]))
# exit_comparison_df['Amount of Cars'] = temp_list
# max_score_index = exit_comparison_df['Amount of Cars'].idxmax()
# name_with_max_score = exit_comparison_df.loc[max_score_index, 'Exit Vectors']
# temp_list = []
# for col in a_entries:
#     temp_list.append(sum(big_day_step_2[col])/sum(big_day_step_2[name_with_max_score]))
# exit_comparison_df['% with Base:'+f'{name_with_max_score}'] = temp_list

#%% Compare differences Before and After
# list_manual_changes = ['A4', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14', 'A15', 'A17', 'E8', 'E14', 'E15', 'E16', 'E17', 'E18', 'E20']
# list_manual_changes = ['E8', 'A4', 'E14', 'A9', 'E15', 'A10', 'E16', 'A11', 'E17', 'A12', 'E18', 'A13', 'E19', 'A14', 'E20', 'A15', 'E23', 'A17']
# for vector in list_manual_changes:
#     print(vector + ": " + str(sum(big_day_step_2[vector]) - sum(big_day_step_2_backup[vector])))


# big_day_step_2['E8'] = big_day_step_2['E19']
# big_day_step_2['E14'] = 0.6*big_day_step_2['E8']
# big_day_step_2['E15'] = 1.2*big_day_step_2['E8']
# big_day_step_2['E16'] = 1.1*big_day_step_2['E8']
# big_day_step_2['E17'] = 1.2*big_day_step_2['E16']
# big_day_step_2['E18'] = 1.1*big_day_step_2['E17']
# big_day_step_2['E20'] = 0.8*big_day_step_2['E22']

# # big_day_step_2['A17'] = 0.15*big_day_step_2['A17']
# big_day_step_2['A4'] = big_day_step_2['A17']
# big_day_step_2['A9'] = 0.2*big_day_step_2['A8']
# big_day_step_2['A10'] = 0.3*big_day_step_2['A8']
# big_day_step_2['A11'] = 1.1*big_day_step_2['A4']
# big_day_step_2['A12'] = 1.1*big_day_step_2['A11']
# big_day_step_2['A13'] = 1.2*big_day_step_2['A11']
# big_day_step_2['A14'] = 0.8*big_day_step_2['A17']
# big_day_step_2['A15'] = 1.2*big_day_step_2['A8']


#%% Check Aggregation
big_day_step_2_aggregated = big_day_step_2.copy()
big_day_step_2_aggregated.set_index("Datetime", inplace=True)
big_day_step_2_aggregated = big_day_step_2_aggregated.resample("15T").sum()


# Save CSV for each Vector
list_vectors_to_csv = big_day_step_2.columns.tolist()
list_vectors_to_csv.remove("Datetime")
for vector in list_vectors_to_csv:
    vector_csv = big_day_step_2[['Datetime', vector]]
    if vector.startswith('A'):
        vector_csv.to_csv(path_ausgang_files+vector+".csv", sep=";")
    elif vector.startswith('E'):    
        vector_csv.to_csv(path_eingang_files+vector+".csv", sep=";")
    else:
        vector_csv.to_csv(path_rest_files+vector+".csv", sep=";")