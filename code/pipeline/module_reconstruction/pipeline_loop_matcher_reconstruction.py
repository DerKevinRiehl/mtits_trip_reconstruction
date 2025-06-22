###############################################################################
# ######### IMPORTS ###########################################################
###############################################################################
import numpy as np
import pandas as pd
import ast
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")




###############################################################################
# ######### METHODS ###########################################################
###############################################################################

######### LOADING
def load_ground_truth_routes(ground_truth_file):
    df_ground_truth = pd.read_csv(ground_truth_file)
    df_ground_truth = df_ground_truth.rename(columns={"Unnamed: 0": "Vehicle_ID"})
    df_ground_truth = df_ground_truth[["Vehicle_ID", "Adjusted_Datetime", "entrance", "route"]]
    df_ground_truth["Car"] = "Car_" + df_ground_truth["Vehicle_ID"].astype(str)
    df_ground_truth = df_ground_truth[["Car", "route"]].drop_duplicates()
    return df_ground_truth

def load_loop_detector_events(log_folder, df_ground_truth):
    detector_lst = os.listdir(log_folder)
    loop_events = []
    for detector in detector_lst:
        with open(os.path.join(log_folder, detector), "r") as f:
            content = f.read()
        lines = [line.strip() for line in content.split("\n") if line.strip().startswith("<instantOut") and "enter" in line]
        for line in lines:
            time = float(line.split("time=\"")[1].split("\"")[0])
            event_id = line.split("vehID=\"")[1].split("\"")[0]
            loop_events.append([time, detector.replace('.xml', ''), event_id])
    df_loop_events = pd.DataFrame(loop_events, columns=["time", "detector", "true_event_id"])
    df_loop_events = df_loop_events.merge(df_ground_truth, left_on="true_event_id", right_on="Car", how="left")
    df_loop_events = df_loop_events.drop(columns=['Car'])
    df_loop_events["unique_event_id"] = range(1, len(df_loop_events) + 1)
    df_loop_events["reconstructed_route"] = "?"
    df_loop_events["reconstructed_status"] = "?"
    df_loop_events["matched_event_ids"] = "?"
    df_loop_events['time'] = pd.to_numeric(df_loop_events['time'])
    df_loop_events['Datetime'] = df_loop_events['time'].apply(lambda x: simulation_time_start + timedelta(seconds=x))
    return df_loop_events

def loadODMatrices(path_OD_matrix):
    od_matrices = {}
    files = os.listdir(path_OD_matrix)
    for file in files:
        od_matrices[file.replace(".csv","")] = np.loadtxt(path_OD_matrix+file, delimiter=',')
    return od_matrices

######### LAMBDA FUNCTIONS
def direct_matching_perimeter_detectors(row):
    if row["reconstructed_status"] != "Done":
        if row["detector"] in direct_matching_perimeter_detectors_map:
            row["reconstructed_route"] = direct_matching_perimeter_detectors_map[row["detector"]][0]
            row["reconstructed_status"] = direct_matching_perimeter_detectors_map[row["detector"]][1]
            row["matched_event_ids"] = str([row["unique_event_id"]])
    return row

def direct_matching_internal_perimeter_detectors(row):
    if not row["End"]:
        last_flow = row["Routes"].split("-")[-1]
        if last_flow in direct_matching_internal_perimeter_detectors_map:
            row["Routes"] = row["Routes"] + "-" + direct_matching_internal_perimeter_detectors_map[last_flow]
            if direct_matching_internal_perimeter_detectors_map[last_flow] in perimeter_exit_flows:
                row["End"] = True
            row["Complete"] = row["Begin"] & row["End"]
    return row

######### D2D DELAY DISTRIBUTION
def determine_d2d_delay_distribution(df_loops, from_detector, to_detector):
    df_loops_sub= df_loops[df_loops['detector'] == from_detector]
    unique_event_ids = df_loops_sub['true_event_id'].unique()
    vehicle_pairs = []
    for event_id in unique_event_ids:
        veh_rows = df_loops[df_loops['true_event_id'] == event_id]
        if len(veh_rows) == 2 and set(veh_rows['detector']) == set([from_detector, to_detector]):
            vehicle_pairs.append(veh_rows)
    if len(vehicle_pairs)==0:
        return -1, -1, -1
    time_deltas = []
    for pair in vehicle_pairs:
        time_delta = (pair['Datetime'].max() - pair['Datetime'].min()).total_seconds()
        time_deltas.append(time_delta)
    median = np.median(time_deltas)
    t_min = np.min(time_deltas)
    t_max = np.max(time_deltas)
    t_desired = median
    return t_min, t_max, t_desired

def determine_d2d_delays(df_loops, from_detectors, to_detectors):
    delays = {}
    for to_detector in to_detectors:
        results = []
        for from_detector in from_detectors:
            if not from_detector.startswith("NO_DETECTOR"):
                t_min, t_max, t_desired = determine_d2d_delay_distribution(df_loops, from_detector, to_detector)
                results.append({
                    'detector': from_detector,
                    't_min': t_min,
                    't_max': t_max,
                    't_desired': t_desired
                })
        df_delays = pd.DataFrame(results)
        delays[to_detector] = df_delays
    return delays

######### MATCHING FUNCTIONS
def conduct_matching_group(label, internal_match_group, df_loop_events, do_scoring, do_max=True):
    # Prepare Matching
    detectors_cer, detectors_unc, detectors_out, detectors_rel, signals_rel, input_detectors, df_loops_sub, d2d_delays = prepare_matching(internal_match_group)
    # Do Matching algorithm
    unmatched_filter = []
    if "unmatched_filter" in internal_match_group:
        unmatched_filter = internal_match_group["unmatched_filter"]
    lst_matched, lst_unmatched_cer, lst_unmatched_unc, output_series = match_internal_flow_group(internal_match_group, df_loops_sub, d2d_delays, detectors_cer, detectors_unc, detectors_out, do_scoring, do_max, unmatched_filter)
    print(">>", label)
    print("\tMatching done", "; Matched:", len(lst_matched), "; Unmatched Certains:", len(lst_unmatched_cer), "; Unmatched Uncertains:", len(lst_unmatched_unc))
    # Do Registration Of Matches in Loop Events
        # Register Every Match
    df_loop_events = register_internal_matches(df_loop_events, lst_matched, internal_match_group)
        # Distribute Unmatched Uncertains
    df_loop_events = register_unmatched_uncertains(df_loop_events, lst_unmatched_unc, internal_match_group)
        # Register Unmatched Destinations
    df_loop_events = register_unmatched_certains(df_loop_events, lst_unmatched_cer, internal_match_group)
    # Evaluate Matching
    share_correct_match, share_correct_turn, share_correct_cer, share_correct_unc, median_correct, std_correct = evaluate_internal_matching(df_loops_sub, output_series, lst_matched)
    return df_loop_events, lst_matched

def prepare_matching(internal_match_group):
    # Determine relevant Lists
    detectors_cer = [*list(internal_match_group["sources_cer"].keys())]
    detectors_unc = [*list(internal_match_group["sources_unc"].keys())]
    detectors_out = internal_match_group["destination"]
    detectors_rel = [*detectors_cer, *detectors_unc, *detectors_out]
    signals_rel = [*list(internal_match_group["sources_cer"].values()), *list(internal_match_group["sources_unc"].values())]
    input_detectors = [*detectors_cer, *detectors_unc]
    # Determine relevant subset for internal matching
    df_loops_sub = df_loop_events[df_loop_events["detector"].isin(detectors_rel)]
    # Determine time delay distribution
    d2d_delays = determine_d2d_delays(df_loops_sub, input_detectors, detectors_out)
    return detectors_cer, detectors_unc, detectors_out, detectors_rel, signals_rel, input_detectors, df_loops_sub, d2d_delays 
    
def determine_candidates(df_loops_sub, delay_dict, lst_detectors, unmatched_filter):
    candidate_series = df_loops_sub[df_loops_sub["detector"].isin(lst_detectors)]
    candidate_series = candidate_series[(candidate_series["matched_event_ids"] == "?") & (candidate_series["matched_event_ids"] != "Done")]
    candidate_series = candidate_series[["Datetime", "detector", "true_event_id", "unique_event_id"]]
    candidate_series['t_min'] = candidate_series['detector'].map(delay_dict['t_min'])
    candidate_series['t_max'] = candidate_series['detector'].map(delay_dict['t_max'])
    candidate_series['t_desired'] = candidate_series['detector'].map(delay_dict['t_desired'])
    candidate_series["Datetime_min_possible_arrival"] = candidate_series["Datetime"]+pd.to_timedelta(candidate_series["t_min"], unit='s')
    candidate_series["Datetime_max_possible_arrival"] = candidate_series["Datetime"]+pd.to_timedelta(candidate_series["t_max"], unit='s')
    candidate_series = candidate_series.sort_values(by="Datetime", ascending=True)
    return candidate_series

def match_internal_flow_group(internal_match_group, df_loops_sub, d2d_delays, detectors_cer, detectors_unc, detectors_out, do_scoring, do_max, unmatched_filter=[]):
    lst_matched = []
    lst_unmatched_cer = []
    lst_unmatched_unc = df_loops_sub[df_loops_sub["detector"].isin(detectors_unc)]["unique_event_id"].tolist()
    # Prepare Candidates and Output Series
    candidates_dict = {}
    for detector_out in detectors_out:
        df_delays = d2d_delays[detector_out]
        delay_dict = df_delays.set_index('detector').to_dict()    
        candidate_series_cer = determine_candidates(df_loops_sub, delay_dict, detectors_cer, unmatched_filter)
        candidate_series_unc = determine_candidates(df_loops_sub, delay_dict, detectors_unc, unmatched_filter)
        candidates_dict[detector_out] = {"cer": candidate_series_cer, "unc": candidate_series_unc}
    output_series = df_loops_sub[df_loops_sub["detector"].isin(detectors_out)]
    output_series = output_series[["Datetime", "detector", "true_event_id", "unique_event_id"]]
    output_series = output_series.sort_values(by="Datetime", ascending=True)
    # Chronologically match each event
    for idx, row in output_series.iterrows():
        detector_out = row["detector"]
        candidate_series_cer = candidates_dict[detector_out]["cer"]
        candidate_series_unc = candidates_dict[detector_out]["unc"]
        # Certain Candidates
        candidates = candidate_series_cer.copy()
        candidates = candidates[candidates["Datetime_min_possible_arrival"]<=row["Datetime"]]
        if do_max:
            candidates = candidates[candidates["Datetime_max_possible_arrival"]>=row["Datetime"]]
        if len(candidates)>0:
            candidates["score"] = 0
            if do_scoring:
                candidates["time"] = candidates["Datetime_min_possible_arrival"]+pd.to_timedelta(candidate_series_cer["t_desired"]-candidate_series_cer["t_min"], unit='s')
                candidates["score"] = (candidates["time"] - row["Datetime"]).abs().dt.total_seconds()
                candidates = candidates.sort_values(by="score", ascending=True)
            selected_candidate = candidates.iloc[0]
            lst_matched.append([row["unique_event_id"], selected_candidate["unique_event_id"], selected_candidate["score"], row["detector"], selected_candidate["detector"]])
            # delete candidate from further consideration
            for detector_out in detectors_out:
                candidates_dict[detector_out]["cer"] = candidates_dict[detector_out]["cer"][
                    candidates_dict[detector_out]["cer"]["unique_event_id"] != selected_candidate["unique_event_id"]
                ]
        # Uncertain Candidates
        else:
            candidates = candidate_series_unc.copy()
            candidates = candidates[candidates["Datetime_min_possible_arrival"]<=row["Datetime"]]
            if do_max:
                candidates = candidates[candidates["Datetime_max_possible_arrival"]>=row["Datetime"]]
            if len(candidates)>0:
                candidates["time"] = candidates["Datetime_min_possible_arrival"]+pd.to_timedelta(candidate_series_unc["t_desired"]-candidate_series_unc["t_min"], unit='s')
                candidates["score"] = (candidates["time"] - row["Datetime"]).abs().dt.total_seconds()
                candidates = candidates.sort_values(by="score", ascending=True)
                selected_candidate = candidates.iloc[0]
                lst_matched.append([row["unique_event_id"], selected_candidate["unique_event_id"], selected_candidate["score"], row["detector"], selected_candidate["detector"]])
                # delete candidate from further consideration
                for detector_out in detectors_out:
                    candidates_dict[detector_out]["unc"] = candidates_dict[detector_out]["unc"][
                        candidates_dict[detector_out]["unc"]["unique_event_id"] != selected_candidate["unique_event_id"]
                    ]
                if selected_candidate["unique_event_id"] in lst_unmatched_unc:
                    lst_unmatched_unc.remove(selected_candidate["unique_event_id"])
            else:
                lst_unmatched_cer.append(row["unique_event_id"])
    return lst_matched, lst_unmatched_cer, lst_unmatched_unc, output_series

def register_internal_matches(df_loop_events, lst_matched, internal_match_group):
    for match in lst_matched:
        # Setup all information
        source_event_id = match[1]
        source_flow = internal_match_group["sources_flow"][match[4]]
        destination_event_id = match[0]
        destination_flow = internal_match_group["destination_flow"][match[3]]
        row_source = df_loop_events.loc[df_loop_events['unique_event_id']==source_event_id]
        row_destination = df_loop_events.loc[df_loop_events['unique_event_id']==destination_event_id]    
        # Emergency break
        if row_source["reconstructed_status"].iloc[0]=="Done" or row_destination["reconstructed_status"].iloc[0]=="Done":
            print("ERROR: TRY TO OVERRIDE ALREADY MATCHED VEHICLE")
            print(row_source)
            print(row_destination)
            continue
        # Populate Source Row
            # Column 'matched_event_ids'
        matched_event_ids = row_source["matched_event_ids"].iloc[0]
        if matched_event_ids=="?":
            matched_event_ids = str([destination_event_id])
        else:
            list_from_string = ast.literal_eval(matched_event_ids.replace("@", ","))
            list_from_string.append(destination_event_id)
            matched_event_ids = str(list_from_string).replace(",", "@")
        df_loop_events.loc[df_loop_events['unique_event_id'] == source_event_id, 'matched_event_ids'] = matched_event_ids
            # Column 'reconstructed_status'
        reconstructed_status = row_source["reconstructed_status"].iloc[0]
        if reconstructed_status=="?":
            reconstructed_status = source_flow+"-"+destination_flow
        else:
            reconstructed_status = reconstructed_status+"-"+destination_flow
        if destination_flow in perimeter_exit_flows:
            df_loop_events.loc[df_loop_events['unique_event_id'] == source_event_id, 'reconstructed_route'] = reconstructed_status
            df_loop_events.loc[df_loop_events['unique_event_id'] == source_event_id, 'reconstructed_status'] = "Done"
        else:
            df_loop_events.loc[df_loop_events['unique_event_id'] == source_event_id, 'reconstructed_status'] = reconstructed_status
        # Populate Destination Row
            # Column 'matched_event_ids'
        matched_event_ids = row_destination["matched_event_ids"].iloc[0]
        if matched_event_ids=="?":
            matched_event_ids = str([source_event_id])
        else:
            list_from_string = ast.literal_eval(matched_event_ids.replace("@", ","))
            list_from_string.append(source_event_id)
            matched_event_ids = str(list_from_string).replace(",", "@")
        df_loop_events.loc[df_loop_events['unique_event_id'] == destination_event_id, 'matched_event_ids'] = matched_event_ids
            # Column 'reconstructed_status'
        reconstructed_status = row_destination["reconstructed_status"].iloc[0]
        if reconstructed_status=="?":
            reconstructed_status = source_flow+"-"+destination_flow
        else:
            reconstructed_status = reconstructed_status+"-"+destination_flow
        if destination_flow in perimeter_exit_flows:
            df_loop_events.loc[df_loop_events['unique_event_id'] == destination_event_id, 'reconstructed_route'] = reconstructed_status
            df_loop_events.loc[df_loop_events['unique_event_id'] == destination_event_id, 'reconstructed_status'] = "Done"
        else:
            df_loop_events.loc[df_loop_events['unique_event_id'] == destination_event_id, 'reconstructed_status'] = reconstructed_status
    return df_loop_events

def register_unmatched_uncertains(df_loop_events, lst_unmatched_uncertains, internal_match_group):
    for event_id in lst_unmatched_uncertains:
        # Setup all information
        source_detector = df_loop_events[df_loop_events["unique_event_id"]==event_id]["detector"].iloc[0]
        source_event_id = event_id
        source_flow = internal_match_group["sources_flow"][source_detector]
        if not source_detector in internal_match_group["destination_flow_unc_alternative"]:
            continue
        destination_flow = internal_match_group["destination_flow_unc_alternative"][source_detector]
        row_source = df_loop_events.loc[df_loop_events['unique_event_id']==source_event_id]
        # Emergency break
        if row_source["reconstructed_status"].iloc[0]=="Done":
            print("ERROR: TRY TO OVERRIDE ALREADY MATCHED VEHICLE")
            print(row_source)
            continue
        # Populate Source Row
            # Column 'reconstructed_status'
        reconstructed_status = row_source["reconstructed_status"].iloc[0]
        if reconstructed_status=="?":
            reconstructed_status = source_flow+"-"+destination_flow
        else:
            reconstructed_status = reconstructed_status+"-"+destination_flow
        if destination_flow in perimeter_exit_flows:
            df_loop_events.loc[df_loop_events['unique_event_id'] == source_event_id, 'reconstructed_route'] = reconstructed_status
            df_loop_events.loc[df_loop_events['unique_event_id'] == source_event_id, 'reconstructed_status'] = "Done"
        else:
            df_loop_events.loc[df_loop_events['unique_event_id'] == source_event_id, 'reconstructed_status'] = reconstructed_status
    return df_loop_events

def register_unmatched_certains(df_loop_events, lst_unmatched, internal_match_group):
    if "NO_DETECTOR" in internal_match_group["sources_cer"]:
        for event_id in lst_unmatched:
            # Setup all information
            destination_detector = df_loop_events[df_loop_events["unique_event_id"]==event_id]["detector"].iloc[0]
            destination_event_id = event_id
            source_detector = "NO_DETECTOR"
            source_flow = internal_match_group["sources_flow"][source_detector]
            destination_flow = internal_match_group["destination_flow"][destination_detector]
            row_destination = df_loop_events.loc[df_loop_events['unique_event_id']==destination_event_id]
            # Emergency break
            if row_destination["reconstructed_status"].iloc[0]=="Done":
                print("ERROR: TRY TO OVERRIDE ALREADY MATCHED VEHICLE")
                print(row_destination)
                continue
            # Populate Destination Row
                # Column 'reconstructed_status'
            reconstructed_status = row_destination["reconstructed_status"].iloc[0]
            if reconstructed_status=="?":
                reconstructed_status = source_flow+"-"+destination_flow
            else:
                reconstructed_status = source_flow+"-"+reconstructed_status
            if destination_flow in perimeter_exit_flows:
                df_loop_events.loc[df_loop_events['unique_event_id'] == destination_event_id, 'reconstructed_route'] = reconstructed_status
                df_loop_events.loc[df_loop_events['unique_event_id'] == destination_event_id, 'reconstructed_status'] = "Done"
            else:
                df_loop_events.loc[df_loop_events['unique_event_id'] == destination_event_id, 'reconstructed_status'] = reconstructed_status
    return df_loop_events
   
def evaluate_internal_matching(df_loops_sub, output_series, lst_matched):
    # DETERMINE GROUND TRUTH FOR EVALUATION
    df_evaluation_lookup = df_loops_sub[["unique_event_id", "true_event_id", "route"]]
    if len(lst_matched)==0:
        return -1, -1, -1, -1, -1, -1
    matches_records_df = pd.DataFrame(np.asarray(lst_matched), columns =["destination", "source", "confidence", "detector_des", "detector_src"])
    matches_records_df['destination'] = matches_records_df['destination'].astype(int)
    matches_records_df['source'] = matches_records_df['source'].astype(int)
    matches_records_df['confidence'] = matches_records_df['confidence'].astype(float)
    # Evaluate Matching
    output_series = output_series.merge(matches_records_df, left_on="unique_event_id", right_on="destination", how="left")
    eval_df = output_series.merge(df_evaluation_lookup, left_on="source", right_on="unique_event_id", how="left")
    eval_df["correct"] = eval_df["true_event_id_x"]==eval_df["true_event_id_y"]
    eval_df_a = eval_df[eval_df["confidence"]==0]
    eval_df_b = eval_df[eval_df["confidence"]!=0]
    share_correct_match = sum(eval_df["correct"])/len(eval_df["correct"])
    share_correct_cer = -1
    share_correct_unc = -1
    if len(eval_df_a["correct"])!= 0:
        share_correct_cer = sum(eval_df_a["correct"])/len(eval_df_a["correct"])
    if len(eval_df_b["correct"])!= 0:
        share_correct_unc = sum(eval_df_b["correct"])/len(eval_df_b["correct"])
    eval_df2 = eval_df.copy()
    eval_df2['cumulative_correct'] = eval_df2['correct'].cumsum()
    eval_df2['cumulative_total'] = range(1, len(eval_df2) + 1)
    eval_df2['share_correct'] = eval_df2['cumulative_correct'] / eval_df2['cumulative_total']
    eval_df2 = eval_df2.sort_values('Datetime')
    eval_df2['Datetime'] = pd.to_datetime(eval_df2['Datetime'])
    def share_correct_window(group):
        return sum(group) / len(group)
    df_indexed = eval_df2.set_index('Datetime')
    rolling_result = df_indexed['correct'].rolling('900S').apply(share_correct_window)
    eval_df2['share_correct_window'] = rolling_result.values
    eval_df2['share_correct_window'] = eval_df2['share_correct_window'].fillna(eval_df2['share_correct'])
    # Evaluate Turns
    eval_df3 = matches_records_df.merge(df_evaluation_lookup, left_on="source", right_on="unique_event_id", how="left")
    share_correct_turn = -1
    if len(eval_df3)!= 0:
        correct_turns_lst = []
        for idx, row in eval_df3.iterrows():
            df_selection = df_loop_events[(df_loop_events["detector"]==row["detector_des"]) & (df_loop_events["true_event_id"]==row["true_event_id"])]
            correct_turns_lst.append(len(df_selection)>0)
        eval_df3["correct_turn"] = correct_turns_lst
        share_correct_turn = np.mean(eval_df3["correct_turn"])
    # Print Evaluation results
    print("\tEvaluation:", share_correct_match, share_correct_turn, share_correct_cer, share_correct_unc, ";", np.median(eval_df2['share_correct_window']), np.std(eval_df2['share_correct_window']))
    return share_correct_match, share_correct_turn, share_correct_cer, share_correct_unc, np.median(eval_df2['share_correct_window']), np.std(eval_df2['share_correct_window'])

# RECONSTRUCTION METHODS
def reconstruct_trips(df_loop_events):
    def getListOfIDs(matched_event_ids_str):
        list_from_string = ast.literal_eval(matched_event_ids_str.replace("@", ","))
        return list_from_string
    captured_index = []
    trip_event_row_group = []
    for idx, row in df_loop_events.iterrows():
        if idx not in captured_index:
            if row["matched_event_ids"]!="?":
                trip_rows = []
                id_list = [row["unique_event_id"]]
                id_list += getListOfIDs(matched_event_ids_str=row["matched_event_ids"])
                trip_rows.append(df_loop_events[df_loop_events["unique_event_id"]==row["unique_event_id"]])
                while True:
                    prev_len = len(id_list)
                    for v_id in id_list:
                        new_row = df_loop_events[df_loop_events["unique_event_id"]==v_id]
                        trip_rows.append(new_row)
                        new_id_list = getListOfIDs(matched_event_ids_str=new_row["matched_event_ids"].iloc[0])
                        for n_v_id in new_id_list:
                            if n_v_id not in id_list:
                                id_list.append(n_v_id)
                    if len(id_list)==prev_len:
                        break
                trip_rows = pd.concat(trip_rows, axis=0)
                trip_rows = trip_rows.drop_duplicates()
                trip_event_row_group.append(trip_rows)
                captured_index += trip_rows.index.tolist()
    trip_data = []
    for trip_rows in trip_event_row_group:
        df = trip_rows.copy()
        df = df.sort_values(by='Datetime', ascending=True)
        routes = ""
        times = df["time"].tolist()
        times = [t-times[0] for t in times]
        starttime = str(df["Datetime"].iloc[0])
        true_event_id = df["true_event_id"].iloc[0]
        true_route = df["route"].iloc[0]
        unique_event_ids = []
        for idx, row in df.iterrows():
            unique_event_ids.append(row["unique_event_id"])
            if row["reconstructed_status"]=="Done":
                routes = row["reconstructed_route"]
            else:
                if routes=="":
                    routes = "-"+row["reconstructed_status"]
                else:
                    parts = row["reconstructed_status"].split("-")
                    for part in parts:
                        if "-"+part not in routes:
                            routes += "-"+part
        if routes.startswith("-"):
            routes = routes[1:]
        completed_begin = False
        completed_end = False
        if routes.split("-")[0] in perimeter_entry_flows:
            completed_begin = True
        if routes.split("-")[-1] in perimeter_exit_flows:
            completed_end = True
        trip_data.append([starttime, routes, str(times), completed_begin, completed_end, true_event_id, true_route, str(unique_event_ids)])
    trip_data = pd.DataFrame(trip_data, columns=["Starttime", "Routes", "Times", "Begin", "End", "true_event_id", "true_route", "unique_event_ids"])
    trip_data["Complete"] = trip_data["Begin"] & trip_data["End"]
    return trip_data

def match_remaining_trips(trip_data, df_loop_events):
    # Determine unmatched starters
    trip_data_starters = trip_data.copy()
    trip_data_starters = trip_data_starters[(trip_data_starters["Begin"]) & (~trip_data_starters["End"])]
    trip_data_starters['last_time'] = trip_data_starters['Times'].apply(lambda x: ast.literal_eval(x)[-1])
    trip_data_starters["Starttime2"] = pd.to_datetime(trip_data_starters['Starttime'], format="mixed")
    trip_data_starters['Endtime'] = trip_data_starters.apply(lambda row: row['Starttime2'] + timedelta(seconds=row['last_time']), axis=1)
    trip_data_starters['Endtime'] = trip_data_starters['Endtime'].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    trip_data_ends = trip_data.copy()
    trip_data_ends = trip_data_ends[(~trip_data_ends["Begin"]) & (trip_data_ends["End"])]
    # Determine closest matches
    lst_matches = []
    lst_match_black_list = []
        # remaining starters
    for idx, row in trip_data_starters.iterrows():
        last_time = ast.literal_eval(row["Times"])[-1]
        start_time = row["Starttime"]
        if "." not in start_time:
            start_time = start_time+".00"
        end_time = (datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S.%f") + timedelta(seconds=last_time)).strftime("%Y-%m-%d %H:%M:%S.%f")
        candidates = trip_data_ends[trip_data_ends["Starttime"]>=end_time]
        candidates["start_flow"] = candidates["Routes"].str.split("-").str[0]
        end_flow = row["Routes"].split("-")[-1]
        candidates = candidates[candidates["start_flow"]==end_flow]
        candidates = candidates[~candidates.apply(lambda row: [row["Starttime"], row["Routes"]] in lst_match_black_list, axis=1)]
        if len(candidates)>0:
            candidates = candidates.sort_values(by="Starttime", ascending=True)
            candidate_selected = candidates.iloc[0]
            lst_matches.append([row["Starttime"], row["Routes"], candidate_selected["Starttime"], candidate_selected["Routes"] ])
            lst_match_black_list.append([ candidate_selected["Starttime"], candidate_selected["Routes"] ])
        # remaining ends
    for idx, row in trip_data_ends.iterrows():
        start_time = row["Starttime"]
        candidates = trip_data_starters[trip_data_starters["Endtime"]<=start_time]
        candidates["end_flow"] = candidates["Routes"].str.split("-").str[-1]
        start_flow = row["Routes"].split("-")[0]
        candidates = candidates[candidates["end_flow"]==start_flow]
        candidates = candidates[~candidates.apply(lambda row: [row["Starttime"], row["Routes"]] in lst_match_black_list, axis=1)]
        if len(candidates)>0:
            candidates = candidates.sort_values(by="Endtime", ascending=False)
            candidate_selected = candidates.iloc[0]
            lst_matches.append([candidate_selected["Starttime"], candidate_selected["Routes"], row["Starttime"], row["Routes"]])
            lst_match_black_list.append([ candidate_selected["Starttime"], candidate_selected["Routes"] ])    
    # Update trip_data & df_loop_events
    trip_data_2 = trip_data.copy()
    df_loop_events_2 = df_loop_events.copy()
    for match in lst_matches:
        # Determine rows
        row_start = trip_data_starters[(trip_data_starters["Starttime"]==match[0]) & (trip_data_starters["Routes"]==match[1])]
        row_end   = trip_data_ends    [(trip_data_ends["Starttime"]==match[2])     & (trip_data_ends["Routes"]==match[3])]
        # Determine unique event ids
        event_ids_start = ast.literal_eval(row_start["unique_event_ids"].iloc[0])
        event_ids_end = ast.literal_eval(row_end["unique_event_ids"].iloc[0])
        event_ids = event_ids_start+event_ids_end
        complete_route = row_start["Routes"].iloc[0] + "-" + "-".join(row_end["Routes"].iloc[0].split("-")[1:])
        # Delete index from trip_data
        trip_data_2 = trip_data_2[~((trip_data_2['Starttime'] == match[0]) & (trip_data_2['Routes'] == match[1]))]
        trip_data_2 = trip_data_2[~((trip_data_2['Starttime'] == match[2]) & (trip_data_2['Routes'] == match[3]))]
        # Add one trip to trip_data
        times_a = ast.literal_eval(row_start["Times"].iloc[0])
        times_b = ast.literal_eval(row_end["Times"].iloc[0])
        times_b = [time + times_a[-1] for time in times_b][1:]
        times_c = times_a+times_b
        new_trip = [
            row_start["Starttime"].iloc[0], # Starttime
            complete_route, # Routes
            str(times_c),# Times
            True, # Begin
            True, # End
            row_start["true_event_id"].iloc[0], # true_event_id
            row_start["true_route"].iloc[0], # true_route
            str(event_ids), # unique_event_ids
            True, # Completete
        ]
        new_row = pd.DataFrame([new_trip], columns=trip_data_2.columns)
        trip_data_2 = pd.concat([trip_data_2, new_row], ignore_index=False)
        # Update df_loop_events
        for event_id in event_ids:
            df_loop_events_2.loc[df_loop_events_2['unique_event_id'] == event_id, 'reconstructed_route'] = complete_route
            df_loop_events_2.loc[df_loop_events_2['unique_event_id'] == event_id, 'reconstructed_status'] = 'Done-X'
            matched_id_list = ast.literal_eval(df_loop_events_2.loc[df_loop_events_2['unique_event_id'] == event_id, 'matched_event_ids'].iloc[0].replace("@", ","))
            matched_id_list += event_ids
            matched_id_list = list(set(matched_id_list))
            matched_id_list = str(matched_id_list).replace(",", "@")
            df_loop_events_2.loc[df_loop_events_2['unique_event_id'] == event_id, 'matched_event_ids'] = matched_id_list     
    return trip_data_2, df_loop_events_2, lst_matches

def get_possible_routes_from_begin(route_path_dict, route_begin):
    candidates = []
    for key in route_path_dict:
        if route_path_dict[key].startswith(route_begin):
            candidates.append(key.split("_")[-1])
    return candidates

def get_possible_routes_from_end(route_path_dict, route_end):
    candidates = []
    for key in route_path_dict:
        if route_path_dict[key].endswith(route_end):
            candidates.append(key.split("_")[1])
    return candidates

def get_possible_routes_from_begin_to_end(route_path_dict, route_begin, route_end):
    candidates = []
    for key in route_path_dict:
        if route_path_dict[key].endswith(route_end) and route_path_dict[key].startswith(route_begin):
            candidates.append(route_path_dict[key])
    return candidates
    
def match_disconnected_remaining_trips(trip_data, df_loop_events):
    trip_data_starters = trip_data.copy()
    trip_data_starters = trip_data_starters[(trip_data_starters["Begin"]) & (~trip_data_starters["End"])]
    trip_data_starters['last_time'] = trip_data_starters['Times'].apply(lambda x: ast.literal_eval(x)[-1])
    trip_data_starters["Starttime2"] = pd.to_datetime(trip_data_starters['Starttime'], format="mixed")
    trip_data_starters['Endtime'] = trip_data_starters.apply(lambda row: row['Starttime2'] + timedelta(seconds=row['last_time']), axis=1)
    trip_data_starters['Endtime'] = trip_data_starters['Endtime'].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    del trip_data_starters["last_time"]
    trip_data_ends = trip_data.copy()
    trip_data_ends = trip_data_ends[(~trip_data_ends["Begin"]) & (trip_data_ends["End"])]
    lst_matches = []
    lst_black_list = []
    # Find disconnected remaining trips
    for idx, row in trip_data_starters.iterrows():
        route_fragment = row["Routes"]
        possible_routes = get_possible_routes_from_begin(route_path_dict, route_fragment)
        # only ends that enable possible route
        candidates = trip_data_ends[trip_data_ends["Routes"].str.split("-").str[-1].isin(possible_routes)]
        candidates = candidates[candidates["Starttime"]>row["Endtime"]]
        # TODO: calculate t_min and add additionally
        candidates = candidates[~candidates.index.isin(lst_black_list)]
        if len(candidates)>0:
            for x in range(0, len(candidates)):
                selected_candidate = candidates.iloc[x]
                if len(get_possible_routes_from_begin_to_end(route_path_dict, route_fragment, selected_candidate["Routes"]))>0:
                    times_a = ast.literal_eval(row["Times"])
                    times_b = ast.literal_eval(selected_candidate["Times"])
                    times_c = str(times_a+[-1]+times_b)
                    event_ids_a = ast.literal_eval(row["unique_event_ids"])
                    event_ids_b = ast.literal_eval(selected_candidate["unique_event_ids"])
                    event_ids_c = str(event_ids_a+event_ids_b)
                    new_complete_trip = [
                        row["Starttime"],
                        get_possible_routes_from_begin_to_end(route_path_dict, route_fragment, selected_candidate["Routes"])[0],
                        times_c,
                        True,
                        True,
                        row["true_event_id"],
                        row["true_route"],
                        event_ids_c,
                        True,
                    ]
                    lst_matches.append([idx, candidates.index[0], new_complete_trip])
                    lst_black_list.append(idx)
                    lst_black_list.append(candidates.index[0])
                    break
    # Update trip_data & df_loop_events
    trip_data_2 = trip_data.copy()
    df_loop_events_2 = df_loop_events.copy()
    # Delete old, fragmented trips
    delete_idx = [match[0] for match in lst_matches]
    delete_idx += [match[1] for match in lst_matches]
    trip_data_2 = trip_data_2.drop(delete_idx, axis=0)
    # Add complete trip
    for match in lst_matches:
        new_row = pd.DataFrame([match[2]], columns=trip_data_2.columns)
        trip_data_2 = pd.concat([trip_data_2, new_row], ignore_index=True)
    # Update Loop Events
    unique_routes = {}
    unique_matches = {}
    for match in lst_matches:
        record = match[2]
        rec_route = record[1]
        ev_ids = ast.literal_eval(record[7])
        for ids in ev_ids:
            if ids not in unique_routes:
                unique_routes[ids] = rec_route
                unique_matches[ids] = str(ev_ids)
    modified_rows = []
    for idx, row in df_loop_events_2.iterrows():
        if row["unique_event_id"] in unique_routes:
            row["reconstructed_status"] = "Done-Y"
            row["reconstructed_route"] = unique_routes[row["unique_event_id"]]
            row["matched_event_ids"] = unique_matches[row["unique_event_id"]]
        modified_rows.append(row)
    df_loop_events_2 = pd.DataFrame(modified_rows)      
    return trip_data_2, df_loop_events_2

def determineCurrentDistributionInSlot(trip_data5, time, search_term, probs, possible_exits):
    trip_data5 = trip_data5.copy()
    if not pd.api.types.is_datetime64_any_dtype(trip_data5['Starttime']):
        trip_data5['Starttime'] = pd.to_datetime(trip_data5['Starttime'], format='mixed')
    mask = (trip_data5['Starttime'].dt.floor(f"{OD_agg}min").dt.strftime("%Y-%m-%d_%H_%M") + "_00" == time) & \
           (trip_data5['Routes'].str.startswith(search_term))
    filtered_data = trip_data5.loc[mask, 'Routes']
    if filtered_data.empty:
        return probs
    exits = filtered_data.str.extract(r'-([^-]+)$')[0]
    exit_counts = exits.value_counts()
    exit_probs = exit_counts.reindex(possible_exits, fill_value=0).values
    exit_probs = exit_probs / exit_probs.sum()
    return exit_probs

def guess_remaining_starter_loop_events(df_loop_events4, trip_data4, od_matrices):
    trip_data5 = trip_data4.copy()
    trip_data5 = trip_data5[trip_data5["Complete"]]
    # Update Loop Events with already completed trips
    df_loop_events5 = df_loop_events4.copy()
    unique_routes = {}
    unique_matches = {}
    for idx, row in trip_data5.iterrows():
        ev_ids = ast.literal_eval(row["unique_event_ids"])
        rec_route = row["Routes"]
        for ids in ev_ids:
            if ids not in unique_routes:
                unique_routes[ids] = rec_route
                unique_matches[ids] = str(ev_ids)
    modified_rows = []
    for idx, row in df_loop_events5.iterrows():
        if row["unique_event_id"] in unique_routes:
            if not row["reconstructed_status"].startswith("Done"):
                row["reconstructed_status"] = "Done-Z1"
                row["reconstructed_route"] = unique_routes[row["unique_event_id"]]
                row["matched_event_ids"] = unique_matches[row["unique_event_id"]]
        modified_rows.append(row)
    df_loop_events5 = pd.DataFrame(modified_rows)    
    # Update Trips with already completed loop events
    lst_all_completed_events = get_list_all_events(trip_data5)
    df_loop_events5_left = df_loop_events5[~df_loop_events5["unique_event_id"].isin(lst_all_completed_events)]
    df_loop_events5_left_enter = df_loop_events5_left[df_loop_events5_left["detector"].isin(entry_detectors)]
    df_loop_events5_left_enter_trips = df_loop_events5_left_enter[df_loop_events5_left_enter["reconstructed_status"].str.startswith("Done")]
    new_trips = []
    for idx, row in df_loop_events5_left_enter_trips.iterrows():
        new_trips.append([
            row["Datetime"],
            row["reconstructed_route"],
            str([0]),
            True,
            True,
            row["true_event_id"],
            row["route"],
            str([row["unique_event_id"]]),
            True
        ])
    for new_trip in new_trips:
        new_row = pd.DataFrame([new_trip], columns=trip_data5.columns)
        trip_data5 = pd.concat([trip_data5, new_row], ignore_index=True)
    # Guess-timate trips of remaining uncompleted entry perimeter loop events (REMAINING STARTER LOOP EVENTS)
    df_loop_events5_left_enter = df_loop_events5_left_enter[~df_loop_events5_left_enter["reconstructed_status"].str.startswith("Done")]
    df_loop_events5_left_enter_gr = df_loop_events5_left_enter.groupby("detector").size().reset_index(name="n")
    df_loop_events5_left_enter_gr['perimeter'] = df_loop_events5_left_enter_gr['detector'].map(entry_detectors)
    lst_completed_loops = {}
    for idx, row in df_loop_events5_left_enter.iterrows():   
        time = str(row["Datetime"].floor(OD_agg+'min')).replace(":","_").replace(" ","_")
        odmatrix = od_matrices[OD_method+time]
        start_entrance = row["reconstructed_status"].split("-")[0]
        search_term = row["reconstructed_status"]
        if start_entrance=="?":
            start_entrance = entry_detectors[row["detector"]]
            search_term = start_entrance
        possible_exits = get_possible_routes_from_begin(route_path_dict, search_term)
        idx = perimeter_entry_flows.index(start_entrance)
        indexes = [perimeter_exit_flows.index(xit) for xit in possible_exits]
        should_probs = odmatrix[idx][indexes]
        should_probs = should_probs/(np.sum(should_probs))
        empiric_probs = determineCurrentDistributionInSlot(trip_data5, time, search_term, should_probs.copy(), possible_exits)
        if (should_probs==empiric_probs).all():
            corrected_probs = should_probs
        else:
            corrected_probs = []
            ctr = 0
            for ex in possible_exits:
                if should_probs[ctr]<empiric_probs[ctr]:
                    corrected_probs.append(0)
                else:
                    corrected_probs.append(should_probs[ctr]-empiric_probs[ctr])
                ctr += 1
            corrected_probs = np.asarray(corrected_probs)
            corrected_probs /= sum(corrected_probs)
        final_probs = corrected_probs
        randomly_selected_exit = possible_exits[np.random.choice(len(final_probs), p=final_probs)]
        randomly_selected_route = route_path_dict["route_"+start_entrance+"_"+randomly_selected_exit]
        new_row = [
            row["Datetime"],
            randomly_selected_route,
            str([]),
            True,
            True,
            row["true_event_id"],
            row["route"],
            str([row["unique_event_id"]]),
            True,
        ]
        new_row = pd.DataFrame([new_row], columns=trip_data5.columns)
        trip_data5 = pd.concat([trip_data5, new_row], ignore_index=True)
        lst_completed_loops[row["unique_event_id"]] = randomly_selected_route
        print(len(trip_data5))
    # Final Update of loop events
    unique_routes = {}
    unique_matches = {}
    for idx, row in trip_data5.iterrows():
        ev_ids = ast.literal_eval(row["unique_event_ids"])
        rec_route = row["Routes"]
        for ids in ev_ids:
            if ids not in unique_routes:
                unique_routes[ids] = rec_route
                unique_matches[ids] = str(ev_ids)
    modified_rows = []
    for idx, row in df_loop_events5.iterrows():
        if row["unique_event_id"] in unique_routes:
            if not row["reconstructed_status"].startswith("Done"):
                row["reconstructed_status"] = "Done-Z2"
                row["reconstructed_route"] = unique_routes[row["unique_event_id"]]
                row["matched_event_ids"] = unique_matches[row["unique_event_id"]]
        modified_rows.append(row)
    df_loop_events5 = pd.DataFrame(modified_rows)    
    return trip_data5, df_loop_events5

# PRINTING METHODS
def get_list_all_events(trip_data):
    lst_all_completed_events = []
    for idx, row in trip_data.iterrows():
        lst_unique_events = ast.literal_eval(row["unique_event_ids"])
        lst_all_completed_events += lst_unique_events
    lst_all_completed_events = list(set(lst_all_completed_events))
    return lst_all_completed_events

def print_trip_statistics(df_loop_events, trip_data):
    print(">> COMPLETED TRIPS", len(trip_data[trip_data["Complete"]]), len(trip_data[trip_data["Complete"]])/len(trip_data), len(trip_data[trip_data["Begin"]])/len(trip_data), len(trip_data[trip_data["End"]])/len(trip_data))
    trip_data_sub = trip_data[trip_data["Complete"]]
    trip_data_sub["estim_route"] = "route_"+trip_data_sub["Routes"].str.split("-").str[0]+"_"+trip_data_sub["Routes"].str.split("-").str[-1]
    trip_data_sub["correct_complete_route"] = trip_data_sub["estim_route"]==trip_data_sub["true_route"]
    print(">> CORRECTNESS", np.mean(np.mean(trip_data_sub["correct_complete_route"])))
    lst_all_completed_events = get_list_all_events(trip_data_sub)
    print(">> COMPLETED LOOP EVENTS", len(lst_all_completed_events)/len(df_loop_events), "\n")

# QUALITY CONTROL METHODS
def doQualityControl(trip_data5):
    trip_data5["entrance_r"] = trip_data5["Routes"].str.split("-").str[0]
    trip_data5["exit_r"] = trip_data5["Routes"].str.split("-").str[-1]
    trip_data5["entrance_t"] = trip_data5["true_route"].str.split("_").str[1]
    trip_data5["exit_t"] = trip_data5["true_route"].str.split("_").str[-1]
    
    trip_data5["entrance_u"] = trip_data5["entrance_r"]==trip_data5["entrance_t"]
    
    df_entrance_r = trip_data5.groupby('entrance_r').size().reset_index(name='count')
    df_exit_r = trip_data5.groupby('exit_r').size().reset_index(name='count')
    df_entrance_t = trip_data5.groupby('entrance_t').size().reset_index(name='count')
    df_exit_t = trip_data5.groupby('exit_t').size().reset_index(name='count')
    
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set up the figure and axes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Calculate differences for entrances
    df_entrance_diff = pd.merge(df_entrance_t, df_entrance_r, left_on='entrance_t', right_on='entrance_r', how='left')
    df_entrance_diff['diff'] = df_entrance_diff['count_x'].fillna(0) - df_entrance_diff['count_y'].fillna(0)
    df_entrance_diff = df_entrance_diff.sort_values('diff', ascending=False)
    
    # Subplot 3: Entrance differences
    sns.barplot(x='entrance_t', y='diff', data=df_entrance_diff, ax=ax2, palette=['red' if x > 0 else 'blue' for x in df_entrance_diff['diff']])
    ax2.set_title('Difference in Entrance Counts (t - r)')
    ax2.set_xlabel('Entrance')
    ax2.set_ylabel('Count Difference')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90, ha='right')
    
    # Calculate differences for exits
    df_exit_diff = pd.merge(df_exit_t, df_exit_r, left_on='exit_t', right_on='exit_r', how='left')
    df_exit_diff['diff'] = df_exit_diff['count_x'].fillna(0) - df_exit_diff['count_y'].fillna(0)
    df_exit_diff = df_exit_diff.sort_values('diff', ascending=False)
    
    # Subplot 4: Exit differences
    sns.barplot(x='exit_t', y='diff', data=df_exit_diff, ax=ax4, palette=['red' if x > 0 else 'blue' for x in df_exit_diff['diff']])
    ax4.set_title('Difference in Exit Counts (t - r)')
    ax4.set_xlabel('Exit')
    ax4.set_ylabel('Count Difference')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=90, ha='right')
    
    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()


###############################################################################
# ######### PARAMETERS ########################################################
###############################################################################
loop_log_folder = "../model_logs/original/loops/"
ground_truth_file = "../model/demand/Spawn_Cars_original.csv"
simulation_time_start = datetime.strptime('2024-03-04 09:15:00', '%Y-%m-%d %H:%M:%S')
# simulation_time_limit = datetime.strptime('2024-03-04 10:15:00', '%Y-%m-%d %H:%M:%S')  
simulation_time_limit = datetime.strptime('2024-03-04 23:00:00', '%Y-%m-%d %H:%M:%S')  

# """
OD_method = "run_entropy_maxim_"
OD_agg = "15"
path_OD_matrix = "module_1_od_estimation/od_matrix_"+OD_agg+"m/"
od_matrices = loadODMatrices(path_OD_matrix)
# """

"""
OD_method = "run_entropy_maxim_"
OD_agg = "30"
path_OD_matrix = "module_1_od_estimation/od_matrix_"+OD_agg+"m/"
od_matrices = loadODMatrices(path_OD_matrix)
"""

"""
OD_method = "run_entropy_maxim_"
OD_agg = "60"
path_OD_matrix = "module_1_od_estimation/od_matrix_"+OD_agg+"m/"
od_matrices = loadODMatrices(path_OD_matrix)
"""



###############################################################################
# ######### NETWORK & MATCHING INFORMATION ####################################
###############################################################################

# FLOW INFORMATION
perimeter_entry_flows = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10", "E11", "E12", "E13", "E14", "E15", "E16", "E17", "E18", "E19", "E20", "E21", "E22", "E23", "E24"]
perimeter_exit_flows  = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18"]
excluded_detectors = ["215_D113", "215_D93", "215_D72", "215_D52", "215_D32", "215_D12", "216_D102", "216_D82", "217_D101", "217_D103A", "217_D103B" ]
entry_detectors = {
    "217_D81": "E1",
    "217_D61": "E2",
    "217_D41": "E3",
    "217_D21": "E4",
    "217_D11": "E5",
    "217_D151": "E6",
    "217_D131": "E7",
    
    "216_D61": "E9",
    "216_D21": "E10",
    "216_D11": "E11",
    
    "215_D31": "E12",
    "215_D11": "E13",
    
    "212_D1": "E19",
    
    "235_D61": "E22",
    "235_D81": "E21",
    "235_D11": "E23",
    "235_D101": "E24"
}
exit_detectors = {
    "217_D102": "A1",
    "217_D103": "A2",

    "216_D21": "A5",
    "216_D101": "A6",

    "215_D111": "A7",
    "215_D51": "A7",
    
    "235_D42": "A16_A18",
    "235_D025": "A17",
    "235_D104": "A17",
    "235_D125": "A18"    
}

internal_matching = {
    "z_1": {
        "sources_cer": {"217_D81": "217_K5", "217_D21": "217_K2",},
        "sources_unc": {"217_D131": "217_K8"},
        "sources_flow": {"217_D81": "E1", "217_D21": "E4", "217_D131": "E7"},
        "destination": ["216_D41"],
        "destination_flow": {"216_D41": "Z1"},
        "destination_flow_unc_alternative": {},
    },
    "z_4": {
        "sources_cer": {"216_D41": "216_K3", "216_D11": "216_K1"},
        "sources_unc": {},
        "sources_flow": {"216_D41": "Z1", "216_D11": "E11"},
        "destination": ["215_D111", "215_D91"],
        "destination_flow": {"215_D111": "Z7", "215_D91": "Z8"},
        "destination_flow_unc_alternative": {},
    },
    "z_10": {
        "sources_cer": {"215_D91": "215_K5", },
        "sources_unc": {"215_D31": "215_K2"},
        "sources_flow": {"215_D91": "Z8", "215_D31": "E12"},
        "destination": ["211_D015"],
        "destination_flow": {"211_D015": "Z10"},
        "destination_flow_unc_alternative": {"215_D31": "A8"},
    },
    "z_15": {
        "sources_cer": {"212_D1": "NONE"},
        "sources_unc": {"211_D015": "211_K1", },
        "sources_flow": {"211_D015": "Z10", "212_D1": "E19"},
        "destination": ["213_D42"],
        "destination_flow": {"213_D42": "Z15"},
        "destination_flow_unc_alternative": {},
    },
    "z_16": {
        "sources_cer": {"NO_DETECTOR": "NONE"},
        "sources_unc": {"213_D42": "213_K2",},
        "sources_flow": {"213_D42": "Z15", "NO_DETECTOR": "E20"},
        "destination": ["235_D42", "235_D025"],
        "destination_flow": {"235_D42": "Z19", "235_D025": "Z18"},
        "destination_flow_unc_alternative": {},
    },
    "z_19": {
        "sources_cer": {},
        "sources_unc": {"235_D42": "NONE", "235_D81": "NONE", "235_D11": "235_K1" },
        "sources_flow": {"235_D42": "Z19", "235_D81": "E21", "235_D11": "E23"},
        "destination": ["235_D125"],
        "destination_flow": {"235_D125": "A18"},
        "destination_flow_unc_alternative": {"235_D81": "A17", },
    },
    "z_20": {
        "sources_cer": {"235_D101": "235_K6"},
        "sources_unc": {},
        "sources_flow": {"235_D101": "E24"},
        "destination": ["235_D104"],
        "destination_flow": {"235_D104": "Z20"},
        "destination_flow_unc_alternative": {},
    },
    "z_17": {
        "sources_cer": {"235_D61": "235_K4"},
        "sources_unc": {"235_D101": "235_K6", "235_D11": "235_K1"},
        "sources_flow": {"235_D61": "E22", "235_D101": "E24", "235_D11": "E23"},
        "unmatched_filter": {"235_D101", "235_D11"},
        "destination": ["213_D52"],
        "destination_flow": {"213_D52": "Z17"},
        "destination_flow_unc_alternative": {},
    },
    "z_14": {
        "sources_cer": {},
        "sources_unc": {"213_D52": "213_K1", "NO_DETECTOR": "NONE"},
        "sources_flow": {"213_D52": "Z17", "NO_DETECTOR": "E20"},
        "destination": ["211_D035"],
        "destination_flow": {"211_D035": "Z14"},
        "destination_flow_unc_alternative": {},
    },
    "z_13": {
        "sources_cer": {"NO_DETECTOR_1": "NONE"},
        "sources_unc": {"211_D035": "NONE"},
        "sources_flow": {"211_D035": "Z14", "NO_DETECTOR_1": "E14"},
        "destination": ["215_D51", "215_D71"],
        "destination_flow": {"215_D51": "Z12", "215_D71": "Z11"},
        "destination_flow_unc_alternative": {"NO_DETECTOR_2": "A9"},
    },
    "z_9": {
        "sources_cer": {"215_D71": "215_K4", "215_D11": "215_K1"},
        "sources_unc": {},
        "sources_flow": {"215_D71": "Z11", "215_D11": "E13"},
        "destination": ["216_D81", "216_D101"],
        "destination_flow": {"216_D81": "Z6", "216_D101": "Z5"},
        "destination_flow_unc_alternative": {},
    },
    "z_6": {
        "sources_cer": {},
        "sources_unc": {"216_D81": "216_K5", "216_D61": "216_K4"},
        "sources_flow": {"216_D81": "Z6", "216_D61": "E9"},
        "destination": ["217_D103", "217_D102"],
        "destination_flow": {"217_D103": "Z3", "217_D102": "Z2"},
        "destination_flow_unc_alternative": {"216_D61": "A6"},
    },
    "z_3": {
        "sources_cer": {"217_D103": "NONE"},
        "sources_unc": {},
        "sources_flow": {"217_D103": "Z3"},
        "destination": ["217_D103_A", "217_D103_B"],
        "destination_flow": {"217_D103_A": "A2", "217_D103_B": "A2"},
        "destination_flow_unc_alternative": {},
    }
}

direct_matching_perimeter_detectors_map = {
    # 217
    "217_D11": ["E5-A3", "Done"],
    "217_D41": ["E3-A1", "Done"],
    "217_D61": ["E2-A3", "Done"],
    "217_D151": ["E6-A2", "Done"],
    # 216
    "216_D21": ["E10-A5", "Done"],
    # 215
    # 211
    # 212
    # 213
    # 235
}

direct_matching_internal_perimeter_detectors_map = {
    # ES217
    "Z2": "A1",
    # ES216
    "Z5": "A6",
    # ES215
    "Z7": "A7",
    "Z12": "A7",
    # ES211
    # ES212
    # ES213
    # ES235
    "Z18": "A17",
    "Z20": "A17",
}

route_path_dict = {
    "route_E1_A4": "E1-A4",
    "route_E1_A7": "E1-Z1-Z7-A7",
    "route_E1_A9": "E1-Z1-Z8-A9",
    "route_E1_A10": "E1-Z1-Z8-Z10-A10",
    "route_E1_A11": "E1-Z1-Z8-Z10-A11",
    "route_E1_A12": "E1-Z1-Z8-Z10-A12",
    "route_E1_A13": "E1-Z1-Z8-Z10-A13",
    "route_E1_A14": "E1-Z1-Z8-Z10-A14",
    "route_E1_A15": "E1-Z1-Z8-Z10-Z15-A15",
    "route_E1_A16": "E1-Z1-Z8-Z10-Z15-Z19-A16",
    "route_E1_A17": "E1-Z1-Z8-Z10-Z15-Z18-A17",
    "route_E1_A18": "E1-Z1-Z8-Z10-Z15-Z19-A18",
    "route_E2_A3": "E2-A3",
    "route_E3_A1": "E3-A1",
    "route_E4_A4": "E4-A4",
    "route_E4_A7": "E4-Z1-Z7-A7",
    "route_E4_A9": "E4-Z1-Z8-A9",
    "route_E4_A10": "E4-Z1-Z8-Z10-A10",
    "route_E4_A11": "E4-Z1-Z8-Z10-A11",
    "route_E4_A12": "E4-Z1-Z8-Z10-A12",
    "route_E4_A13": "E4-Z1-Z8-Z10-A13",
    "route_E4_A14": "E4-Z1-Z8-Z10-A14",
    "route_E4_A15": "E4-Z1-Z8-Z10-Z15-A15",
    "route_E4_A16": "E4-Z1-Z8-Z10-Z15-Z19-A16",
    "route_E4_A17": "E4-Z1-Z8-Z10-Z15-Z18-A17",
    "route_E4_A18": "E4-Z1-Z8-Z10-Z15-Z19-A18",
    "route_E5_A3": "E5-A3",
    "route_E6_A2": "E6-A2",
    "route_E7_A1": "E7-A1",
    "route_E7_A4": "E7-A4",
    "route_E7_A7": "E7-Z1-Z7-A7",
    "route_E7_A9": "E7-Z1-Z8-A9",
	"route_E7_A10": "E7-Z1-Z8-Z10-A10",
    "route_E7_A11": "E7-Z1-Z8-Z10-A11",
    "route_E7_A12": "E7-Z1-Z8-Z10-A12",
    "route_E7_A13": "E7-Z1-Z8-Z10-A13",
    "route_E7_A14": "E7-Z1-Z8-Z10-A14",
    "route_E7_A15": "E7-Z1-Z8-Z10-Z15-A15",
    "route_E7_A16": "E7-Z1-Z8-Z10-Z15-Z19-A16",
    "route_E7_A17": "E7-Z1-Z8-Z10-Z15-Z18-A17",
    "route_E7_A18": "E7-Z1-Z8-Z10-Z15-Z19-A18",
    "route_E8_A1": "E8-Z2-A1",
    "route_E8_A2": "E8-Z3-A2",
    "route_E8_A7": "E8-Z1-Z7-A7",
    "route_E8_A9": "E8-Z1-Z8-A9",
	"route_E8_A10": "E8-Z1-Z8-Z10-A10",
    "route_E8_A11": "E8-Z1-Z8-Z10-A11",
    "route_E8_A12": "E8-Z1-Z8-Z10-A12",
    "route_E8_A13": "E8-Z1-Z8-Z10-A13",
    "route_E8_A14": "E8-Z1-Z8-Z10-A14",
    "route_E8_A15": "E8-Z1-Z8-Z10-Z15-A15",
    "route_E8_A16": "E8-Z1-Z8-Z10-Z15-Z19-A16",
    "route_E8_A17": "E8-Z1-Z8-Z10-Z15-Z18-A17",
    "route_E8_A18": "E8-Z1-Z8-Z10-Z15-Z19-A18",
    "route_E9_A1": "E9-Z2-A1",
    "route_E9_A2": "E9-Z3-A2",
    "route_E9_A4": "E9-A4",
    "route_E9_A6": "E9-A6",
    "route_E10_A5": "E10-A5",
    "route_E11_A7": "E11-Z7-A7",
    "route_E11_A9": "E11-Z8-A9",
    "route_E11_A10": "E11-Z8-Z10-A10",
    "route_E11_A11": "E11-Z8-Z10-A11",
    "route_E11_A12": "E11-Z8-Z10-A12",
    "route_E11_A13": "E11-Z8-Z10-A13",
    "route_E11_A14": "E11-Z8-Z10-A14",
    "route_E11_A15": "E11-Z8-Z10-Z15-A15",
    "route_E11_A16": "E11-Z8-Z10-Z15-Z19-A16",
    "route_E11_A17": "E11-Z8-Z10-Z15-Z18-A17",
    "route_E11_A18": "E11-Z8-Z10-Z15-Z19-A18",
    "route_E12_A8": "E12-A8",
    "route_E12_A9": "E12-A9",
    "route_E12_A10": "E12-Z10-A10",
    "route_E12_A11": "E12-Z10-A11",
    "route_E12_A12": "E12-Z10-A12",
    "route_E12_A13": "E12-Z10-A13",
    "route_E12_A14": "E12-Z10-A14",
    "route_E12_A15": "E12-Z10-Z15-A15",
    "route_E12_A16": "E12-Z10-Z15-Z19-A16",
    "route_E12_A17": "E12-Z10-Z15-Z18-A17",
    "route_E12_A18": "E12-Z10-Z15-Z19-A18",
    "route_E13_A1": "E13-Z6-Z2-A1",
    "route_E13_A2": "E13-Z6-Z3-A2",
    "route_E13_A4": "E13-Z6-A4",
    "route_E13_A5": "E13-Z6-A5",
    "route_E13_A6": "E13-Z5-A6",
    "route_E14_A1": "E14-Z11-Z6-Z2-A1",
    "route_E14_A2": "E14-Z11-Z6-Z3-A2",
    "route_E14_A4": "E14-Z11-Z6-A4",
    "route_E14_A5": "E14-Z11-Z6-A5",
    "route_E14_A6": "E14-Z11-Z6-A6",
    "route_E14_A7": "E14-Z12-A7",
    "route_E14_A10": "E14-Z10-A10",
    "route_E14_A11": "E14-Z10-A11",
    "route_E14_A12": "E14-Z10-A12",
    "route_E14_A13": "E14-Z10-A13",
    "route_E14_A14": "E14-Z10-A14",
    "route_E14_A15": "E14-Z10-Z15-A15",
    "route_E14_A16": "E14-Z10-Z15-Z19-A16",
    "route_E14_A17": "E14-Z10-Z15-Z18-A17",
    "route_E14_A18": "E14-Z10-Z15-Z19-A18",
    "route_E15_A1": "E15-Z11-Z6-Z2-A1",
    "route_E15_A2": "E15-Z11-Z6-Z3-A2",
    "route_E15_A4": "E15-Z11-Z6-A4",
    "route_E15_A5": "E15-Z11-Z6-A5",
    "route_E15_A6": "E15-Z11-Z6-A6",
    "route_E15_A7": "E15-Z12-A7",
    "route_E15_A9": "E15-A9",
    "route_E15_A11": "E15-A11",
    "route_E15_A12": "E15-A12",
    "route_E15_A13": "E15-A13",
    "route_E15_A14": "E15-A14",
    "route_E15_A15": "E15-Z15-A15",
    "route_E15_A16": "E15-Z15-Z19-A16",
    "route_E15_A17": "E15-Z15-Z18-A17",
    "route_E15_A18": "E15-Z15-Z19-A18",
    "route_E16_A1": "E16-Z14-Z11-Z6-Z2-A1",
    "route_E16_A2": "E16-Z14-Z11-Z6-Z3-A2",
    "route_E16_A4": "E16-Z14-Z11-Z6-A4",
    "route_E16_A5": "E16-Z14-Z11-Z6-A5",
    "route_E16_A6": "E16-Z14-Z11-Z6-A6",
    "route_E16_A7": "E16-Z14-Z12-A7",
    "route_E16_A9": "E16-A9",
    "route_E16_A10": "E16-A10",
    "route_E16_A12": "E16-A12",
    "route_E16_A13": "E16-A13",
    "route_E16_A14": "E16-A14",
    "route_E16_A15": "E16-Z15-A15",
    "route_E16_A16": "E16-Z15-Z19-A16",
    "route_E16_A17": "E16-Z15-Z18-A17",
    "route_E16_A18": "E16-Z15-Z19-A18",
    "route_E17_A1": "E17-Z14-Z11-Z6-Z2-A1",
    "route_E17_A2": "E17-Z14-Z11-Z6-Z3-A2",
    "route_E17_A4": "E17-Z14-Z11-Z6-A4",
    "route_E17_A5": "E17-Z14-Z11-Z6-A5",
    "route_E17_A6": "E17-Z14-Z11-Z6-A6",
    "route_E17_A7": "E17-Z14-Z12-A7",
    "route_E17_A9": "E17-Z14-A9",
    "route_E17_A10": "E17-Z14-A10",
    "route_E17_A11": "E17-A11",
    "route_E17_A13": "E17-A13",
    "route_E17_A14": "E17-A14",
    "route_E17_A15": "E17-Z15-A15",
    "route_E17_A16": "E17-Z15-Z19-A16",
    "route_E17_A17": "E17-Z15-Z18-A17",
    "route_E17_A18": "E17-Z15-Z19-A18",
    "route_E18_A1": "E18-Z14-Z11-Z6-Z2-A1",
    "route_E18_A2": "E18-Z14-Z11-Z6-Z3-A2",
    "route_E18_A4": "E18-Z14-Z11-Z6-A4",
    "route_E18_A5": "E18-Z14-Z11-Z6-A5",
    "route_E18_A6": "E18-Z14-Z11-Z6-A6",
    "route_E18_A7": "E18-Z14-Z12-A7",
    "route_E18_A9": "E18-Z14-A9",
    "route_E18_A10": "E18-Z14-A10",
    "route_E18_A11": "E18-A11",
    "route_E18_A12": "E18-A12",
    "route_E18_A14": "E18-A14",
    "route_E18_A15": "E18-Z15-A15",
    "route_E18_A16": "E18-Z15-Z19-A16",
    "route_E18_A17": "E18-Z15-Z18-A17",
    "route_E18_A18": "E18-Z15-Z19-A18",   
    "route_E19_A1": "E19-Z14-Z11-Z6-Z2-A1",
    "route_E19_A2": "E19-Z14-Z11-Z6-Z3-A2",
    "route_E19_A4": "E19-Z14-Z11-Z6-A4",
    "route_E19_A5": "E19-Z14-Z11-Z6-A5",
    "route_E19_A6": "E19-Z14-Z11-Z6-A6",
    "route_E19_A7": "E19-Z14-Z12-A7",
    "route_E19_A9": "E19-Z14-A9",
    "route_E19_A10": "E19-Z14-A10",
    "route_E19_A11": "E19-A11",
    "route_E19_A12": "E19-A12",
    "route_E19_A13": "E19-A13",
    "route_E19_A15": "E19-Z15-A15",
    "route_E19_A16": "E19-Z15-Z19-A16",
    "route_E19_A17": "E19-Z15-Z18-A17",
    "route_E19_A18": "E19-Z15-Z19-A18",
    "route_E20_A1": "E20-Z14-Z11-Z6-Z2-A1",
    "route_E20_A2": "E20-Z14-Z11-Z6-Z3-A2",
    "route_E20_A4": "E20-Z14-Z11-Z6-A4",
    "route_E20_A5": "E20-Z14-Z11-Z6-A5",
    "route_E20_A6": "E20-Z14-Z11-Z6-A6",
    "route_E20_A7": "E20-Z14-Z12-A7",
    "route_E20_A9": "E20-Z14-A9",
    "route_E20_A10": "E20-Z14-A10",
    "route_E20_A11": "E20-A11",
    "route_E20_A12": "E20-A12",
    "route_E20_A13": "E20-A13",
    "route_E20_A14": "E20-A14",
    "route_E20_A16": "E20-Z19-A16",
    "route_E20_A17": "E20-Z18-A17",
    "route_E20_A18": "E20-Z19-A18",
    "route_E21_A17": "E21-A17",
    "route_E21_A18": "E21-A18",  
    "route_E22_A1": "E22-Z17-Z14-Z11-Z6-Z2-A1",
    "route_E22_A2": "E22-Z17-Z14-Z11-Z6-Z3-A2",
    "route_E22_A4": "E22-Z17-Z14-Z11-Z6-A4",
    "route_E22_A5": "E22-Z17-Z14-Z11-Z6-A5",
    "route_E22_A6": "E22-Z17-Z14-Z11-Z5-A6",
    "route_E22_A7": "E22-Z17-Z14-Z12-A7",
    "route_E22_A9": "E22-Z17-Z14-A9",
    "route_E22_A10": "E22-Z17-Z14-A10",
    "route_E22_A11": "E22-Z17-A11",
    "route_E22_A12": "E22-Z17-A12",
    "route_E22_A13": "E22-Z17-A13",
    "route_E22_A14": "E22-Z17-A14",
    "route_E22_A15": "E22-Z17-A15",
    "route_E23_A1": "E23-Z17-Z14-Z11-Z6-Z2-A1",
    "route_E23_A2": "E23-Z17-Z14-Z11-Z6-Z3-A2",
    "route_E23_A4": "E23-Z17-Z14-Z11-Z6-A4",
    "route_E23_A5": "E23-Z17-Z14-Z11-Z6-A5",
    "route_E23_A6": "E23-Z17-Z14-Z11-Z5-A6",
    "route_E23_A7": "E23-Z17-Z14-Z12-A7",
    "route_E23_A9": "E23-Z17-Z14-A9",
    "route_E23_A10": "E23-Z17-Z14-A10",
    "route_E23_A11": "E23-Z17-A11",
    "route_E23_A12": "E23-Z17-A12",
    "route_E23_A13": "E23-Z17-A13",
    "route_E23_A14": "E23-Z17-A14",
    "route_E23_A15": "E23-Z17-A15",
    "route_E23_A16": "E23-Z17-Z14-A16",
    "route_E23_A18": "E23-A18",
    "route_E24_A1": "E24-Z17-Z14-Z11-Z6-Z2-A1",
    "route_E24_A2": "E24-Z17-Z14-Z11-Z6-Z3-A2",
    "route_E24_A4": "E24-Z17-Z14-Z11-Z6-A4",
    "route_E24_A5": "E24-Z17-Z14-Z11-Z6-A5",
    "route_E24_A6": "E24-Z17-Z14-Z11-Z5-A6",
    "route_E24_A7": "E24-Z17-Z14-Z12-A7",
    "route_E24_A9": "E24-Z17-Z14-A9",
    "route_E24_A10": "E24-Z17-Z14-A10",
    "route_E24_A11": "E24-Z17-A11",
    "route_E24_A12": "E24-Z17-A12",
    "route_E24_A13": "E24-Z17-A13",
    "route_E24_A14": "E24-Z17-A14",
    "route_E24_A15": "E24-Z17-A15",
    "route_E24_A16": "E24-Z17-Z14-A16",
    "route_E24_A17": "E24-Z20-A17",
}

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



###############################################################################
# ######### RECONSTRUCT TRIPS #################################################
###############################################################################
# LOAD DATA
    # Load Car-Route Ground Truth
df_ground_truth = load_ground_truth_routes(ground_truth_file)
    # Load Loop Detectors
df_loop_events = load_loop_detector_events(loop_log_folder, df_ground_truth)

# Step 1: CONSIDER ONLY RELEVANT DETECTORS
df_loop_events = df_loop_events[~ df_loop_events["detector"].isin(excluded_detectors)]

# Step 2: DIRECT PERIMETER MATCHING
df_loop_events = df_loop_events.apply(direct_matching_perimeter_detectors, axis=1)

# Step 3: INTERNAL MATCHING
print("INTERNAL MATCHING")
df_loop_events, lst_matched01 = conduct_matching_group("Z_1",    internal_matching["z_1"],    df_loop_events, do_scoring=False)
df_loop_events, lst_matched02 = conduct_matching_group("Z_4",    internal_matching["z_4"],    df_loop_events, do_scoring=False)
df_loop_events, lst_matched03 = conduct_matching_group("Z_10",   internal_matching["z_10"],   df_loop_events, do_scoring=False)
df_loop_events, lst_matched04 = conduct_matching_group("Z_15",   internal_matching["z_15"],   df_loop_events, do_scoring=False)
df_loop_events, lst_matched05 = conduct_matching_group("Z_16",   internal_matching["z_16"],   df_loop_events, do_scoring=False)
df_loop_events, lst_matched06 = conduct_matching_group("Z_19",   internal_matching["z_19"],   df_loop_events, do_scoring=False)
df_loop_events, lst_matched07 = conduct_matching_group("Z_20",   internal_matching["z_20"],   df_loop_events, do_scoring=False)
df_loop_events, lst_matched08 = conduct_matching_group("Z_17",   internal_matching["z_17"],   df_loop_events, do_scoring=False)
df_loop_events, lst_matched10 = conduct_matching_group("Z_14",   internal_matching["z_14"],   df_loop_events, do_scoring=False)
df_loop_events, lst_matched11 = conduct_matching_group("Z_13",   internal_matching["z_13"],   df_loop_events, do_scoring=False)
df_loop_events, lst_matched12 = conduct_matching_group("Z_9",    internal_matching["z_9"],    df_loop_events, do_scoring=False)
df_loop_events, lst_matched13 = conduct_matching_group("Z_6",    internal_matching["z_6"],    df_loop_events, do_scoring=False,  do_max=False)
df_loop_events, lst_matched14 = conduct_matching_group("Z_3",    internal_matching["z_3"],    df_loop_events, do_scoring=False,  do_max=False)
print(">> UNMATCHED LOOP EVENTS", len(df_loop_events[df_loop_events["reconstructed_status"]=="?"])/len(df_loop_events), "\n")

# Step 4: TRIP RECONSTRUCTION
    # reconstruction
trip_data = reconstruct_trips(df_loop_events)
print("STEP 4.1")
print_trip_statistics(df_loop_events, trip_data)
    # matching internal perimeter detectors
trip_data2 = trip_data.apply(direct_matching_internal_perimeter_detectors, axis=1)
print("STEP 4.2")
print_trip_statistics(df_loop_events, trip_data2)
    # matching remaining trips
trip_data3, df_loop_events3, lst_matches = match_remaining_trips(trip_data2, df_loop_events)
trip_data3 = trip_data3.reset_index()
del trip_data3["index"]
print("STEP 4.3")
print_trip_statistics(df_loop_events3, trip_data3)
    # matching disconnected remaining trips
trip_data4, df_loop_events4 = match_disconnected_remaining_trips(trip_data3, df_loop_events3)
print("STEP 4.4")
print_trip_statistics(df_loop_events4, trip_data4)
    # guestimating remaining starters
trip_data5, df_loop_events5 = guess_remaining_starter_loop_events(df_loop_events4, trip_data4, od_matrices)
print("STEP 4.5")
print_trip_statistics(df_loop_events5, trip_data5)


import sys
sys.exit(0)


############# QUALITY CONTROL
# doQualityControl(trip_data5)

############# FINAL CONVERSION TO SPAWN CARS FILE
vehicle_spawn_df = trip_data5.copy()
vehicle_spawn_df["Datetime"] = vehicle_spawn_df["Starttime"]
vehicle_spawn_df["n_spawn"] = 1.0
vehicle_spawn_df["entrance"] = vehicle_spawn_df["Routes"].str.split("-").str[0]
vehicle_spawn_df["route"] = "route_"+vehicle_spawn_df["Routes"].str.split("-").str[0]+"_"+vehicle_spawn_df["Routes"].str.split("-").str[-1]
vehicle_spawn_df['spawn_delay'] = vehicle_spawn_df['entrance'].map(spawn_delay)
vehicle_spawn_df["Datetime"] = pd.to_datetime(vehicle_spawn_df["Datetime"], format="mixed")
vehicle_spawn_df['Adjusted_Datetime'] = vehicle_spawn_df['Datetime'] - pd.to_timedelta(vehicle_spawn_df['spawn_delay'], unit='seconds')
vehicle_spawn_df = vehicle_spawn_df[["Datetime", "n_spawn", "entrance", "route", "spawn_delay", "Adjusted_Datetime"]]
vehicle_spawn_df['Datetime'] = vehicle_spawn_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
vehicle_spawn_df['Adjusted_Datetime'] = vehicle_spawn_df['Adjusted_Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
vehicle_spawn_df["route"] = vehicle_spawn_df["route"].str.replace("route_E12_A7", "route_E12_A8", regex=False)
vehicle_spawn_df.to_csv("Spawn_Cars_trip_reconstruction_t6.csv")

perfect_fake = trip_data5.copy()
perfect_fake["Routes"] = perfect_fake["true_route"].map(route_path_dict)
vehicle_spawn_df = perfect_fake.copy()
vehicle_spawn_df["Datetime"] = vehicle_spawn_df["Starttime"]
vehicle_spawn_df["n_spawn"] = 1.0
vehicle_spawn_df["entrance"] = vehicle_spawn_df["Routes"].str.split("-").str[0]
vehicle_spawn_df["route"] = "route_"+vehicle_spawn_df["Routes"].str.split("-").str[0]+"_"+vehicle_spawn_df["Routes"].str.split("-").str[-1]
vehicle_spawn_df['spawn_delay'] = vehicle_spawn_df['entrance'].map(spawn_delay)
vehicle_spawn_df["Datetime"] = pd.to_datetime(vehicle_spawn_df["Datetime"], format="mixed")
vehicle_spawn_df['Adjusted_Datetime'] = vehicle_spawn_df['Datetime'] - pd.to_timedelta(vehicle_spawn_df['spawn_delay'], unit='seconds')
vehicle_spawn_df = vehicle_spawn_df[["Datetime", "n_spawn", "entrance", "route", "spawn_delay", "Adjusted_Datetime"]]
vehicle_spawn_df['Datetime'] = vehicle_spawn_df['Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
vehicle_spawn_df['Adjusted_Datetime'] = vehicle_spawn_df['Adjusted_Datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
vehicle_spawn_df["route"] = vehicle_spawn_df["route"].str.replace("route_E12_A7", "route_E12_A8", regex=False)
vehicle_spawn_df.to_csv("Spawn_Cars_trip_reconstruction_perfect.csv")

############# LOG
"""
INTERNAL MATCHING
>> Z_1
	Matching done ; Matched: 7463 ; Unmatched Certains: 10 ; Unmatched Uncertains: 643
	Evaluation: 0.44948481198983004 0.9781589173254723 0.4690721649484536 0.16973415132924335 ; 0.4296875 0.16362416828912352
>> Z_4
	Matching done ; Matched: 1102 ; Unmatched Certains: 7449 ; Unmatched Uncertains: 0
	Evaluation: 0.04958484387790902 0.6941923774954628 0.38475499092558985 0.0 ; 0.04819277108433735 0.027090479115406532
>> Z_10
	Matching done ; Matched: 5088 ; Unmatched Certains: 319 ; Unmatched Uncertains: 243
	Evaluation: 0.5718513038653598 0.9127358490566038 0.7249686323713928 0.14275668073136427 ; 0.5488721804511278 0.14046185091399574
>> Z_15
	Matching done ; Matched: 347 ; Unmatched Certains: 5104 ; Unmatched Uncertains: 5089
	Evaluation: 0.010273344340487984 0.9740634005763689 0.5517241379310345 0.0073773515308004425 ; 0.007518796992481203 0.013145722207057635
>> Z_16
	Matching done ; Matched: 5017 ; Unmatched Certains: 307 ; Unmatched Uncertains: 434
	Evaluation: 0.27460555972952666 0.9722941997209488 0.25 0.27462406015037594 ; 0.2682926829268293 0.14681912796419516
>> Z_19
	Matching done ; Matched: 321 ; Unmatched Certains: 2805 ; Unmatched Uncertains: 5125
	Evaluation: 0.05118362124120281 0.7881619937694704 1.0 0.05027217419148255 ; 0.04285714285714286 0.03710595500661659
>> Z_20
	Matching done ; Matched: 0 ; Unmatched Certains: 0 ; Unmatched Uncertains: 0
>> Z_17
	Matching done ; Matched: 4786 ; Unmatched Certains: 13 ; Unmatched Uncertains: 94
	Evaluation: 0.7003542404667639 0.9922691182615964 0.8376184032476319 0.5824941905499613 ; 0.6989247311827957 0.12113539791335368
>> Z_14
	Matching done ; Matched: 13 ; Unmatched Certains: 5403 ; Unmatched Uncertains: 4786
	Evaluation: 0.00018463810930576072 0.9230769230769231 -1 0.00018463810930576072 ; 0.0 0.0011868363072859006
>> Z_13
	Matching done ; Matched: 5266 ; Unmatched Certains: 144 ; Unmatched Uncertains: 150
	Evaluation: 0.36820702402957484 0.9441701481200152 1.0 0.36809022000369757 ; 0.3305785123966942 0.20530343598256653
>> Z_9
	Matching done ; Matched: 1425 ; Unmatched Certains: 5051 ; Unmatched Uncertains: 0
	Evaluation: 0.1063928350833848 0.7684210526315789 0.48350877192982455 0.0 ; 0.09552350962570672 0.0616440184285094
>> Z_6
	Matching done ; Matched: 4548 ; Unmatched Certains: 804 ; Unmatched Uncertains: 1146
	Evaluation: 0.31352765321375187 0.73636763412489 0.75 0.3132011967090501 ; 0.3 0.1403620900446505
>> Z_3
	Matching done ; Matched: 674 ; Unmatched Certains: 3839 ; Unmatched Uncertains: 0
	Evaluation: 0.1072457345446488 0.8887240356083086 0.7181008902077152 0.0 ; 0.09615384615384616 0.06963775540527004
>> UNMATCHED LOOP EVENTS 0.13393951888542877 

STEP 4.1
>> COMPLETED TRIPS 7459 0.17179510801971531 0.5491040582246994 0.1926159657284997
>> CORRECTNESS 0.9979890065692452
>> COMPLETED LOOP EVENTS 0.08192185414654789 

STEP 4.2
>> COMPLETED TRIPS 8302 0.19121101847160163 0.5491040582246994 0.24091390667465107
>> CORRECTNESS 0.9563960491447844
>> COMPLETED LOOP EVENTS 0.10021592647649223 

STEP 4.3
>> COMPLETED TRIPS 11062 0.25523765574526996 0.5801338255652977 0.27318874019381634
>> CORRECTNESS 0.7702043030193455
>> COMPLETED LOOP EVENTS 0.1618037998719632 

STEP 4.4
>> COMPLETED TRIPS 11794 0.27680247840781075 0.5901004506196019 0.27788208787082236
>> CORRECTNESS 0.7401220959810073
>> COMPLETED LOOP EVENTS 0.19357428847343236 

STEP 4.5
>> COMPLETED TRIPS 26165 1.0 1.0 1.0
>> CORRECTNESS 0.5375119434358876
>> COMPLETED LOOP EVENTS 0.3495079263462853 

TRUE NUMBER OF CARS
26,601
"""

# 7459/26714
