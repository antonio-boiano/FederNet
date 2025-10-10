import re
import csv
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes,mark_inset

# folder = 'avg_cs_iid' # CHANGE HERE FOLDER NAME
folder = 'avg_cs_iid' 
zoom = "end" # ['start','end,'']
log_folder = f'./logs/{folder}'
csv_folder = f'./results/{folder}'
os.makedirs(csv_folder, exist_ok=True) # create output folder
log_files = [f for f in os.listdir(log_folder) if f.endswith('.log')]
map_algo = {
    'avg_all': "FedAvgPruneCS",
    'avg': "FedAvg",
    'avg_cs': "FedAvgCS",
    'avg_prune': "FedAvgPrune",
    'dyn_all': "FedDynPruneCS",
    'dyn': "FedDyn",
    'dyn_cs': "FedDynCS",
    'dyn_prune': "FedDynPrune",
    'scaff_all': "ScaffoldPruneCS",
    'scaffold': "Scaffold",
    'scaff': "Scaffold",
    'scaff_cs': "ScaffoldCS",
    'scaff_prune': "ScaffoldPrune",
    'lr0_005': "FedAvg (LR 0.005)",
    'lr0_05': "FedAvg (LR 0.05)",
    'lr0_01': "FedAvg (LR 0.01)",
    'lr0_1': "FedAvg (LR 0.1)",
    'feddyn_a_0_2': "FedDyn (alpha 0.2)",
    'feddyn_a_0_1': "FedDyn (alpha 0.1)",
    'feddyn_a_0_01': "FedDyn (alpha 0.01)",
    'feddyn_a_0_001': "FedDyn (alpha 0.001)",
    'scaffold_gr1': "SCAFFOLD (GR 1.00)",
    'scaffold_gr1_25': "SCAFFOLD (GR 1.25)",
    'scaffold_gr0_75': "SCAFFOLD (GR 0.75)",
}

def process_log_file(log_path, output_csv):
    with open(log_path, 'r') as file:
        log_data = file.read()

    pattern = re.compile(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) \| INFO \| abstract\.py:114 \| ACCURACY: (\d+\.\d+)")
    extracted_data = []
    for match in pattern.finditer(log_data):
        timestamp_str = match.group(1)
        accuracy = float(match.group(2))
        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
        extracted_data.append((timestamp, accuracy))
        with open(output_csv, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['time', 'accuracy'])
            writer.writerow([0, extracted_data[0][1]])
            for i in range(1, len(extracted_data), 1):
                avg_accuracy = extracted_data[i][1]
                writer.writerow([i, avg_accuracy])

if not log_files:
    print(f"No log files found in the folder {log_folder}.")
else:
    for i, log_file in enumerate(log_files):
        process_log_file(f'{log_folder}/{log_file}', f'{csv_folder}/{log_file[:-4]}.csv')
fig = plt.figure(figsize=(15, 12))
ax = plt.subplot(111)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
csv_files = sorted([f for f in os.listdir(csv_folder) if f.endswith('.csv')],reverse=False)
if not csv_files:
    print(f"No CSV file found in the folder {csv_folder}.")
else:
    for i, csv_file in enumerate(csv_files):
        file_path = os.path.join(csv_folder, csv_file)
        data = pd.read_csv(file_path)
        name = os.path.splitext(csv_file)[0][:-11]
        print(name)
        ax.plot(data['time'], data['accuracy'], color=colors[i], label=f'{map_algo[name]}', linewidth=2)

    ax.set_xlabel('Round', fontsize=36)
    ax.set_ylabel('Accuracy', fontsize=36)

    ax.set_xticks([0, 25, 50, 75, 100])  # Valori dei ticks sull'asse x
    ax.tick_params(axis='y', labelsize=36)  # Imposta il fontsize per l'asse y
    ax.tick_params(axis='x', labelsize=36)  # Imposta il fontsize per l'asse x

    # Abilitare la griglia
    plt.grid(True, which='both', axis='both', linestyle='-', linewidth=2)

    # Legenda
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles=handles, loc='lower right', fontsize=28)

    if zoom != "":
        if zoom == "end":
            axins = zoomed_inset_axes(ax, 2.3, bbox_to_anchor=(.45, .34, .6, .5),
                       bbox_transform=ax.transAxes, loc=3)
        else:
            axins = zoomed_inset_axes(ax, 1.3, bbox_to_anchor=(.55, .34, .6, .5),
                                      bbox_transform=ax.transAxes, loc=3)
        for i, csv_file in enumerate(csv_files):
            file_path = os.path.join(csv_folder, csv_file)
            data = pd.read_csv(file_path)
            name = os.path.splitext(csv_file)[0][:-11]
            axins.plot(data['time'], data['accuracy'], color=colors[i], linewidth=2)

        axins.tick_params(axis='both', labelsize=22)  # Imposta il fontsize per l'asse y
        axins.grid(True, which='both', axis='both', linestyle='--', linewidth=1)

        # Impostare i limiti per la parte zoomata
        if zoom == "end":
            axins.set_xlim(79, 99)
            axins.set_ylim(0.75, 0.89)
            axins.set_xticks([80,85,90,95])  # Valori dei ticks sull'asse x
            axins.set_yticks([0.775,0.8,0.825,0.85,0.875])  # Valori dei ticks sull'asse y
            mark_inset(ax, axins, loc1=4, loc2=2)
        else: # zoom = "start"
            axins.set_xlim(15, 45)
            axins.set_ylim(0.54, 0.83)
            axins.set_xticks([20,25,30,35,40])  # Valori dei ticks sull'asse x
            axins.set_yticks([0.60,0.65,0.7,0.75,0.8])  # Valori dei ticks sull'asse y
            mark_inset(ax,axins,loc1=3,loc2=1)

    #plt.savefig(f"./img/{folder}.svg", bbox_inches='tight')
    plt.show()
