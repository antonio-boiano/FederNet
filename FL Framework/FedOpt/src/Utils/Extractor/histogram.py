import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

algorithms = ['avg_all', 'avg', 'dyn_all', 'dyn', 'scaff_all','scaff']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
folders = {
    'gRPC': './times/grpc_italy',
    'MQTT': './times/mqtt_italy',
}

def extract_avg_times_from_files(log_files):
    file_avg_times = {}
    for log_file in log_files:
        total_time = 0
        count = 0
        with open(log_file, 'r') as file:
            for line in file:
                match = re.search(r'Round (\d+), time for [\d.]+ is ([\d.]+)', line)
                if match:
                    total_time += float(match.group(2))
                    count += 1
        if count > 0:
            file_name = os.path.basename(log_file)[:-15]
            file_avg_times[file_name] = total_time / count
    return file_avg_times

data = {}
for group, folder in folders.items():
    log_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.log')]
    data[group] = extract_avg_times_from_files(log_files)

sort_algo = sorted(algorithms)
fig, ax = plt.subplots(figsize=(15, 10))
width = 0
for group in data.keys():
    for i, alg in enumerate(sort_algo):
        ax.bar(width + i * 0.15, data[group][alg], 0.15, color=colors[i])
    width += 2

handles, labels = plt.gca().get_legend_handles_labels()
legend = []
for i,algo in enumerate(sort_algo):
    legend.append(mpatches.Patch(color=colors[i], label=algo))
handles.extend(legend)
ax.legend(handles=handles,loc='upper left', fontsize=22)

ax.set_xlabel('Protocols',fontsize=24)
ax.set_ylabel('Avg RTT Time [s]',fontsize=24)
ax.set_xticks([0.4,2.4])
ax.set_xticklabels(data.keys())
ax.tick_params(axis='both', labelsize=24)
plt.grid(True, which='both', axis='y', linestyle='-')
plt.savefig(f"./img/histogram.svg", bbox_inches='tight')
plt.show()

