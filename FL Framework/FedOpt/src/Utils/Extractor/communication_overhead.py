import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

rif = 6590  # valori in kB
dict_data = {
    "gRPC IID": [6900,6372,6893,6357,14574,13434],
    "MQTT IID": [6896,6366,6895,6354,14475,13388],
    "gRPC no IID": [6901,6370,6893,6363,14425,13380],
    "MQTT no IID": [6900,6368,6896,6367,14554,13430],
}

# Sottrazione del valore di riferimento
for key in dict_data:
    dict_data[key] = [round((val/rif - 1)*100,2) for val in dict_data[key]]

# Creazione del DataFrame
df = pd.DataFrame(
    [[k] + v for k, v in dict_data.items()],
    columns=["Protocols", "FedAvg", "FedAvgPruneCS", "FedDyn", "FedDynPruneCS", "Scaffold", "ScaffoldPruneCS"]
)

# Creazione del grafico a barre con figsize
fig, ax = plt.subplots(figsize=(14, 9))

# Creazione del grafico a barre
bars = df.set_index('Protocols').plot(kind='bar', stacked=False, width=0.6, ax=ax)  # Aumenta la larghezza delle barre
for bar in bars.containers:
    ax.bar_label(bar, fmt='%.2f%%', label_type='edge', fontsize=16,rotation=90,padding=5)  # Aggiungi i valori sopra le barre

# Limitazione dell'asse y
ax.set_ylim(-50, 1000)

# Impostazioni aggiuntive
plt.xticks(range(len(df)), df['Protocols'])
plt.yscale('symlog')
ax.set_axisbelow(True)
plt.grid(axis='y', color='gray', linestyle='-', linewidth=0.7)

minor_ticks =  [-50, -40, -30, -20, -10, -9, -8, -7, -6, -5, -4, -3, -2, 2, 3,4, 5,6, 7,8,9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200,300,400,500,600,700,800,900 ]
plt.gca().yaxis.set_minor_locator(FixedLocator(minor_ticks))
plt.grid(which='minor', color='#cccccc', linestyle='-', linewidth=0.1)

plt.xticks(rotation=0,fontsize=20)
plt.yticks(fontsize=20)

plt.xlabel('Protocol',fontsize=20)
plt.ylabel('Communication OverHead (Log scale)',fontsize=20)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1),fontsize=20)
plt.tight_layout()
plt.savefig(f"./img/overhead.svg", bbox_inches='tight')
plt.show()
