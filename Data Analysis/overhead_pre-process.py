# %%
import os, glob, re
from typing import Dict
import pyshark
import asyncio
import nest_asyncio
import pandas as pd
import pickle
from pathlib import Path

# Apply the patch
nest_asyncio.apply()

_ROUNDS_RE = re.compile(
    r"""Round\s+(\d+),\s+time\s+for\s+\d+\s+is\s+[0-9]*\.?[0-9]+""",
    re.VERBOSE,
)

# remove a trailing «-<letters+digits>»  (e.g. -eth0, -wlan1, -lo)
_IFACE_TAIL = re.compile(r"-[a-zA-Z]+[0-9]*$")


import os
import subprocess

def get_flow_stats(pcap_path: str) -> pd.DataFrame:
    """
    Ritorna un DataFrame aggregato per flusso (ip_src, ip_dst, bytes).
    Gestisce intestazioni con il punto (es. ip.src) rinominandole con _.
    """
    if not os.path.isfile(pcap_path):
        print(f"[warn] PCAP file not found: {pcap_path}")
        return pd.DataFrame(columns=["ip_src", "ip_dst", "bytes"])

    try:
        # 1. carica il CSV
        df = pd.read_csv(pcap_path, na_filter=False)        # mantieni stringhe vuote

        # 2. uniforma i nomi colonna: ip.src ➜ ip_src, ip.dst ➜ ip_dst
        df.columns = [c.replace(".", "_") for c in df.columns]

        required = {"ip_src", "ip_dst", "frame_len"}
        if not required.issubset(df.columns):
            raise ValueError(f"Mancano colonne: {required - set(df.columns)}")

        # 3. filtra solo traffico verso 10.0.0.100 da sorgenti diverse
        mask = (df["ip_dst"] == "10.0.0.100") & (df["ip_src"] != "10.0.0.100")
        df_filtered = df.loc[mask, ["ip_src", "ip_dst", "frame_len"]]

        # 4. aggrega per flusso e rinomina
        flows = (
            df_filtered.groupby(["ip_src", "ip_dst"], as_index=False)["frame_len"]
            .sum()
            .rename(columns={"frame_len": "bytes"})
        )
        return flows

    except Exception as e:
        print(f"[error] Failed to process PCAP {pcap_path}: {e}")
        return pd.DataFrame(columns=["ip_src", "ip_dst", "bytes"])

def _base_stem(path: str) -> str:
    """Stem without interface suffix."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return _IFACE_TAIL.sub("", stem)

def _round_stats(log_path: str) -> tuple[int, int]:
    txt = open(log_path, encoding="utf-8").read()
    ids = [int(m.group(1)) for m in _ROUNDS_RE.finditer(txt)]
    return len(set(ids)), len(ids)

def analyze_federated_learning_data(folder: str) -> Dict[str, Dict]:
    pcaps = glob.glob(os.path.join(folder, "*ps_fed_opt0*.csv"))
    logs  = glob.glob(os.path.join(folder, "*ps_fed_opt0*.log"))

    # index logs by *base* stem
    logs_by_base = { _base_stem(p): p for p in logs }

    results = {}
    for pcap in pcaps:
        base = _base_stem(pcap)
        log  = logs_by_base.get(base)

        if log is None:
            print(f"[warn] n_msgsNo matching log for {pcap}")
            continue

        n_rounds, n_msgs = _round_stats(log)
        dict_flow_B = get_flow_stats(pcap)

        results[os.path.basename(pcap)] = {
            "rounds_completed": n_rounds,
            "client_messages":  n_msgs,
            "dict_flow_B": dict_flow_B,
            "log":              log,
        }

    return results


# %%
base_path = "/home/antoniob/FederNet/src/output"

df_list_sync = [
    {
    "file_path":"config_preliminary_test/2025-05-27-20-48-47_mqtt_fed_opt_C10_wifi_80211ac_rpi4_FedAvgN_A10_CSFalse",
    "model":"FedAvg",
    "protocol":"MQTT",
    "data_distribution":"iid",
    "client_selection":"False",
    },{
    "file_path":"config_preliminary_test/2025-05-28-14-55-27_mqtt_fed_opt_C10_wifi_80211ac_rpi4_FedAvgN_A0.5_CSTrue",
    "model":"FedAvg",
    "protocol":"MQTT",
    "data_distribution":"non-iid",
    "client_selection":"True",
    },{
    "file_path":"config_preliminary_test/2025-05-30-17-43-09_mqtt_fed_opt_C10_wifi_80211ac_rpi4_FedAvgN_A0.5_CSFalse",
    "model":"FedAvg",
    "protocol":"MQTT",
    "data_distribution":"non-iid",
    "client_selection":"False",
    },{
    "file_path":"config_preliminary_test/2025-05-29-00-36-48_mqtt_fed_opt_C10_wifi_80211ac_rpi4_SCAFFOLD_A10_CSFalse",
    "model":"SCAFFOLD",
    "protocol":"MQTT",
    "data_distribution":"iid",
    "client_selection":"False",
    },{
    "file_path":"config_preliminary_test/2025-05-29-12-16-54_mqtt_fed_opt_C10_wifi_80211ac_rpi4_SCAFFOLD_A0.5_CSTrue",
    "model":"SCAFFOLD",
    "protocol":"MQTT",
    "data_distribution":"non-iid",
    "client_selection":"True",
    },{
    "file_path":"config_preliminary_test/2025-05-31-03-15-34_mqtt_fed_opt_C10_wifi_80211ac_rpi4_SCAFFOLD_A0.5_CSFalse",
    "model":"SCAFFOLD",
    "protocol":"MQTT",
    "data_distribution":"non-iid",
    "client_selection":"False",
    },{
    "file_path":"config_grpc/2025-06-05-16-04-12_grpc_fed_opt_C10_wifi_80211ac_rpi4_FedAvgN_A10_CSFalse",
    "model":"FedAvg",
    "protocol":"gRPC",
    "data_distribution":"iid",
    "client_selection":"False",
    },{
    "file_path":"config_grpc/2025-06-10-01-33-31_grpc_fed_opt_C10_wifi_80211ac_rpi4_FedAvgN_A0.5_CSTrue",
    "model":"FedAvg",
    "protocol":"gRPC",
    "data_distribution":"non-iid",
    "client_selection":"True",
    },{
    "file_path":"config_grpc/2025-06-19-15-25-05_grpc_fed_opt_C10_wifi_80211ac_rpi4_FedAvgN_A0.5_CSFalse",
    "model":"FedAvg",
    "protocol":"gRPC",
    "data_distribution":"non-iid",
    "client_selection":"False",
    },{
    "file_path":"config_grpc/2025-06-06-10-44-39_grpc_fed_opt_C10_wifi_80211ac_rpi4_SCAFFOLD_A10_CSFalse",
    "model":"SCAFFOLD",
    "protocol":"gRPC",
    "data_distribution":"iid",
    "client_selection":"False",
    },{
    "file_path":"config_grpc/2025-06-10-15-05-02_grpc_fed_opt_C10_wifi_80211ac_rpi4_SCAFFOLD_A0.5_CSTrue",
    "model":"SCAFFOLD",
    "protocol":"gRPC",
    "data_distribution":"non-iid",
    "client_selection":"True",
    },{
    "file_path":"config_grpc/2025-06-09-16-02-38_grpc_fed_opt_C10_wifi_80211ac_rpi4_SCAFFOLD_A0.5_CSFalse",
    "model":"SCAFFOLD",
    "protocol":"gRPC",
    "data_distribution":"non-iid",
    "client_selection":"False",
    }
]


results_by_folder: Dict[str, Dict] = {}

for entry in df_list_sync:
    full_path = os.path.join(base_path, entry["file_path"])
    entry["full_path"] = full_path

    flows_dict = analyze_federated_learning_data(full_path)  # <- usa get_flow_stats

    # flows_dict[pcap]["dict_flow_B"] ora è un DataFrame, non un int
    entry["flows"]       = flows_dict
    entry["n_flows"]     = len(flows_dict)
    entry["bytes_total"] = sum(
        v["dict_flow_B"]["bytes"].sum() for v in flows_dict.values()
    )

    results_by_folder[entry["file_path"]] = entry
    
# ---------------------------------------------------------------------------
pkl_path = Path(base_path) / "sync_analysis.pkl"
with open(pkl_path, "wb") as fout:
    pickle.dump(df_list_sync, fout)

print(f"[info] Analisi completata. Risultati salvati in: {pkl_path}")