#!/bin/bash

BASE_PATH="/home/antoniob/FederNet/src/output"
NUM_CORES=$(nproc)  # Numero massimo di job paralleli
OVERWRITE=false     # true per forzare rigenerazione CSV

declare -a FILE_PATHS=(
    "config_preliminary_test"
    "config_grpc"
    "config_cs_iid"
    "config_mqtt_async"
    
)

# Trova tutti i file .pcap nelle directory specificate
PCAP_LIST=()
for REL_PATH in "${FILE_PATHS[@]}"; do
    DIR_PATH="${BASE_PATH}/${REL_PATH}"
    if [[ -d "$DIR_PATH" ]]; then
        while IFS= read -r -d '' file; do
            PCAP_LIST+=("$file")
        done < <(find "$DIR_PATH" -name "*.pcap" -print0)
    fi
done

# Funzione per processare un singolo file .pcap
process_pcap() {
    PCAP_FILE="$1"
    CSV_FILE="${PCAP_FILE%.pcap}.csv"

    if [[ -f "$CSV_FILE" && "$OVERWRITE" == false ]]; then
        echo "✅ Già processato (skipped): $CSV_FILE"
        return
    fi

    echo "📦 Processando: $PCAP_FILE"

    tshark -r "$PCAP_FILE" \
        -T fields -E header=y -E separator=, \
        -e frame.time_epoch -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e frame.len \
        > "$CSV_FILE"

    # Pulizia pattern ripetuti tipo IP\,IP,IP\ → IP
    sed -E -i 's/([0-9]+\.[0-9]+\.[0-9]+\.[0-9]+)\\?,\1,\\?\1\\?/\1/g' "$CSV_FILE"

    echo "✅ Salvato e pulito: $CSV_FILE"
}

export -f process_pcap
export OVERWRITE

# Esecuzione parallela
printf "%s\n" "${PCAP_LIST[@]}" | xargs -n 1 -P "$NUM_CORES" -I {} bash -c 'process_pcap "$@"' _ {}
