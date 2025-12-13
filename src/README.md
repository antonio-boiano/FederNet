# Containernet Experiment Framework

A modular framework for running containerized network experiments with clear separation between network configuration and application logic.

## Key Features

- **Clean Separation of Concerns**: Network topology setup is completely independent from application commands
- **Flexible Role-Based Execution**: Define custom roles (server, client, coordinator, etc.) with their own commands
- **Backward Compatible**: Works with existing FL experiment configs
- **Smart Resource Management**: Automatic CPU allocation with spread/share modes via `ContainerResourceManager`

## Project Structure

```
your_project/
├── src/
│   ├── __init__.py
│   ├── application_runner.py    # Role-based command execution
│   ├── containernet_manager.py  # Network/container setup
│   ├── main.py                  # Entry point
│   ├── emulation.py             # (your existing file)
│   ├── network_debug.py         # (your existing file)
│   ├── quickstart_script.sh     # (your existing file)
│   └── resources/
│       ├── __init__.py
│       ├── performance.py       # Device/network profiles
│       ├── clean_containernet.py
│       ├── device_specs.json
│       ├── network_specs.json
│       └── disable_offload.sh
├── experiments/
│   └── your_config.yaml
└── output/
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py                                  │
│                   (Entry Point & Orchestration)                 │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          ▼                                       ▼
┌─────────────────────────┐         ┌─────────────────────────────┐
│  containernet_manager   │         │    application_runner       │
│                         │         │                             │
│ • Network topology      │         │ • Role definitions          │
│ • Container creation    │         │ • Command templates         │
│ • Resource allocation   │         │ • Execution orchestration   │
│ • Link configuration    │         │ • Logging                   │
└─────────────────────────┘         └─────────────────────────────┘
          │                                       │
          └───────────────┬───────────────────────┘
                          ▼
        ┌─────────────────────────────────────────┐
        │       resources/performance.py          │
        │                                         │
        │ • device_profile()                      │
        │ • network_profile()                     │
        │ • ContainerResourceManager              │
        │ • CPUAllocator                          │
        └─────────────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
    device_specs.json            network_specs.json
```

## Quick Start

### 1. Run with existing config (backward compatible)

Your existing config format works unchanged:

```bash
cd your_project
python3 -m src.main --config experiments/mqtt_iid.yaml
```

### 2. Interactive mode (debug network)

```bash
python3 -m src.main --config experiments/config.yaml --interactive
```

### 3. Network only (no application)

```bash
python3 -m src.main --config experiments/config.yaml --network-only
```

## Configuration

For more control, use explicit role definitions:

```yaml
containernet:
  num_containers: 25
  image_name: anboiano/fedopt:latest
  device_type: [rpi4, rpi4, rpi5, ...]
  network_type: [wifi_80211ac, 4g_lte, ...]
  host_single_core_score: 1554
  device_variance: 0.05

application:
  name: my_experiment
  
  variables:
    protocol: mqtt
    port: 1883
    rounds: 100
  
  roles:
    server:
      container_ids: [0]
      command: >-
        python3 -u run.py 
        --protocol {protocol} 
        --mode Server 
        --port {port} 
        --ip {container_ip}
      startup_delay: 0
    
    client:
      container_ids: "all_except_server"
      command: >-
        python3 -u run.py 
        --protocol {protocol} 
        --mode Client 
        --server {server_ip}:{port}
      startup_delay: 20
  
  role_order: [server, client]
```

## Role Configuration

Each role can specify:

| Field | Description |
|-------|-------------|
| `container_ids` | List of container IDs, or `"all"`, `"all_except_server"` |
| `command` | Command template to execute |
| `image` | Docker image override for containers in this role |
| `volumes` | Additional volume mounts (list of `"host:container"` strings) |
| `working_dir` | Working directory inside the container |
| `docker_args` | Custom Docker run arguments (dict) |
| `environment` | Environment variables (dict) |
| `shell` | Shell to use (`/bin/bash` or `/bin/sh` for minimal images) |
| `startup_delay` | Seconds to wait before starting this role |
| `wait_for_completion` | Whether to wait for commands to finish |
| `pre_commands` | Commands to run before the main command |
| `post_commands` | Commands to run after the main command |

### Minimal Images Requirements

If you are running a minimal images chanches are that the iproute2 is missing

You have two options, run a local container based on the minimal image:

```Dockerfile
FROM dockerhub-minimal-image
USER root
RUN apt-get update && apt-get install -y iproute2 iputils-ping
```

mount the static iproute2 we included:
```yaml
  volumes:
    - "./bin/ip:/sbin/ip"
```

### Per-Role Docker Image Override

Each role can use a different Docker image:

```yaml
roles:
  broker:
    container_ids: [0]
    image: eclipse-mosquitto:latest  # Different image for broker
    command: "mosquitto -v"
  
  server:
    container_ids: [1]
    # Uses default image_name from containernet section
    command: "python3 server.py --broker {ip_0}"
```

### Volume Mounts

Add custom volume mounts at global or per-role level:

```yaml
containernet:
  # Global volumes (mounted to all containers)
  volumes:
    - "./data:/app/data"
    - "./models:/app/models:ro"

application:
  roles:
    worker:
      container_ids: [1, 2, 3]
      # Role-specific volumes (in addition to global)
      volumes:
        - "./worker-config:/config"
      command: "python3 worker.py"
```

### Custom Docker Arguments

Pass custom arguments to Docker:

```yaml
containernet:
  # Global docker args
  docker_args:
    shm_size: "2g"

application:
  roles:
    app:
      container_ids: [0]
      working_dir: /workspace
      docker_args:
        user: "1000:1000"
        cap_add: ["NET_ADMIN"]
      command: "python3 app.py"
```

## Template Variables

Available in command templates:

| Variable | Description |
|----------|-------------|
| `{container_id}` | Container index (0, 1, 2, ...) |
| `{container_name}` | Container name |
| `{container_ip}` | This container's IP address |
| `{ip_N}` | **IP of container N by index** (e.g., `{ip_0}`, `{ip_1}`, `{ip_5}`) |
| `{cN_ip}` | Alternative syntax for container N's IP (e.g., `{c0_ip}`, `{c1_ip}`) |
| `{<role>_ip}` | First container's IP in named role (e.g., `{server_ip}`, `{broker_ip}`) |
| `{device_profile}` | Device profile name |
| `{network_profile}` | Network profile name |
| `{output_dir}` | Output directory path |
| `{index}` | Alias for container_id |
| `{my_ip}` | Alias for container_ip |
| Any key from `variables` | User-defined variables |

### IP Reference Examples

```yaml
# Reference container by index (most flexible)
command: "python3 run.py --server {ip_0} --backup {ip_1}"

# Reference container by role name (convenient)
command: "python3 run.py --broker {broker_ip} --coordinator {coordinator_ip}"

# Mix both approaches
command: "python3 run.py --primary {server_ip} --fallback {ip_2}"
```

## Volume Mounts

By default, the output directory is mounted at `/app/saved_output`. Add extra volumes:

```yaml
containernet:
  # Mount additional host directories into all containers
  volumes:
    - "/path/to/data:/app/data:ro"           # Read-only data
    - "/path/to/models:/app/models"          # Read-write
    - "/home/user/config:/app/config:ro"     # Config files
```

Per-container volume overrides:
```yaml
containernet:
  nodes:
    - id: 0
      volumes:
        - "/special/server/data:/app/server_data"
```

## Custom Docker Arguments

Pass custom flags to `docker run` for all containers:

```yaml
containernet:
  docker_args:
    # Any valid Containernet/Docker parameter
    shm_size: "2g"                    # Shared memory size
    cap_add: ["NET_ADMIN", "SYS_PTRACE"]
    security_opt: ["seccomp=unconfined"]
    device: ["/dev/nvidia0:/dev/nvidia0"]  # GPU passthrough
    runtime: "nvidia"                 # NVIDIA runtime
    ipc: "host"                       # IPC namespace
    network_mode: "host"              # Network mode (use carefully!)
```

Per-container docker_args overrides:
```yaml
containernet:
  nodes:
    - id: 0
      docker_args:
        shm_size: "4g"  # Server gets more shared memory
```

## Device Profiles

Device profiles are loaded from `resources/device_specs.json`:

| Profile | Cores | RAM | Single-Core Score |
|---------|-------|-----|-------------------|
| `rpi4` | 4 | 4 GiB | 187 |
| `rpi5` | 4 | 4 GiB | 516 |
| `jetson_nano` | 6 | 4 GiB | 629 |
| `intel_nuc8` | 4 | 8 GiB | 1150 |
| `smartphone_generic` | 8 | 4 GiB | 1000 |
| **`none`** | - | - | **No constraints (full host resources)** |

Use `none`, `null`, or `nan` to indicate no CPU/memory constraints - the container gets full access to host resources.

Add custom profiles to `device_specs.json`:
```json
{
  "my_device": {
    "cores": 4,
    "ram_gib": 8,
    "freq_mhz": 2000,
    "single_core_score": 800
  }
}
```

## Network Profiles

Network profiles are loaded from `resources/network_specs.json`:

| Tier | Profiles |
|------|----------|
| `low` | `adsl2_plus`, `satellite_geo`, `4g_lte`, `dsl_vdsl` |
| `middle` | `4g_lte_advanced`, `satellite_leo_starlink`, `cable_docsis_3.0` |
| `high` | `wifi_80211ac`, `5g_sub6`, `gigabit_ethernet` |
| `ultra` | `5g_mmwave`, `fiber_ftth`, `10gig_ethernet`, `wifi_80211ax` |
| **`none`** | **No limitations (unlimited bandwidth, zero delay/jitter/loss)** |

Use `none`, `null`, or `nan` to indicate no network limitations - the link has no artificial delay, bandwidth cap, jitter, or loss.

Each profile includes: `band_mbps`, `delay_ms`, `loss_percent`, `jitter_ms`

## Default Configuration Values

These are the default values when not specified in config:

```yaml
# Resource management
cpu_spread_threshold: 0.8    # Spread containers if total load < 80% of host cores
allow_overscaling: true      # Faster devices get more cores allocated
device_variance: 0.2         # Random variation in device performance

# Network (when network_type is none/unspecified)
default_delay_ms: 0          # No artificial delay
default_bandwidth_mbps: 0    # No bandwidth limit (0 = unlimited)
default_jitter_ms: 0         # No jitter

# Features
enable_tcpdump: false        # Packet capture disabled by default
enable_nat: true             # NAT for internet access enabled by default
```

## Internet Access

By default, containers can reach the internet via NAT. The framework automatically:
1. Enables IP forwarding on the host
2. Configures iptables masquerading for container traffic
3. Adds default routes in containers

To disable internet access:
```yaml
containernet:
  enable_nat: false
```

## Output Structure

Each experiment creates an output directory:

```
output/
└── experiments/
    └── 2024-01-15-10-30-00_fedopt_C24_wifi_80211ac_rpi4_mqtt_experiment/
        ├── config_original.yaml      # Original config
        ├── network_topology.json     # Network structure
        ├── application_config.json   # Resolved app config
        ├── commands_executed.json    # Command log
        ├── *_server_*.log            # Server logs
        ├── *_client_*.log            # Client logs (with device/network info)
        └── *.pcap                    # Packet captures (if enabled)
```

## License

MIT License
