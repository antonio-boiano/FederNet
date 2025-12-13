# %%
import json
import os
import math
import psutil
import warnings
from typing import List, Dict, Any, Sequence, Union, Optional, Tuple
import numpy as np
from dataclasses import dataclass, field


@dataclass
class CoreAllocation:
    """Tracks allocation state for a single CPU core."""
    core_id: int
    total_capacity: float = 1.0  # 1.0 = 100% of one core
    used_capacity: float = 0.0
    assigned_containers: List[str] = field(default_factory=list)
    
    @property
    def available_capacity(self) -> float:
        return max(0.0, self.total_capacity - self.used_capacity)
    
    @property
    def is_saturated(self) -> bool:
        return self.available_capacity < 0.01  # Less than 1% available


class CPUAllocator:
    """
    Manages CPU core allocation across multiple containers with sharing support.
    
    Strategy:
    - When cores are scarce: share cores between containers
    - When cores are plentiful: spread containers to avoid interference
    - Always respect performance ratios based on single_core_score
    """
    
    def __init__(self, host_cores: int, host_score: int, spread_threshold: float = 0.5, 
                 allow_overscaling: bool = False):
        """
        Args:
            host_cores: Number of physical CPU cores on host
            host_score: Host's single-core benchmark score
            spread_threshold: If total requested capacity < this * host_cores, spread containers
            allow_overscaling: If True, devices faster than host get more cores allocated
                              (e.g., 4-core device @ 2x speed → 8 host cores)
                              If False (default), ratio is capped at 1.0
        """
        self.host_cores = host_cores
        self.host_score = host_score
        self.spread_threshold = spread_threshold
        self.allow_overscaling = allow_overscaling
        self.cores: List[CoreAllocation] = [
            CoreAllocation(core_id=i) for i in range(host_cores)
        ]
        self.allocations: Dict[str, Dict] = {}  # container_name -> allocation info
        self._use_spread_mode = True  # Default to spread, will be set by set_mode()
        
    def reset(self):
        """Reset all allocations."""
        self.cores = [CoreAllocation(core_id=i) for i in range(self.host_cores)]
        self.allocations = {}
        self._use_spread_mode = True
    
    def set_mode(self, total_cores_needed: int, total_effective_capacity: float):
        """
        Set the allocation mode based on total planned load.
        
        Args:
            total_cores_needed: Total number of device cores across all containers
            total_effective_capacity: Total effective CPU capacity needed
        
        Mode decision:
        - SPREAD: If total_cores_needed <= host_cores (each container gets unique cores)
        - SHARED: If total_cores_needed > host_cores (containers must share cores)
        
        Additionally, if total_effective_capacity is low enough, we can pack more.
        """
        # If we need more cores than available, we MUST share
        if total_cores_needed > self.host_cores:
            self._use_spread_mode = False
            return
        
        # If total effective capacity is below threshold, spread for isolation
        if total_effective_capacity < self.host_cores * self.spread_threshold:
            self._use_spread_mode = True
        else:
            # Higher load - share to fit
            self._use_spread_mode = False
    
    def calculate_effective_capacity(self, device_cores: int, device_score: int) -> float:
        """
        Calculate effective host core capacity needed to emulate a device.
        
        If device is slower than host (device_score < host_score):
            - Each device core needs less than 1 host core
            - effective = device_cores * (device_score / host_score)
        
        If device is faster than host (device_score > host_score):
            - Each device core needs more than 1 host core (capped at device_cores)
            - We can't make the host faster, so we give full cores but note the limitation
        """
        perf_ratio = device_score / self.host_score
        return device_cores * min(perf_ratio, 1.0)
    
    def calculate_cpu_limit_per_core(self, device_score: int) -> float:
        """
        Calculate CPU limit per assigned core based on performance ratio.
        
        Returns a value between 0 and 1 representing max utilization per core.
        """
        perf_ratio = device_score / self.host_score
        # If device is slower, limit each core's usage
        # If device is faster or equal, allow full core usage
        return min(1.0, perf_ratio)
    
    def allocate(
        self,
        container_name: str,
        device_cores: int,
        device_score: int,
        prefer_spread: bool = None
    ) -> Dict[str, Any]:
        """
        Allocate CPU resources for a container.
        
        Strategy: Assign host cores based on device performance ratio.
        
        When allow_overscaling=False (default):
        - Slower device (score < host): Assign device_cores, limit each to ratio%
        - Faster device (score > host): Assign device_cores at 100% (capped)
        
        When allow_overscaling=True:
        - Slower device: Same as above
        - Faster device: Assign MORE cores to emulate the faster device
          e.g., 4-core device @ 2x speed → 8 host cores at 100% each
        
        Args:
            container_name: Unique identifier for the container
            device_cores: Number of cores the emulated device has
            device_score: Single-core benchmark score of the device
            prefer_spread: Override auto-detection of spread vs share mode
        
        Returns:
            Dict with cpuset_cpus, cpu_quota, cpu_period, and other Docker params
        """
        # Raw performance ratio (can be > 1.0)
        raw_perf_ratio = device_score / self.host_score
        
        if self.allow_overscaling:
            # Allow ratio > 1.0 - allocate more cores for faster devices
            perf_ratio = raw_perf_ratio
            
            if perf_ratio > 1.0:
                # Faster device: allocate more host cores
                # e.g., 4 device cores @ 2x speed → 8 host cores
                cores_to_assign = math.ceil(device_cores * perf_ratio)
                cpu_limit_per_core = 1.0  # Each host core runs at 100%
                effective_capacity = device_cores * perf_ratio  # Total work capacity
            else:
                # Slower device: normal behavior
                cores_to_assign = device_cores
                cpu_limit_per_core = perf_ratio
                effective_capacity = device_cores * perf_ratio
        else:
            # Default: cap ratio at 1.0
            perf_ratio = min(1.0, raw_perf_ratio)
            cores_to_assign = device_cores
            cpu_limit_per_core = perf_ratio
            effective_capacity = device_cores * perf_ratio
        
        # Use the allocator's global mode decision if not overridden
        if prefer_spread is None:
            prefer_spread = self._use_spread_mode
        
        # Find cores to allocate
        core_allocs, remaining = self._find_cores_for_device(
            num_cores_needed=cores_to_assign,
            capacity_per_core=cpu_limit_per_core,
            prefer_empty=prefer_spread
        )
        
        # Apply allocations
        assigned_core_ids = []
        total_allocated = 0.0
        is_oversubscribed = False
        
        for core_id, capacity in core_allocs:
            if self.cores[core_id].used_capacity + capacity > 1.0:
                is_oversubscribed = True
            
            self.cores[core_id].used_capacity += capacity
            self.cores[core_id].assigned_containers.append(container_name)
            assigned_core_ids.append(core_id)
            total_allocated += capacity
        
        if is_oversubscribed:
            warnings.warn(
                f"Container {container_name} causes core oversubscription. "
                f"Requested {cores_to_assign} cores at {cpu_limit_per_core:.2%} each, "
                f"allocated to cores {assigned_core_ids}. Performance may be degraded.",
                stacklevel=2
            )
        
        # Calculate Docker CPU constraints
        cpu_period = 100_000  # 100ms in microseconds
        cpu_quota = int(cpu_period * effective_capacity)
        nano_cpus = int(effective_capacity * 1e9)
        cpu_shares = int(effective_capacity * 1024)
        
        cpuset_cpus = ",".join(str(c) for c in sorted(set(assigned_core_ids)))
        
        allocation_info = {
            "container_name": container_name,
            "device_cores": device_cores,
            "device_score": device_score,
            "host_cores_assigned": len(assigned_core_ids),
            "performance_ratio": raw_perf_ratio,
            "effective_capacity": effective_capacity,
            "allocated_capacity": total_allocated,
            "cpu_limit_per_core": cpu_limit_per_core,
            "assigned_cores": assigned_core_ids,
            "cpuset_cpus": cpuset_cpus,
            "cpu_period": cpu_period,
            "cpu_quota": cpu_quota,
            "nano_cpus": nano_cpus,
            "cpu_shares": cpu_shares,
            "mode": "spread" if prefer_spread else "shared",
            "is_oversubscribed": is_oversubscribed,
            "overscaling_enabled": self.allow_overscaling,
        }
        
        self.allocations[container_name] = allocation_info
        return allocation_info
    
    def _find_cores_for_device(
        self, 
        num_cores_needed: int, 
        capacity_per_core: float,
        prefer_empty: bool = True
    ) -> Tuple[List[Tuple[int, float]], float]:
        """
        Find exactly `num_cores_needed` cores to assign for a multi-core device.
        
        Args:
            num_cores_needed: Number of host cores to assign (= device_cores)
            capacity_per_core: How much capacity each core will use (= perf_ratio)
            prefer_empty: If True, prefer empty cores (spread mode)
                          If False, prefer sharing already-used cores (shared mode)
        
        Returns:
            List of (core_id, capacity) tuples and remaining unallocated cores
        """
        allocations = []
        
        # Get list of all cores with their available capacity
        core_list = [(c.core_id, c.available_capacity, c.used_capacity) for c in self.cores]
        
        if prefer_empty:
            # SPREAD mode: prefer emptiest cores (minimize interference)
            # Sort by available capacity descending (most available first)
            core_list.sort(key=lambda x: -x[1])
        else:
            # SHARED mode: prefer cores that are already used but still have room
            # This allows multiple containers to share the same physical cores
            # Sort by: 1) has enough capacity for this allocation, 2) most used first
            # This packs containers onto fewer cores
            
            def shared_sort_key(core_info):
                core_id, available, used = core_info
                has_room = available >= capacity_per_core
                # Priority: cores with room that are already used > empty cores > cores without room
                if has_room:
                    return (0, -used)  # Has room, prefer more used
                else:
                    return (1, -available)  # No room, but might still work
            
            core_list.sort(key=shared_sort_key)
        
        # Assign exactly num_cores_needed cores
        for i in range(min(num_cores_needed, len(core_list))):
            core_id, avail, used = core_list[i]
            allocations.append((core_id, capacity_per_core))
        
        # Calculate how many cores we couldn't assign (if host has fewer cores than device)
        cores_assigned = len(allocations)
        remaining_cores = max(0, num_cores_needed - cores_assigned)
        
        return allocations, remaining_cores * capacity_per_core
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of all allocations."""
        return {
            "host_cores": self.host_cores,
            "host_score": self.host_score,
            "mode": "spread" if self._use_spread_mode else "shared",
            "core_usage": [
                {
                    "core_id": c.core_id,
                    "used_capacity": c.used_capacity,
                    "available_capacity": c.available_capacity,
                    "containers": c.assigned_containers
                }
                for c in self.cores
            ],
            "allocations": self.allocations
        }


def generate_container_configs(
    client_count: int,
    device_parma: Dict[str, int] = None,
    host_score: int = 1150,
    warn_level: int = 1,
    cpu_period: int = 100_000,
    allocator: CPUAllocator = None,
    container_idx: int = 0
) -> Dict[str, Any]:
    """
    Generate Docker HostConfig dicts for simulating devices or max host-capacity split.

    Args:
        client_count: Number of containers to create (>=1).
        device_parma: Device parameters dict with cores, ram_gib, freq_mhz, single_core_score
        host_score: Host single-core benchmark score
        warn_level: Warning verbosity (0=silent, 1=warnings)
        cpu_period: CFS period in microseconds (default 100ms)
        allocator: Optional CPUAllocator for smart core assignment
        container_idx: Index of this container (for naming in allocator)

    Returns:
        Dict with CPU/memory constraints for Docker API and environment variables.
    """
    if client_count < 1:
        raise ValueError("client_count must be >= 1")

    host_cores = os.cpu_count() or 1
    host_ram_total = psutil.virtual_memory().total
    host_freq_info = psutil.cpu_freq()
    host_max_freq = host_freq_info.max if host_freq_info else 2500

    if device_parma:
        dev = device_parma
        device_cores = dev["cores"]
        device_ram_bytes = dev["ram_gib"] * 1024**3
        device_freq = dev["freq_mhz"]
        device_score = dev.get("single_core_score", host_score)
        
        # Performance ratio: how much slower/faster is device vs host
        perf_ratio = device_score / host_score if device_score else 1.0
        
        # Use allocator if provided for smart core assignment
        if allocator is not None:
            alloc = allocator.allocate(
                container_name=f"container_{container_idx}",
                device_cores=device_cores,
                device_score=device_score
            )
            
            nano_cpus = alloc["nano_cpus"]
            cpu_quota = alloc["cpu_quota"]
            cpu_shares = alloc["cpu_shares"]
            cpuset_cpus = alloc["cpuset_cpus"]
            effective_cores = alloc["allocated_capacity"]
        else:
            # Legacy behavior: calculate without smart allocation
            effective_cores = device_cores * min(perf_ratio, 1.0)
            nano_cpus = int(effective_cores * 1e9)
            cpu_quota = int(cpu_period * effective_cores)
            cpu_shares = int(effective_cores * 1024)
            cpuset_cpus = None  # No specific core assignment
        
        memory = int(device_ram_bytes)
        env_threads = max(1, math.ceil(effective_cores))
        
        # Warn if device exceeds host (only when not using allocator, which handles this)
        if allocator is None:
            total_requested_cores = effective_cores * client_count
            total_requested_ram = device_ram_bytes * client_count
            
            if total_requested_cores > host_cores and warn_level > 0:
                warnings.warn(
                    f"Requested {total_requested_cores:.2f} effective cores "
                    f"({device_cores} cores @ {perf_ratio:.2f}x speed) × {client_count} clients, "
                    f"but host only has {host_cores} physical cores."
                )
            if total_requested_ram > host_ram_total and warn_level > 0:
                warnings.warn(
                    f"Requested {total_requested_ram / 1024**3:.2f} GiB RAM, "
                    f"but host only has {host_ram_total / 1024**3:.2f} GiB."
                )
            if device_freq > host_max_freq and warn_level > 0:
                warnings.warn(
                    f"Device freq {device_freq} MHz exceeds "
                    f"host max CPU freq {host_max_freq:.0f} MHz."
                )
    else:
        # No device specified: use minimal default allocation (1 core per container)
        # This prevents consuming all host resources when device profile is not set
        default_cores_per_container = 1  # Conservative default
        default_ram_per_container = 2 * 1024**3  # 2 GiB default
        
        if allocator is not None:
            # Use allocator with a default "lightweight" device profile
            alloc = allocator.allocate(
                container_name=f"container_{container_idx}",
                device_cores=default_cores_per_container,
                device_score=host_score  # Same as host = no slowdown
            )
            
            nano_cpus = alloc["nano_cpus"]
            cpu_quota = alloc["cpu_quota"]
            cpu_shares = alloc["cpu_shares"]
            cpuset_cpus = alloc["cpuset_cpus"]
            effective_cores = alloc["allocated_capacity"]
        else:
            # Legacy: split evenly but cap at reasonable limits
            cores_per_container = min(default_cores_per_container, host_cores / client_count)
            nano_cpus = int(cores_per_container * 1e9)
            cpu_quota = int(cpu_period * cores_per_container)
            cpu_shares = int(cores_per_container * 1024)
            cpuset_cpus = None
            effective_cores = cores_per_container
        
        memory = int(min(default_ram_per_container, host_ram_total / client_count))
        env_threads = max(1, math.ceil(effective_cores))

    # Environment variables to enforce thread limits
    env_vars = {
        'OMP_NUM_THREADS': str(env_threads),
        'MKL_NUM_THREADS': str(env_threads),
        'OPENBLAS_NUM_THREADS': str(env_threads),
        'NUMEXPR_NUM_THREADS': str(env_threads),
        'VECLIB_MAXIMUM_THREADS': str(env_threads),
        'TORCH_NUM_THREADS': str(env_threads),
    }

    config = {
        "NanoCPUs": nano_cpus,
        "Cpus": device_parma["cores"] if device_parma else cores_per_container,
        "EffectiveCpus": effective_cores,
        "CpuPeriod": cpu_period,
        "CpuQuota": cpu_quota,
        "CpuShares": cpu_shares,
        "CpusetCpus": cpuset_cpus,  # NEW: specific core assignment
        "Memory": memory,
        "Environment": env_vars,
    }

    return config


def plan_container_allocations(
    client_count: int,
    devices: List[Dict[str, int]],
    host_score: int = 1150,
    host_cores: int = None,
    spread_threshold: float = 0.5
) -> Tuple[CPUAllocator, List[Dict[str, Any]]]:
    """
    Plan CPU allocations for multiple containers before creating them.
    
    This allows optimal distribution of cores based on total requirements.
    
    Args:
        client_count: Number of containers
        devices: List of device specs (one per container, or single spec for all)
        host_score: Host's single-core benchmark score
        host_cores: Number of host cores (auto-detected if None)
        spread_threshold: Threshold for spread vs share mode
    
    Returns:
        Tuple of (CPUAllocator, list of configs)
    """
    if host_cores is None:
        host_cores = os.cpu_count() or 1
    
    allocator = CPUAllocator(
        host_cores=host_cores,
        host_score=host_score,
        spread_threshold=spread_threshold
    )
    
    configs = []
    for idx in range(client_count):
        # Get device spec for this container
        if isinstance(devices, list) and len(devices) > 0:
            device = devices[idx] if idx < len(devices) else devices[-1]
        else:
            device = devices
        
        config = generate_container_configs(
            client_count=client_count,
            device_parma=device,
            host_score=host_score,
            allocator=allocator,
            container_idx=idx
        )
        configs.append(config)
    
    return allocator, configs


def load_profile_data(json_file):
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            json_path = os.path.join(script_dir, json_file)
            return load_profile_data(json_path)
        except:
            print(f'I am in {os.getcwd()}')
            raise FileNotFoundError(f"File '{json_file}' not found.")


def extract_param_stats(param_dict):
    """
    Extract mean and std-dev for each parameter.
    If 'range' exists, use (max-min)/4 as std-dev (approx for normal covering 95%).
    If only 'typical' and no range, use typical and 20% as fallback stdev.
    """
    means, stddevs, keys = [], [], []
    for k, v in param_dict.items():
        if "typical" in v and "range" in v:
            mean = v["typical"]
            r = v["range"]
            stddev = (r[1] - r[0]) / 4 if r[1] > r[0] else 0.2 * abs(mean)
        elif "typical" in v:
            mean = v["typical"]
            stddev = 0.2 * abs(mean)
        else:
            continue
        means.append(mean)
        stddevs.append(stddev)
        keys.append(k)
    return np.array(means), np.array(stddevs), keys


def deep_get(d: dict, key: str):
    """
    Depth-first search for *key* in a nested dict/list structure.
    Returns (value, path) on success; raises KeyError if not found.
    """
    stack = [(d, [])]

    while stack:
        node, path = stack.pop()

        if isinstance(node, dict):
            if key in node:
                return node[key], path + [key]
            for k, v in node.items():
                stack.append((v, path + [k]))
        elif isinstance(node, list):
            for i, item in enumerate(node):
                stack.append((item, path + [i]))

    raise KeyError(f"'{key}' not found anywhere in structure")


def sample_profile(
    data,
    profile: str,
    idx: int,
    conn_type: str = None,
    corr: float = None
):
    """
    Given data loaded from JSON, sample a realistic set of parameters for a
    given profile and optionally a specific connection type, using a
    repeatable seed (idx).

    Returns dict mapping param name to sampled value.
    """
    if profile not in data:
        _, full_prof = deep_get(data, profile)
        profile = full_prof[0]
        conn_type = full_prof[1]
    p_data = data[profile]

    if conn_type is None:
        all_means = []
        all_stds = []
        keys = None
        for ct in p_data:
            means, stds, these_keys = extract_param_stats(p_data[ct])
            if keys is None:
                keys = these_keys
            else:
                assert keys == these_keys, "Parameter keys must match across connection types"
            all_means.append(means)
            all_stds.append(stds)
        means = np.mean(all_means, axis=0)
        stddevs = np.mean(all_stds, axis=0)
    else:
        if conn_type not in p_data:
            raise ValueError(f"Connection type '{conn_type}' not in profile '{profile}'")
        means, stddevs, keys = extract_param_stats(p_data[conn_type])

    rng = np.random.RandomState(idx)
    if corr is not None:
        n = len(means)
        cov = np.full((n, n), corr, dtype=float)
        np.fill_diagonal(cov, 1.0)
        cov = cov * np.outer(stddevs, stddevs)
        sample = rng.multivariate_normal(mean=means, cov=cov)
    else:
        sample = rng.normal(loc=means, scale=stddevs)

    result = {}
    for i, k in enumerate(keys):
        v = float(sample[i])
        v = max(0.0, v)
        result[k] = v
    return result


def perturb_device(
    data: dict,
    device_name: str,
    idx: int,
    variation: float = 0.2,
    corr: float = 0.5,
    lock_keys: list = None
) -> dict:
    """
    Perturb the numeric specs of a given device using a multivariate normal distribution.
    """
    if device_name not in data:
        raise ValueError(f"Device '{device_name}' not in data.")

    lock_keys = set(lock_keys or [])
    device = data[device_name]
    keys = [k for k in device if k not in lock_keys]
    locked = {k: device[k] for k in device if k in lock_keys}

    means = np.array([device[k] for k in keys], dtype=float)
    n = len(means)
    if n == 0:
        return dict(locked)

    sigma = variation * np.abs(means)
    cov = np.full((n, n), corr, dtype=float)
    np.fill_diagonal(cov, 1.0)
    cov = cov * np.outer(sigma, sigma)
    rng = np.random.RandomState(idx)
    sample = rng.multivariate_normal(mean=means, cov=cov)

    perturbed = {}
    for i, k in enumerate(keys):
        v = sample[i]
        if device[k] == int(device[k]):
            v = int(round(v))
        v = max(1, v)
        perturbed[k] = v

    perturbed.update(locked)
    return perturbed


def network_profile(
    profile: Union[str, List[str]],
    idx: int,
    conn_type: str = None,
    corr: float = None
):
    if isinstance(profile, List):
        try:
            profile = profile[idx - 1]
        except IndexError:
            profile = profile[-1]
    data = load_profile_data("network_specs.json")
    return sample_profile(data, profile, idx, conn_type, corr)


def device_profile(
    idx: int,
    host_score: int,
    device_name: Union[str, List[str]],
    client_count: int = 10,
    variation: float = 0.2,
    corr: float = 0.5,
    allocator: CPUAllocator = None
):
    """
    Get device profile and generate container config.
    
    Args:
        idx: Container index
        host_score: Host's single-core benchmark score
        device_name: Device name or list of device names
        client_count: Total number of clients
        variation: Variation in device specs
        corr: Correlation between device parameters
        allocator: Optional CPUAllocator for smart core assignment
    """
    dev = None
    if device_name is not None:
        data = load_profile_data("device_specs.json")
        if isinstance(device_name, list):
            try:
                device_name = device_name[idx - 1]
            except IndexError:
                if idx == 0:
                    return generate_container_configs(1, host_score=host_score)
                device_name = device_name[-1]
        dev = perturb_device(data, device_name, idx, variation, corr, ["cores"])
    
    return generate_container_configs(
        client_count,
        device_parma=dev,
        host_score=host_score,
        allocator=allocator,
        container_idx=idx
    )


# =============================================================================
# High-level API for Containernet integration
# =============================================================================

class ContainerResourceManager:
    """
    High-level manager for container CPU/memory resources.
    
    Use this class to plan and allocate resources for all containers
    before creating them, ensuring optimal core distribution.
    
    Example:
        manager = ContainerResourceManager(host_score=1079, client_count=10)
        
        # Add containers with their device profiles
        for i in range(10):
            manager.add_container(f"client_{i}", device_name="rpi4")
        
        # Plan allocations (determines spread vs share based on total load)
        manager.plan_allocations()
        
        # Get config for each container
        for i in range(10):
            config = manager.get_container_config(f"client_{i}")
            # Use config["CpusetCpus"], config["CpuQuota"], etc.
    """
    
    def __init__(
        self,
        host_score: int = 1150,
        client_count: int = 10,
        host_cores: int = None,
        spread_threshold: float = 0.5,
        device_variation: float = 0.2,
        allow_overscaling: bool = False
    ):
        """
        Args:
            host_score: Host's single-core benchmark score
            client_count: Number of containers
            host_cores: Number of host cores (auto-detected if None)
            spread_threshold: Threshold for spread vs share mode
            device_variation: Variation in device specs for perturbation
            allow_overscaling: If True, devices faster than host get more cores
                              (e.g., 4-core device @ 2x speed → 8 host cores)
        """
        self.host_score = host_score
        self.client_count = client_count
        self.host_cores = host_cores or (os.cpu_count() or 1)
        self.spread_threshold = spread_threshold
        self.device_variation = device_variation
        self.allow_overscaling = allow_overscaling
        
        self.allocator = CPUAllocator(
            host_cores=self.host_cores,
            host_score=self.host_score,
            spread_threshold=self.spread_threshold,
            allow_overscaling=self.allow_overscaling
        )
        
        self.containers: Dict[str, Dict] = {}  # name -> device_spec
        self.configs: Dict[str, Dict] = {}  # name -> container_config
        self._planned = False
    
    def add_container(
        self,
        name: str,
        device_name: str = None,
        device_spec: Dict = None,
        idx: int = None,
        default_cores: int = 1,
        default_ram_gib: float = 2.0
    ):
        """
        Add a container to be allocated.
        
        Args:
            name: Unique container name
            device_name: Name of device profile to use
            device_spec: Direct device specification (overrides device_name)
            idx: Index for perturbation seed (defaults to container count)
            default_cores: Default cores if no device specified
            default_ram_gib: Default RAM in GiB if no device specified
        """
        if self._planned:
            raise RuntimeError("Cannot add containers after planning. Call reset() first.")
        
        if idx is None:
            idx = len(self.containers)
        
        if device_spec:
            self.containers[name] = device_spec
        elif device_name:
            data = load_profile_data("device_specs.json")
            dev = perturb_device(
                data, device_name, idx,
                variation=self.device_variation,
                lock_keys=["cores"]
            )
            self.containers[name] = dev
        else:
            # No device - use minimal default allocation instead of equal share
            # This prevents consuming all host resources
            self.containers[name] = {
                "cores": default_cores,
                "ram_gib": default_ram_gib,
                "freq_mhz": 2500,  # Assume modern CPU
                "single_core_score": self.host_score  # Same as host = full speed
            }
    
    def plan_allocations(self):
        """
        Plan CPU allocations for all added containers.
        
        This determines whether to spread or share cores based on total load
        BEFORE allocating any containers.
        """
        self.allocator.reset()
        self._planned = True
        
        # Calculate total cores and effective capacity needed
        total_cores_needed = 0
        total_effective = 0.0
        
        for name, dev in self.containers.items():
            if dev:
                perf_ratio = dev.get("single_core_score", self.host_score) / self.host_score
                
                if self.allow_overscaling and perf_ratio > 1.0:
                    # Faster device needs more cores
                    cores_for_device = math.ceil(dev["cores"] * perf_ratio)
                else:
                    cores_for_device = dev["cores"]
                    perf_ratio = min(1.0, perf_ratio)
                
                total_cores_needed += cores_for_device
                total_effective += dev["cores"] * perf_ratio
            else:
                # Fallback: assume 1 core per container
                total_cores_needed += 1
                total_effective += 1
        
        # Set the mode BEFORE allocating
        self.allocator.set_mode(total_cores_needed, total_effective)
        
        # Allocate each container
        for idx, (name, dev) in enumerate(self.containers.items()):
            if dev:
                config = generate_container_configs(
                    client_count=self.client_count,
                    device_parma=dev,
                    host_score=self.host_score,
                    allocator=self.allocator,
                    container_idx=idx
                )
            else:
                config = generate_container_configs(
                    client_count=self.client_count,
                    device_parma=None,
                    host_score=self.host_score,
                    allocator=self.allocator,
                    container_idx=idx
                )
            self.configs[name] = config
    
    def get_container_config(self, name: str) -> Dict[str, Any]:
        """Get the Docker config for a container after planning."""
        if not self._planned:
            raise RuntimeError("Must call plan_allocations() before getting configs.")
        return self.configs.get(name)
    
    def get_all_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get all container configs."""
        if not self._planned:
            raise RuntimeError("Must call plan_allocations() before getting configs.")
        return self.configs.copy()
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of CPU core allocations."""
        return self.allocator.get_allocation_summary()
    
    def reset(self):
        """Reset manager for re-planning."""
        self.allocator = CPUAllocator(
            host_cores=self.host_cores,
            host_score=self.host_score,
            spread_threshold=self.spread_threshold,
            allow_overscaling=self.allow_overscaling
        )
        self.containers.clear()
        self.configs.clear()
        self._planned = False


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    # Example 1: Using the high-level ContainerResourceManager
    print("=" * 60)
    print("Example 1: ContainerResourceManager")
    print("=" * 60)
    
    manager = ContainerResourceManager(
        host_score=1079,
        client_count=10,
        host_cores=8,  # Simulating 8-core host
        spread_threshold=0.5
    )
    
    # Add 10 containers with Raspberry Pi 4 specs
    # RPi4: 4 cores, score ~500 vs host score 1079
    # Effective cores per RPi4 = 4 * (500/1079) ≈ 1.85
    # Total for 10 = ~18.5 effective cores on 8 physical
    # This will trigger SHARED mode
    
    rpi4_spec = {
        "cores": 4,
        "ram_gib": 4,
        "freq_mhz": 1500,
        "single_core_score": 500
    }
    
    for i in range(10):
        manager.add_container(f"client_{i}", device_spec=rpi4_spec)
    
    manager.plan_allocations()
    
    print("\nAllocation Summary:")
    summary = manager.get_allocation_summary()
    
    print(f"\nHost: {summary['host_cores']} cores, score {summary['host_score']}")
    print("\nCore Usage:")
    for core in summary['core_usage']:
        print(f"  Core {core['core_id']}: {core['used_capacity']:.2f} used, "
              f"{core['available_capacity']:.2f} available, "
              f"containers: {core['containers']}")
    
    print("\nContainer Configs:")
    for name, config in manager.get_all_configs().items():
        print(f"  {name}: cpuset={config['CpusetCpus']}, "
              f"quota={config['CpuQuota']}, "
              f"effective={config['EffectiveCpus']:.2f}")
    
    # Example 2: Spread mode with lighter load
    print("\n" + "=" * 60)
    print("Example 2: Spread mode with 3 light containers")
    print("=" * 60)
    
    manager2 = ContainerResourceManager(
        host_score=1079,
        client_count=3,
        host_cores=8,
        spread_threshold=0.5
    )
    
    # Only 3 containers, each needing ~1.85 effective cores
    # Total ≈ 5.5 effective cores on 8 physical
    # This should trigger SPREAD mode
    
    for i in range(3):
        manager2.add_container(f"client_{i}", device_spec=rpi4_spec)
    
    manager2.plan_allocations()
    
    print("\nAllocation Summary (Spread Mode):")
    summary2 = manager2.get_allocation_summary()
    
    print("\nCore Usage:")
    for core in summary2['core_usage']:
        if core['used_capacity'] > 0:
            print(f"  Core {core['core_id']}: {core['used_capacity']:.2f} used, "
                  f"containers: {core['containers']}")
    
    print("\nContainer Configs:")
    for name, config in manager2.get_all_configs().items():
        print(f"  {name}: cpuset={config['CpusetCpus']}, "
              f"quota={config['CpuQuota']}, "
              f"effective={config['EffectiveCpus']:.2f}")