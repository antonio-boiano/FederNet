"""
ContainernetManager: Handles all Containernet network and container setup.
Completely decoupled from application-specific logic.
Uses existing resources.performance module for device/network profiles.
"""

from mininet.net import Containernet
from mininet.node import Controller, Node
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import info, error
from mininet.node import OVSSwitch as _OVSSwitch

import ipaddress
import itertools
import os
import datetime
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

# Import from resources submodule (src/resources/)
from .resources.performance import (
    device_profile,
    network_profile,
    ContainerResourceManager,
    CPUAllocator,
    load_profile_data,
    perturb_device
)

# Import cleanup utility
from .resources import clean_containernet


class PatchedOVSSwitch(_OVSSwitch):
    """OVS Switch with version fix for Mininet compatibility."""
    OVSVersion = '2.5'


class LinuxRouter(Node):
    """Linux-based router node with IP forwarding enabled."""
    
    def config(self, **params):
        super(LinuxRouter, self).config(**params)
        self.cmd('sysctl net.ipv4.ip_forward=1')
        self.cmd('ethtool -K', self, 'gro', 'off', 'tx', 'off', 'rx', 'off')
    
    def terminate(self):
        self.cmd('sysctl net.ipv4.ip_forward=0')
        super(LinuxRouter, self).terminate()


@dataclass
class ContainerConfig:
    """Configuration for a single container."""
    id: int
    name: str
    ip_address: str
    subnet: str
    default_gateway: str
    image: str
    
    # Resource constraints (from device profile)
    cpu_period: int = 0
    cpu_quota: int = 0
    cpu_shares: int = 0
    cpuset_cpus: Optional[str] = None
    nano_cpus: int = 0
    memory_limit: int = 0  # bytes
    effective_cpus: float = 0.0
    
    # Network characteristics (from network profile)
    network_profile_name: Optional[str] = None
    link_delay_ms: float = 12.0
    link_bandwidth_mbps: float = 80.0
    link_jitter_ms: float = 2.0
    link_loss_percent: float = 0.0
    
    # Device profile
    device_profile_name: Optional[str] = None
    
    # Custom settings
    volumes: List[str] = field(default_factory=list)
    environment: Dict[str, str] = field(default_factory=dict)
    docker_args: Dict[str, Any] = field(default_factory=dict)  # Custom Docker run flags
    
    # Runtime references (set after container creation)
    docker_container: Any = None
    router: Any = None
    
    # Docker network info (captured during routing setup for internet access)
    docker_gateway: Optional[str] = None
    docker_iface: Optional[str] = None


@dataclass
class NetworkConfig:
    """Global network configuration."""
    num_containers: int
    default_image: str
    output_dir: str
    
    # Default link parameters (used when network_type is None)
    default_delay_ms: float = 0.0      # No artificial delay
    default_bandwidth_mbps: float = 0   # 0 = no bandwidth limit
    default_jitter_ms: float = 0.0      # No jitter
    
    # Resource management (passed to ContainerResourceManager)
    host_single_core_score: int = 1079
    device_variance: float = 0.2
    cpu_spread_threshold: float = 0.8   # Spread containers if total load < 80% of host cores
    default_cores_per_container: int = 1
    default_ram_gib: float = 2.0
    allow_overscaling: bool = True      # Faster devices get more cores
    
    # Device and network profiles (list or single value)
    # None = no constraints (full host resources / unlimited network)
    device_type: Optional[Any] = None  # str or List[str] or None
    network_type: Optional[Any] = None  # str or List[str] or None
    
    # Optional features
    enable_tcpdump: bool = False
    enable_nat: bool = False  # Enable internet access from containers (requires 'ip' command)
    
    # Volume mounts (in addition to output_dir)
    # Format: ["/host/path:/container/path", "/host/path2:/container/path2:ro"]
    extra_volumes: List[str] = field(default_factory=list)
    
    # Custom Docker run arguments (applied to all containers)
    # These are passed directly to docker run
    docker_args: Dict[str, Any] = field(default_factory=dict)
    
    # Per-container overrides (indexed by container id)
    container_overrides: Dict[int, Dict] = field(default_factory=dict)


class RouterWrapper:
    """Wrapper for router management."""
    
    def __init__(self, router_id: int, network_ip: ipaddress.IPv4Network, net: Containernet, max_eth: int = 100):
        self.id = router_id
        self.name = f"r{router_id}"
        self.network_ip = network_ip
        self.main_ip = f'{next(network_ip.hosts())}/24'
        self.eth_available = [f'{self.name}-eth{eth}' for eth in range(max_eth)]
        self.eth_used = []
        self.routing_bindings = []
        self.switch = None
        self.router = net.addHost(self.name, cls=LinuxRouter, ip=self.main_ip)
        self._net = net
    
    def get_eth(self) -> str:
        eth = self.eth_available.pop(0)
        self.eth_used.append(eth)
        return eth
    
    def add_binding(self, dest_network, via_ip, interface):
        self.routing_bindings.append((dest_network, via_ip, interface))
    
    def add_switch(self):
        self.switch = self._net.addSwitch(f's{self.id}')
        self.switch.cmd('ethtool -K', self.switch, 'gro off', 'tx off', 'rx off')
        self._net.addLink(
            self.switch, 
            self.router,
            intfName=self.get_eth(),
            params2={'ip': self.main_ip}
        )
        return self.switch


class ContainernetManager:
    """
    Manages Containernet network topology and container lifecycle.
    Uses resources.performance for device/network profile handling.
    """
    
    def __init__(self, config: NetworkConfig):
        self.config = config
        self.net = Containernet(controller=Controller, switch=PatchedOVSSwitch)
        self.routers: List[RouterWrapper] = []
        self.containers: Dict[int, ContainerConfig] = {}
        self.switches = []
        self.resource_manager: Optional[ContainerResourceManager] = None
        self._started = False
        
        # Ensure output directory exists
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _initialize_resource_manager(self) -> None:
        """Initialize the ContainerResourceManager for smart CPU allocation."""
        host_cores = os.cpu_count() or 1
        
        self.resource_manager = ContainerResourceManager(
            host_score=self.config.host_single_core_score,
            client_count=self.config.num_containers,
            host_cores=host_cores,
            spread_threshold=self.config.cpu_spread_threshold,
            device_variation=self.config.device_variance,
            allow_overscaling=self.config.allow_overscaling
        )
        
        info(f"*** Initialized CPU Resource Manager: {host_cores} cores, "
             f"score {self.config.host_single_core_score}, "
             f"spread threshold {self.config.cpu_spread_threshold}\n")
        
        # Pre-register all containers with their device profiles
        for idx in range(self.config.num_containers):
            container_name = f"c{idx}"  # Short name to match container creation
            device_name = self._get_device_type_for_container(idx)
            
            if device_name:
                self.resource_manager.add_container(
                    container_name, 
                    device_name=device_name, 
                    idx=idx
                )
                info(f"*** Registered {container_name} with device profile: {device_name}\n")
            else:
                self.resource_manager.add_container(
                    container_name,
                    device_spec=None,
                    idx=idx,
                    default_cores=self.config.default_cores_per_container,
                    default_ram_gib=self.config.default_ram_gib
                )
                info(f"*** Registered {container_name} with default allocation\n")
        
        # Plan all allocations
        self.resource_manager.plan_allocations()
        
        # Log allocation summary
        summary = self.resource_manager.get_allocation_summary()
        info(f"\n*** CPU Allocation Summary:\n")
        info(f"    Mode: {summary.get('mode', 'unknown')}\n")
        
        active_cores = [c for c in summary.get('core_usage', []) if c['used_capacity'] > 0]
        if active_cores:
            info(f"    Active cores: {len(active_cores)}\n")
    
    def _get_device_type_for_container(self, container_id: int) -> Optional[str]:
        """Get device type for a specific container. Returns None for no constraints."""
        # Check per-container overrides first
        overrides = self.config.container_overrides.get(container_id, {})
        if 'device_type' in overrides:
            val = overrides['device_type']
            # Handle None/none/nan as "no constraints"
            if val is None or (isinstance(val, str) and val.lower() in ('none', 'nan', '')):
                return None
            return val
        
        # Use global device_type setting
        device_type = self.config.device_type
        
        # Handle None/none/nan as "no constraints"
        if device_type is None:
            return None
        if isinstance(device_type, str) and device_type.lower() in ('none', 'nan', ''):
            return None
        
        if isinstance(device_type, list):
            try:
                # Container 0 is typically server, clients start at index 1
                if container_id == 0:
                    val = device_type[0] if device_type else None
                else:
                    val = device_type[container_id - 1] if container_id <= len(device_type) else device_type[-1]
                # Check if this specific value is None/none/nan
                if val is None or (isinstance(val, str) and val.lower() in ('none', 'nan', '')):
                    return None
                return val
            except IndexError:
                val = device_type[-1] if device_type else None
                if val is None or (isinstance(val, str) and val.lower() in ('none', 'nan', '')):
                    return None
                return val
        
        return device_type
    
    def _get_network_type_for_container(self, container_id: int) -> Optional[str]:
        """Get network type for a specific container. Returns None for no limitations."""
        # Check per-container overrides first
        overrides = self.config.container_overrides.get(container_id, {})
        if 'network_type' in overrides:
            val = overrides['network_type']
            # Handle None/none/nan as "no limitations"
            if val is None or (isinstance(val, str) and val.lower() in ('none', 'nan', '')):
                return None
            return val
        
        # Use global network_type setting
        network_type = self.config.network_type
        
        # Handle None/none/nan as "no limitations"
        if network_type is None:
            return None
        if isinstance(network_type, str) and network_type.lower() in ('none', 'nan', ''):
            return None
        
        if isinstance(network_type, list):
            try:
                if container_id == 0:
                    val = network_type[0] if network_type else None
                else:
                    val = network_type[container_id - 1] if container_id <= len(network_type) else network_type[-1]
                # Check if this specific value is None/none/nan
                if val is None or (isinstance(val, str) and val.lower() in ('none', 'nan', '')):
                    return None
                return val
            except IndexError:
                val = network_type[-1] if network_type else None
                if val is None or (isinstance(val, str) and val.lower() in ('none', 'nan', '')):
                    return None
                return val
        
        return network_type
    
    def setup_network(self) -> None:
        """Set up the complete network topology."""
        info('*** Setting up network topology\n')
        
        # Initialize resource manager first
        self._initialize_resource_manager()
        
        # Add controller
        self.net.addController('c0', port=6654)
        
        # Create routers
        self._create_routers()
        
        # Create switches
        self._create_switches()
        
        # Create router-to-router links
        self._create_router_links()
        
        # Configure routing tables
        self._configure_routing()
        
        info('*** Network topology ready\n')
    
    def _create_routers(self) -> None:
        """Create routers for each container."""
        info('*** Adding routers\n')
        for i in range(self.config.num_containers):
            network = ipaddress.ip_network(f'10.0.{i}.0/24')
            router = RouterWrapper(i, network, self.net, self.config.num_containers + 10)
            self.routers.append(router)
    
    def _create_switches(self) -> None:
        """Create switches and connect to routers."""
        info('*** Adding switches\n')
        self.switches = [router.add_switch() for router in self.routers]
    
    def _create_router_links(self) -> None:
        """Create mesh links between routers."""
        info('*** Adding router-router links\n')
        
        for router1, router2 in itertools.combinations(self.routers, 2):
            intf1 = router1.get_eth()
            intf2 = router2.get_eth()
            
            # Calculate link IPs
            link_id = int(''.join(sorted([str(router1.id), str(router2.id)])))
            ip1 = f'10.{link_id}.0.1'
            ip2 = f'10.{link_id}.0.2'
            
            # Add routing bindings
            router1.add_binding(router2.network_ip, ip2, intf1)
            router2.add_binding(router1.network_ip, ip1, intf2)
            
            # Get link parameters using network_profile from resources.performance
            delay, bw, jitter, loss = self._get_link_params(router2.id)
            
            # Build link parameters, only include non-zero/non-None values
            link_params = {
                'intfName1': intf1,
                'intfName2': intf2,
                'params1': {'ip': f'{ip1}/24'},
                'params2': {'ip': f'{ip2}/24'},
                'cls': TCLink,
            }
            
            # Only add delay/jitter if > 0
            if delay and delay > 0:
                link_params['delay'] = f'{delay}ms'
            if jitter and jitter > 0:
                link_params['jitter'] = f'{jitter}ms'
            # Only add bandwidth if specified (None = no limit)
            if bw is not None:
                link_params['bw'] = bw
            # Only add loss if > 0
            if loss and loss > 0:
                link_params['loss'] = loss
            
            self.net.addLink(router1.router, router2.router, **link_params)
    
    def _get_link_params(self, container_id: int) -> tuple:
        """
        Get link parameters for a container using network_profile.
        
        Returns:
            tuple: (delay_ms, bandwidth_mbps, jitter_ms, loss_percent)
            
        When network_type is None: returns (0, None, 0, 0) meaning no limitations.
        bandwidth=None tells TCLink to not apply bandwidth shaping.
        """
        # Check for per-container link overrides
        overrides = self.config.container_overrides.get(container_id, {})
        link_config = overrides.get('link', {})
        
        if link_config:
            # Use explicit link configuration
            delay = link_config.get('delay_ms', self.config.default_delay_ms) / 2
            bw = link_config.get('bandwidth_mbps', self.config.default_bandwidth_mbps)
            jitter = link_config.get('jitter_ms', self.config.default_jitter_ms)
            loss = link_config.get('loss_percent', 0.0)
            # Convert 0 bandwidth to None (no limit)
            if bw == 0:
                bw = None
            return delay, bw, jitter, loss
        
        # Use network profile from resources.performance
        network_type = self._get_network_type_for_container(container_id)
        
        if network_type:
            try:
                ntw_prof = network_profile(network_type, container_id)
                delay = ntw_prof.get('delay_ms', self.config.default_delay_ms) / 2
                bw = ntw_prof.get('band_mbps', self.config.default_bandwidth_mbps)
                jitter = ntw_prof.get('jitter_ms', self.config.default_jitter_ms)
                loss = ntw_prof.get('loss_percent', 0.0)
                # Convert 0 bandwidth to None (no limit)
                if bw == 0:
                    bw = None
                return delay, bw, jitter, loss
            except Exception as e:
                info(f"*** Warning: Could not load network profile {network_type}: {e}\n")
        
        # No network profile = no limitations
        # Return None for bandwidth (no shaping), 0 for delay/jitter/loss
        default_bw = self.config.default_bandwidth_mbps if self.config.default_bandwidth_mbps > 0 else None
        return (
            self.config.default_delay_ms / 2 if self.config.default_delay_ms > 0 else 0,
            default_bw,
            self.config.default_jitter_ms,
            0.0
        )
    
    def _configure_routing(self) -> None:
        """Configure routing tables on all routers."""
        info('*** Configuring routing tables\n')
        cmd_template = "ip route add {dest} via {via} dev {interface}"
        
        for router in self.routers:
            for dest, via, interface in router.routing_bindings:
                router.router.cmd(cmd_template.format(dest=dest, via=via, interface=interface))
    
    def _normalize_volume_path(self, volume_spec: str) -> str:
        """
        Convert relative paths in volume specifications to absolute paths.
        Docker requires absolute paths for host mounts.
        
        Args:
            volume_spec: Volume specification in format "host_path:container_path[:options]"
        
        Returns:
            Volume specification with absolute host path
        """
        parts = volume_spec.split(':')
        if len(parts) < 2:
            return volume_spec
        
        host_path = parts[0]
        container_path = parts[1]
        options = parts[2] if len(parts) > 2 else None
        
        # Convert relative path to absolute
        if not os.path.isabs(host_path):
            host_path = os.path.abspath(host_path)
        
        # Reassemble volume specification
        if options:
            return f"{host_path}:{container_path}:{options}"
        else:
            return f"{host_path}:{container_path}"
    
    def _deduplicate_volumes(self, volumes: List[str]) -> List[str]:
        """
        Deduplicate volumes by container mount point.
        Later entries override earlier ones (role volumes override global).
        
        Args:
            volumes: List of volume specifications
        
        Returns:
            Deduplicated list of volumes
        """
        seen_mount_points = {}
        
        for vol in volumes:
            parts = vol.split(':')
            if len(parts) >= 2:
                container_path = parts[1]
                # Later entries override earlier ones
                seen_mount_points[container_path] = vol
            else:
                # Keep volumes without proper format as-is
                seen_mount_points[vol] = vol
        
        return list(seen_mount_points.values())
    
    def create_containers(self) -> Dict[int, ContainerConfig]:
        """Create all containers with their configurations."""
        info('*** Creating containers\n')
        
        for router in self.routers:
            container_config = self._build_container_config(router)
            docker_container = self._create_docker_container(container_config)
            container_config.docker_container = docker_container
            container_config.router = router
            self.containers[router.id] = container_config
        
        return self.containers
    
    def _build_container_config(self, router: RouterWrapper) -> ContainerConfig:
        """Build configuration for a single container."""
        container_id = router.id
        container_name = f"c{container_id}"  # Short name to avoid interface name limit (15 chars)
        overrides = self.config.container_overrides.get(container_id, {})
        
        # Base network info
        network_ip = router.network_ip
        gateway = str(list(network_ip.hosts())[0])
        container_ip = str(list(network_ip.hosts())[99])  # .100 address
        
        # Get resource constraints from resource manager
        rm_config = None
        if self.resource_manager:
            rm_config = self.resource_manager.get_container_config(container_name)
        
        # Build volumes list: output_dir + extra_volumes + per-container volumes
        volumes = [f"{self.config.output_dir}:/app/saved_output"]
        volumes.extend(self.config.extra_volumes)
        volumes.extend(overrides.get('volumes', []))
        
        # Convert relative paths to absolute paths (Docker requires absolute paths)
        volumes = [self._normalize_volume_path(v) for v in volumes]
        
        # Deduplicate volumes by container mount point (later entries override earlier ones)
        volumes = self._deduplicate_volumes(volumes)
        
        # Build docker_args: global + per-container overrides
        docker_args = dict(self.config.docker_args)
        docker_args.update(overrides.get('docker_args', {}))
        
        # Build container config
        config = ContainerConfig(
            id=container_id,
            name=container_name,
            ip_address=container_ip,
            subnet=f"{container_ip}/24",
            default_gateway=gateway,
            image=overrides.get('image', self.config.default_image),
            device_profile_name=self._get_device_type_for_container(container_id),
            network_profile_name=self._get_network_type_for_container(container_id),
            volumes=volumes,
            environment=overrides.get('environment', {}),
            docker_args=docker_args
        )
        
        # Apply resource constraints from resource manager
        if rm_config:
            config.cpu_period = int(rm_config.get('CpuPeriod', 0))
            config.cpu_quota = int(rm_config.get('CpuQuota', 0))
            config.cpu_shares = int(rm_config.get('CpuShares', 0))
            config.cpuset_cpus = rm_config.get('CpusetCpus')
            config.nano_cpus = int(rm_config.get('NanoCPUs', 0))
            config.memory_limit = int(rm_config.get('Memory', 0))
            config.effective_cpus = rm_config.get('EffectiveCpus', 0.0)
            
            # Merge environment variables
            env_vars = rm_config.get('Environment', {})
            if env_vars:
                config.environment.update(env_vars)
        
        # Apply per-container constraint overrides
        constraints = overrides.get('constraints', {})
        if constraints:
            if 'cpu_period' in constraints:
                config.cpu_period = constraints['cpu_period']
            if 'cpu_quota' in constraints:
                config.cpu_quota = constraints['cpu_quota']
            if 'cpuset_cpus' in constraints:
                config.cpuset_cpus = constraints['cpuset_cpus']
            if 'memory_mb' in constraints:
                config.memory_limit = constraints['memory_mb'] * 1024 * 1024
        
        # Apply link characteristics
        delay, bw, jitter, loss = self._get_link_params(container_id)
        config.link_delay_ms = delay * 2  # Store full RTT
        config.link_bandwidth_mbps = bw
        config.link_jitter_ms = jitter
        config.link_loss_percent = loss
        
        return config
    
    def _create_docker_container(self, config: ContainerConfig) -> Any:
        """Create a Docker container in the network."""
        docker_params = {
            'name': config.name,
            'ip': config.subnet,
            'dimage': config.image,
            'volumes': config.volumes,
            'privileged': True,
            'user': 'root',  # Run as root to allow network configuration
        }
        
        # Add memory limit
        if config.memory_limit > 0:
            docker_params['mem_limit'] = config.memory_limit
        
        # Add CPU constraints
        if config.cpu_period > 0:
            docker_params['cpu_period'] = config.cpu_period
        if config.cpu_quota > 0:
            docker_params['cpu_quota'] = config.cpu_quota
        if config.cpuset_cpus:
            docker_params['cpuset_cpus'] = config.cpuset_cpus
            info(f"*** {config.name}: pinned to cores {config.cpuset_cpus}\n")
        
        # Add environment variables
        if config.environment:
            docker_params['environment'] = config.environment
        
        # Add custom docker args (these can override above settings)
        if config.docker_args:
            docker_params.update(config.docker_args)
            info(f"*** {config.name}: custom docker_args applied: {list(config.docker_args.keys())}\n")
        
        info(f"*** {config.name}: creating container with image={config.image}\n")
        
        container = self.net.addDocker(**docker_params)
        
        # Disable network offloading
        try:
            container.cmd("ethtool -K eth0 gro off tx off rx off 2>/dev/null || true")
        except:
            pass
        
        info(f"*** {config.name}: effective_cpus={config.effective_cpus:.2f}, "
             f"quota={config.cpu_quota}, cpuset={config.cpuset_cpus}\n")
        
        return container
    
    def connect_containers_to_network(self) -> None:
        """Connect containers to their switches."""
        info('*** Connecting containers to switches\n')
        
        for container_id, config in self.containers.items():
            switch = self.switches[container_id]
            self.net.addLink(config.docker_container, switch, cls=TCLink)
    
    def _configure_container_routing(self) -> None:
        """Configure routing on all containers. Must be called AFTER network start.
        
        Makes 10.0.x.x (Containernet) the primary network for container-to-container traffic.
        Internet access is configured separately if enable_nat=True.
        """
        info('*** Configuring container routing\n')
        
        for container_id, config in self.containers.items():
            container = config.docker_container
            intf_name = f"{config.name}-eth0"
            gateway = config.default_gateway
            
            # Check if 'ip' command exists and what type it is
            check_ip = container.cmd("which ip 2>/dev/null").strip()
            ip_version = container.cmd("ip -V 2>&1 || ip --version 2>&1 || echo 'unknown'").strip()
            
            if not check_ip:
                # Try to install iproute2 (running as root)
                info(f"*** {config.name}: 'ip' not found, attempting to install iproute2...\n")
                container.cmd("apt-get update -qq 2>/dev/null && apt-get install -y -qq iproute2 2>/dev/null || true")
                check_ip = container.cmd("which ip 2>/dev/null").strip()
            
            if not check_ip:
                info(f"*** {config.name}: WARNING - 'ip' command not available, cannot configure routing\n")
                continue
            
            info(f"*** {config.name}: using ip at {check_ip} ({ip_version[:50]})\n")
            
            # Check if this is busybox ip (limited functionality)
            is_busybox = 'BusyBox' in ip_version or 'busybox' in check_ip.lower()
            if is_busybox:
                info(f"*** {config.name}: detected BusyBox ip - using compatible commands\n")
            
            # Get Docker's default gateway before we change routes
            docker_gateway = container.cmd("ip route | grep default | awk '{print $3}'").strip()
            docker_iface = container.cmd("ip route | grep default | awk '{print $5}'").strip()
            
            # Store Docker gateway info for later (used by _configure_internet_routing)
            config.docker_gateway = docker_gateway
            config.docker_iface = docker_iface
            
            # First, check current interface state
            intf_check = container.cmd(f"ip link show {intf_name} 2>&1")
            if 'does not exist' in intf_check or 'not found' in intf_check:
                info(f"*** {config.name}: ERROR - interface {intf_name} does not exist!\n")
                # List available interfaces for debugging
                all_intfs = container.cmd("ip link show 2>/dev/null")
                info(f"*** {config.name}: available interfaces:\n{all_intfs}\n")
                continue
            
            info(f"*** {config.name}: bringing up interface {intf_name}\n")
            
            # Bring the interface UP
            result = container.cmd(f"ip link set {intf_name} up 2>&1")
            if result.strip():
                # Check if it's a permission error or other issue
                if 'Operation not permitted' in result:
                    info(f"*** {config.name}: ERROR - cannot bring up interface (not privileged?): {result.strip()}\n")
                elif 'RTNETLINK' in result:
                    info(f"*** {config.name}: kernel error bringing up interface: {result.strip()}\n")
                else:
                    info(f"*** {config.name}: ip link set up: {result.strip()}\n")
            
            # Give the interface a moment to come up
            import time
            time.sleep(0.5)
            
            # Verify interface is up
            intf_state = container.cmd(f"ip link show {intf_name} 2>/dev/null").strip()
            if 'UP' in intf_state or 'LOWER_UP' in intf_state:
                info(f"*** {config.name}: interface {intf_name} is UP\n")
            else:
                info(f"*** {config.name}: WARNING - interface {intf_name} may not be fully up\n")
                info(f"*** {config.name}: state: {intf_state[:200]}\n")
            
            # Verify IP address is assigned
            ip_addr = container.cmd(f"ip addr show {intf_name} 2>/dev/null | grep 'inet '").strip()
            if ip_addr:
                info(f"*** {config.name}: IP assigned: {ip_addr}\n")
            else:
                info(f"*** {config.name}: WARNING - no IP address on {intf_name}\n")
            
            # Step 1: Delete Docker's default route (we want Containernet as primary)
            container.cmd("ip route del default 2>/dev/null || true")
            
            # Step 2: Add route for container-to-container traffic via Containernet
            result = container.cmd(f"ip route add 10.0.0.0/8 via {gateway} dev {intf_name} 2>&1")
            if result.strip():
                if 'File exists' in result or 'RTNETLINK answers: File exists' in result:
                    info(f"*** {config.name}: route 10.0.0.0/8 already exists\n")
                elif 'Network is unreachable' in result:
                    info(f"*** {config.name}: ERROR - gateway {gateway} unreachable (interface may be down)\n")
                else:
                    info(f"*** {config.name}: route add result: {result.strip()}\n")
            
            # Step 3: Set Containernet as default (for container traffic)
            result = container.cmd(f"ip route add default via {gateway} dev {intf_name} 2>&1")
            if result.strip():
                if 'File exists' in result or 'RTNETLINK answers: File exists' in result:
                    info(f"*** {config.name}: default route already exists\n")
                elif 'Network is unreachable' in result:
                    info(f"*** {config.name}: ERROR - cannot set default route, gateway unreachable\n")
                else:
                    info(f"*** {config.name}: default route result: {result.strip()}\n")
            
            # Show final routes for verification
            routes = container.cmd("ip route 2>/dev/null")
            info(f"*** {config.name} final routes:\n{routes}\n")
            
            # Verify connectivity to gateway
            ping_result = container.cmd(f"ping -c 1 -W 1 {gateway} 2>&1")
            if '1 received' in ping_result or '1 packets received' in ping_result:
                info(f"*** {config.name}: gateway {gateway} is reachable ✓\n")
            else:
                info(f"*** {config.name}: WARNING - cannot ping gateway {gateway}\n")
    
    def _configure_nat(self) -> None:
        """Configure NAT for internet access from containers."""
        info('*** Configuring NAT for internet access\n')
        
        # Find the host's external interface (the one with default route)
        try:
            import subprocess
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True, text=True
            )
            # Parse: "default via 192.168.1.1 dev eth0 ..."
            if result.stdout:
                parts = result.stdout.split()
                if 'dev' in parts:
                    dev_idx = parts.index('dev')
                    host_interface = parts[dev_idx + 1]
                    
                    # Enable IP forwarding
                    os.system("sysctl -w net.ipv4.ip_forward=1 >/dev/null 2>&1")
                    
                    # Set up NAT (masquerading) for container subnets
                    # First, clean up any existing rules for our subnets
                    os.system(f"iptables -t nat -D POSTROUTING -s 10.0.0.0/8 -o {host_interface} -j MASQUERADE 2>/dev/null")
                    
                    # Add NAT rule
                    os.system(f"iptables -t nat -A POSTROUTING -s 10.0.0.0/8 -o {host_interface} -j MASQUERADE")
                    
                    # Allow forwarding
                    os.system(f"iptables -D FORWARD -i {host_interface} -o s+ -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null")
                    os.system(f"iptables -D FORWARD -o {host_interface} -i s+ -j ACCEPT 2>/dev/null")
                    os.system(f"iptables -A FORWARD -i {host_interface} -o s+ -m state --state RELATED,ESTABLISHED -j ACCEPT")
                    os.system(f"iptables -A FORWARD -o {host_interface} -i s+ -j ACCEPT")
                    
                    info(f"*** NAT configured on interface {host_interface}\n")
                else:
                    info("*** Warning: Could not determine host interface for NAT\n")
            else:
                info("*** Warning: No default route found, NAT not configured\n")
        except Exception as e:
            info(f"*** Warning: Failed to configure NAT: {e}\n")
    
    def _configure_internet_routing(self) -> None:
        """Add routes for internet access via Docker's network."""
        info('*** Configuring internet routing\n')
        
        for container_id, config in self.containers.items():
            container = config.docker_container
            
            # Use stored Docker gateway (captured before we changed routes)
            docker_gateway = getattr(config, 'docker_gateway', None)
            docker_iface = getattr(config, 'docker_iface', None)
            
            if not docker_gateway or not docker_iface:
                info(f"*** {config.name}: No Docker gateway info, skipping internet route\n")
                continue
            
            # Add route for internet traffic via Docker gateway (with higher metric so 10.x.x.x takes priority)
            container.cmd(f"ip route add default via {docker_gateway} dev {docker_iface} metric 100 2>/dev/null || true")
            
            info(f"*** {config.name}: internet via {docker_gateway} ({docker_iface})\n")
    
    def start(self) -> None:
        """Start the network."""
        if self._started:
            return
        
        info('*** Starting network\n')
        self.net.start()
        self._started = True
        
        # Disable network offloading on host
        script_dir = os.path.dirname(os.path.abspath(__file__))
        disable_script = os.path.join(script_dir, "resources", "disable_offload.sh")
        os.system(f"bash {disable_script} >/dev/null 2>&1")
        
        # Configure container routing AFTER network start
        self._configure_container_routing()
        
        # Configure NAT and internet access if enabled
        if self.config.enable_nat:
            self._configure_nat()
            self._configure_internet_routing()
        
        # Brief pause for network stabilization
        import time
        time.sleep(max(2, self.config.default_delay_ms / 2))
    
    def stop(self) -> None:
        """Stop the network and clean up."""
        info('*** Stopping network\n')
        self.net.stop()
        self._started = False
        clean_containernet.clean_session()
    
    def debug_network(self) -> dict:
        """Debug network configuration. Returns diagnostic info."""
        debug_info = {
            'containers': {},
            'routers': {},
            'connectivity': []
        }
        
        info('*** Network Debug ***\n')
        
        # Container info
        for cid, config in self.containers.items():
            container = config.docker_container
            
            # Get interfaces
            interfaces = container.cmd('ip -o addr show')
            routes = container.cmd('ip route')
            
            debug_info['containers'][config.name] = {
                'ip': config.ip_address,
                'gateway': config.default_gateway,
                'interfaces': interfaces.strip(),
                'routes': routes.strip()
            }
            
            info(f'\n=== Container {config.name} (ID: {cid}) ===\n')
            info(f'Expected IP: {config.ip_address}\n')
            info(f'Gateway: {config.default_gateway}\n')
            info(f'Interfaces:\n{interfaces}\n')
            info(f'Routes:\n{routes}\n')
        
        # Router info
        for router in self.routers:
            interfaces = router.router.cmd('ip -o addr show')
            routes = router.router.cmd('ip route')
            
            debug_info['routers'][router.name] = {
                'interfaces': interfaces.strip(),
                'routes': routes.strip()
            }
            
            info(f'\n=== Router {router.name} ===\n')
            info(f'Interfaces:\n{interfaces}\n')
            info(f'Routes:\n{routes}\n')
        
        # Connectivity tests
        info('\n=== Connectivity Tests ===\n')
        for cid, config in self.containers.items():
            container = config.docker_container
            
            # Ping gateway
            gateway_result = container.cmd(f'ping -c 1 -W 2 {config.default_gateway}')
            gateway_ok = '1 received' in gateway_result
            info(f'{config.name} -> gateway ({config.default_gateway}): {"OK" if gateway_ok else "FAIL"}\n')
            
            # Ping other containers
            for other_cid, other_config in self.containers.items():
                if other_cid != cid:
                    ping_result = container.cmd(f'ping -c 1 -W 2 {other_config.ip_address}')
                    ping_ok = '1 received' in ping_result
                    debug_info['connectivity'].append({
                        'from': config.name,
                        'to': other_config.name,
                        'target_ip': other_config.ip_address,
                        'success': ping_ok
                    })
                    info(f'{config.name} -> {other_config.name} ({other_config.ip_address}): {"OK" if ping_ok else "FAIL"}\n')
        
        # Internet connectivity test
        info('\n=== Internet Connectivity Tests ===\n')
        debug_info['internet'] = []
        test_targets = [
            ('8.8.8.8', 'Google DNS'),
            ('1.1.1.1', 'Cloudflare DNS'),
        ]
        
        for cid, config in self.containers.items():
            container = config.docker_container
            for target_ip, target_name in test_targets:
                ping_result = container.cmd(f'ping -c 1 -W 3 {target_ip}')
                ping_ok = '1 received' in ping_result
                debug_info['internet'].append({
                    'from': config.name,
                    'target': target_name,
                    'target_ip': target_ip,
                    'success': ping_ok
                })
                info(f'{config.name} -> {target_name} ({target_ip}): {"OK" if ping_ok else "FAIL"}\n')
                
                # Only test one target per container if first succeeds
                if ping_ok:
                    break
        
        return debug_info
    
    def get_container(self, container_id: int) -> Optional[ContainerConfig]:
        """Get container configuration by ID."""
        return self.containers.get(container_id)
    
    def get_container_by_name(self, name: str) -> Optional[ContainerConfig]:
        """Get container configuration by name."""
        for config in self.containers.values():
            if config.name == name:
                return config
        return None
    
    def get_all_containers(self) -> Dict[int, ContainerConfig]:
        """Get all container configurations."""
        return self.containers
    
    def run_cli(self) -> None:
        """Start interactive CLI."""
        CLI(self.net)
    
    def save_topology(self, filepath: Optional[str] = None) -> dict:
        """Save network topology to JSON file."""
        filepath = filepath or os.path.join(self.config.output_dir, 'network_topology.json')
        
        topology = {
            'timestamp': datetime.datetime.now().isoformat(),
            'num_containers': self.config.num_containers,
            'host_score': self.config.host_single_core_score,
            'routers': [],
            'containers': []
        }
        
        for router in self.routers:
            topology['routers'].append({
                'id': router.id,
                'name': router.name,
                'network': str(router.network_ip),
                'main_ip': router.main_ip,
                'bindings': [
                    {'dest': str(b[0]), 'via': b[1], 'interface': b[2]}
                    for b in router.routing_bindings
                ]
            })
        
        for container_id, config in self.containers.items():
            topology['containers'].append({
                'id': config.id,
                'name': config.name,
                'ip_address': config.ip_address,
                'subnet': config.subnet,
                'gateway': config.default_gateway,
                'image': config.image,
                'device_profile': config.device_profile_name,
                'network_profile': config.network_profile_name,
                'link': {
                    'delay_ms': config.link_delay_ms,
                    'bandwidth_mbps': config.link_bandwidth_mbps,
                    'jitter_ms': config.link_jitter_ms,
                    'loss_percent': config.link_loss_percent
                },
                'resources': {
                    'cpu_period': config.cpu_period,
                    'cpu_quota': config.cpu_quota,
                    'cpu_shares': config.cpu_shares,
                    'cpuset_cpus': config.cpuset_cpus,
                    'nano_cpus': config.nano_cpus,
                    'memory_limit': config.memory_limit,
                    'effective_cpus': config.effective_cpus
                }
            })
        
        # Add resource manager summary if available
        if self.resource_manager:
            topology['cpu_allocation_summary'] = self.resource_manager.get_allocation_summary()
        
        with open(filepath, 'w') as f:
            json.dump(topology, f, indent=2, default=str)
        
        info(f'*** Topology saved to {filepath}\n')
        return topology


def create_network_config_from_dict(config_dict: dict, output_dir: str) -> NetworkConfig:
    """Create NetworkConfig from a dictionary (e.g., parsed YAML)."""
    # Support both nested 'containernet' section and flat config
    containernet_config = config_dict.get('containernet', config_dict)
    
    # Calculate num_containers: either explicit or clients + 1 (for server)
    num_containers = containernet_config.get('num_containers')
    if num_containers is None:
        clients = containernet_config.get('clients', 1)
        num_containers = clients + 1
    
    network_config = NetworkConfig(
        num_containers=num_containers,
        default_image=containernet_config.get('image_name', 'ubuntu:latest'),
        output_dir=output_dir,
        # Default link parameters (0 = no limitation)
        default_delay_ms=containernet_config.get('default_delay_ms', 0.0),
        default_bandwidth_mbps=containernet_config.get('default_bandwidth_mbps', 0),
        default_jitter_ms=containernet_config.get('default_jitter_ms', 0.0),
        # Resource management defaults
        host_single_core_score=containernet_config.get('host_single_core_score', 1079),
        device_variance=containernet_config.get('device_variance', 0.2),
        cpu_spread_threshold=containernet_config.get('cpu_spread_threshold', 0.8),
        default_cores_per_container=containernet_config.get('default_cores_per_container', 1),
        default_ram_gib=containernet_config.get('default_ram_gib', 2.0),
        allow_overscaling=containernet_config.get('allow_overscaling', True),
        # Device/network profiles (None = no constraints/limitations)
        device_type=containernet_config.get('device_type'),
        network_type=containernet_config.get('network_type'),
        # Features
        enable_tcpdump=containernet_config.get('enable_tcpdump', False),
        enable_nat=containernet_config.get('enable_nat', False),  # Default: no internet, only container-to-container
        # Extra volumes and docker args
        extra_volumes=containernet_config.get('volumes', []),
        docker_args=containernet_config.get('docker_args', {})
    )
    
    # Process per-container overrides from 'nodes' list
    container_overrides = {}
    nodes = containernet_config.get('nodes', [])
    
    for node in nodes:
        node_id = node.get('id')
        if node_id is not None:
            container_overrides[node_id] = node
    
    network_config.container_overrides = container_overrides
    
    return network_config
