"""
Containernet Experiment Framework

A modular framework for running containerized network experiments with
clean separation between network configuration and application logic.
"""

from .containernet_manager import (
    ContainernetManager,
    NetworkConfig,
    ContainerConfig,
    create_network_config_from_dict
)

from .application_runner import (
    ApplicationRunner,
    ApplicationConfig,
    RoleConfig,
    CommandTemplate,
    create_application_config_from_dict
)

# Import from resources submodule
from .resources.performance import (
    device_profile, 
    network_profile,
    ContainerResourceManager,
    CPUAllocator,
    load_profile_data,
    perturb_device
)

__version__ = "1.0.0"
__all__ = [
    # Network management
    "ContainernetManager",
    "NetworkConfig", 
    "ContainerConfig",
    "create_network_config_from_dict",
    
    # Application execution
    "ApplicationRunner",
    "ApplicationConfig",
    "RoleConfig",
    "CommandTemplate",
    "create_application_config_from_dict",
    
    # Profiles (from resources.performance)
    "device_profile",
    "network_profile",
    "ContainerResourceManager",
    "CPUAllocator",
    "load_profile_data",
    "perturb_device",
]
