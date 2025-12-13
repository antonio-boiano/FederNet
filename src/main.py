#!/usr/bin/env python3
"""
Main entry point for running containerized experiments.
Cleanly separates network configuration from application logic.

Usage:
    python3 main.py --config config.yaml [--interactive] [--network-only]
"""

import argparse
import datetime
import os
import sys
import yaml
import json
from mininet.log import setLogLevel, info, error

# Import our modules (relative imports for package structure)
from .containernet_manager import (
    ContainernetManager, 
    NetworkConfig,
    create_network_config_from_dict
)
from .application_runner import (
    ApplicationRunner,
    ApplicationConfig,
    create_application_config_from_dict
)

# Import cleanup utility
from .resources import clean_containernet


def setup_output_directory(config_path: str, config_dict: dict) -> str:
    """Create and return the output directory path."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Get experiment name
    experiment_name = config_dict.get('application', {}).get('name',
                      config_dict.get('experiment_name', 'experiment'))
    
    # Build output path in current working directory
    pwd = os.getcwd()
    output_root = os.path.join(pwd, "output")
    
    # Get info for output directory name
    containernet_config = config_dict.get('containernet', config_dict)
    clients = containernet_config.get('clients', containernet_config.get('num_containers', 1) - 1)
    
    image_name = containernet_config.get('image_name', 'unknown')
    image_name_str = image_name.split("/")[-1] if "/" in image_name else image_name
    image_name_str = image_name_str.split(":")[0] if ":" in image_name_str else image_name_str
    
    device_type = containernet_config.get('device_type')
    device_type_str = device_type[0] if isinstance(device_type, list) and device_type else str(device_type or 'none')
    
    network_type = containernet_config.get('network_type')
    network_type_str = network_type[0] if isinstance(network_type, list) and network_type else str(network_type or 'none')
    
    output_dir_name = "{}_{}_C{}_{}_{}_{}".format(
        timestamp, image_name_str, clients, 
        network_type_str[:20], device_type_str[:20], experiment_name
    )
    
    output_dir = os.path.join(output_root, output_dir_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set permissions to allow containers to write (they may run as different users)
    os.chmod(output_dir, 0o777)
    os.chmod(output_root, 0o777)
    
    return output_dir


def load_config(config_path: str) -> dict:
    """Load and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Run containerized network experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run a full experiment
  python3 main.py --config experiments/fl_experiment.yaml
  
  # Interactive mode (drops to CLI after setup)
  python3 main.py --config experiments/fl_experiment.yaml --interactive
  
  # Network only (no application execution)
  python3 main.py --config experiments/fl_experiment.yaml --network-only
        """
    )
    parser.add_argument('--config', required=True, help='Path to YAML configuration file')
    parser.add_argument('--interactive', action='store_true', help='Drop to CLI after network setup')
    parser.add_argument('--network-only', action='store_true', help='Set up network but do not run application')
    parser.add_argument('--debug-network', action='store_true', help='Run network diagnostics after setup')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Show commands being executed')
    parser.add_argument('--show-output', '-o', action='store_true',
                        help='Show command output in terminal')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='Enable debug mode (very verbose, shows everything)')
    parser.add_argument('--log-level', default='info', choices=['debug', 'info', 'warning', 'error'],
                        help='Logging level')
    
    args = parser.parse_args()
    
    # Set log level
    setLogLevel(args.log_level)
    
    # Clean up any existing session
    info('*** Cleaning up previous session\n')
    clean_containernet.clean_session()
    
    # Load configuration
    info(f'*** Loading configuration from {args.config}\n')
    config = load_config(args.config)
    
    # Setup output directory
    output_dir = setup_output_directory(args.config, config)
    info(f'*** Output directory: {output_dir}\n')
    
    # Save original config for reproducibility
    config_backup_path = os.path.join(output_dir, 'config_original.yaml')
    with open(config_backup_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Determine number of containers
    containernet_section = config.get('containernet', config)
    num_containers = containernet_section.get('num_containers')
    if num_containers is None:
        clients = containernet_section.get('clients', 1)
        num_containers = clients + 1
    
    # Get experiment name for logging
    experiment_name = config.get('application', {}).get('name',
                      config.get('experiment_name', 'experiment'))
    
    info(f'*** Experiment: {experiment_name}\n')
    info(f'*** Number of containers: {num_containers}\n')
    
    # Create network configuration
    network_config = create_network_config_from_dict(config, output_dir)
    
    # Extract role-based overrides from application config and merge into network config
    # This allows roles to specify their own Docker images, volumes, and docker args
    app_section = config.get('application', {})
    roles_config = app_section.get('roles', {})
    
    for role_name, role_data in roles_config.items():
        # Get role overrides
        role_image = role_data.get('image')
        role_volumes = role_data.get('volumes', [])
        role_docker_args = role_data.get('docker_args', {})
        role_working_dir = role_data.get('working_dir')
        role_environment = role_data.get('environment', {})
        
        # Check if there's anything to override
        has_overrides = role_image or role_volumes or role_docker_args or role_working_dir or role_environment
        
        if has_overrides:
            container_ids = role_data.get('container_ids', [])
            
            # Handle special container_ids values
            if container_ids == "all_except_server":
                server_ids = roles_config.get('server', {}).get('container_ids', [0])
                container_ids = [i for i in range(num_containers) if i not in server_ids]
            elif container_ids == "all":
                container_ids = list(range(num_containers))
            
            # Merge overrides into container config
            for cid in container_ids:
                if cid not in network_config.container_overrides:
                    network_config.container_overrides[cid] = {}
                
                override = network_config.container_overrides[cid]
                
                if role_image:
                    override['image'] = role_image
                    info(f'*** Container {cid} ({role_name}): image={role_image}\n')
                
                if role_volumes:
                    existing_volumes = override.get('volumes', [])
                    override['volumes'] = existing_volumes + role_volumes
                    info(f'*** Container {cid} ({role_name}): volumes={role_volumes}\n')
                
                if role_docker_args or role_working_dir:
                    existing_args = override.get('docker_args', {})
                    existing_args.update(role_docker_args)
                    if role_working_dir:
                        existing_args['working_dir'] = role_working_dir
                    override['docker_args'] = existing_args
                    info(f'*** Container {cid} ({role_name}): docker_args={list(existing_args.keys())}\n')
                
                if role_environment:
                    existing_env = override.get('environment', {})
                    existing_env.update(role_environment)
                    override['environment'] = existing_env
    
    # Create network manager
    network_manager = ContainernetManager(network_config)
    
    try:
        # Set up network topology
        network_manager.setup_network()
        
        # Create containers
        network_manager.create_containers()
        
        # Connect containers to network
        network_manager.connect_containers_to_network()
        
        # Start network
        network_manager.start()
        
        # Save topology
        network_manager.save_topology()
        
        # Run network diagnostics if requested
        if args.debug_network:
            info('*** Running network diagnostics\n')
            debug_info = network_manager.debug_network()
            
            # Save debug info
            debug_path = os.path.join(output_dir, 'network_debug.json')
            with open(debug_path, 'w') as f:
                json.dump(debug_info, f, indent=2)
            info(f'*** Debug info saved to {debug_path}\n')
        
        if args.interactive:
            info('*** Starting interactive CLI\n')
            network_manager.run_cli()
        elif args.network_only or args.debug_network:
            if not args.debug_network:
                info('*** Network-only mode: network is ready\n')
            info('*** Press Ctrl+C to stop\n')
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
        else:
            # Create and run application
            app_config = create_application_config_from_dict(
                config, output_dir, num_containers, 
                verbose=args.verbose or args.debug,
                show_output=args.show_output or args.debug,
                debug=args.debug
            )
            app_runner = ApplicationRunner(network_manager, app_config)
            
            # Save application config
            app_runner.save_config()
            
            # Run the application
            app_runner.run()
        
    except Exception as e:
        error(f'*** Exception: {e}\n')
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        info('*** Stopping network\n')
        network_manager.stop()
    
    info('*** Done\n')


if __name__ == '__main__':
    main()
