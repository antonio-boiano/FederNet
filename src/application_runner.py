"""
ApplicationRunner: Handles execution of commands in containers.
Completely decoupled from Containernet setup logic.
Supports flexible role-based command configuration.
"""

import os
import time
import datetime
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from string import Template
from mininet.log import info, error

from .containernet_manager import ContainernetManager, ContainerConfig


@dataclass
class CommandTemplate:
    """Template for a command with variable substitution."""
    template: str
    description: str = ""
    
    def render(self, variables: Dict[str, Any]) -> str:
        """Render the command template with given variables."""
        # Support both {var} and $var style templates
        try:
            # First try Python format style
            return self.template.format(**variables)
        except (KeyError, ValueError):
            # Fall back to Template style
            return Template(self.template).safe_substitute(variables)


@dataclass
class RoleConfig:
    """Configuration for a container role."""
    name: str
    container_ids: List[int]
    command: CommandTemplate
    startup_delay: float = 0.0  # Delay before starting this role
    wait_for_completion: bool = True
    pre_commands: List[str] = field(default_factory=list)  # Commands to run before main command
    post_commands: List[str] = field(default_factory=list)  # Commands to run after main command
    environment: Dict[str, str] = field(default_factory=dict)
    image: Optional[str] = None  # Override Docker image for containers in this role
    volumes: List[str] = field(default_factory=list)  # Additional volume mounts for this role
    docker_args: Dict[str, Any] = field(default_factory=dict)  # Custom Docker args for this role
    working_dir: Optional[str] = None  # Working directory inside container


@dataclass
class ApplicationConfig:
    """Configuration for an application to run in containers."""
    name: str
    output_dir: str
    roles: Dict[str, RoleConfig]
    
    # Global variables available to all command templates
    global_variables: Dict[str, Any] = field(default_factory=dict)
    
    # Execution order for roles
    role_order: List[str] = field(default_factory=list)
    
    # Features
    enable_tcpdump: bool = False
    tcpdump_interfaces: str = "any"
    
    # Logging options
    log_commands: bool = True      # Log commands to file
    verbose: bool = False          # Print commands to terminal
    show_output: bool = False      # Print command output to terminal
    debug: bool = False            # Enable debug mode (very verbose)


class CommandLogger:
    """Logs all executed commands with timestamps."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.commands: List[Dict] = []
    
    def log(self, container_name: str, command: str, role: str = "") -> None:
        """Log a command execution."""
        entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'container': container_name,
            'role': role,
            'command': command
        }
        self.commands.append(entry)
        info(f"*** [{container_name}] {command[:80]}{'...' if len(command) > 80 else ''}\n")
    
    def save(self, filename: str = "commands_executed.json") -> None:
        """Save command log to file."""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(self.commands, f, indent=2)
        info(f"*** Command log saved to {filepath}\n")


class ApplicationRunner:
    """
    Executes application commands in containers.
    Supports role-based execution with flexible command templates.
    """
    
    def __init__(self, network_manager: ContainernetManager, config: ApplicationConfig):
        self.network = network_manager
        self.config = config
        self.logger = CommandLogger(config.output_dir)
        self._running_containers: Dict[int, Any] = {}
    
    def _build_variables(self, container: ContainerConfig, role: RoleConfig) -> Dict[str, Any]:
        """Build variable dictionary for command template rendering."""
        # Start with global variables
        variables = dict(self.config.global_variables)
        
        # Add container-specific variables
        variables.update({
            'container_id': container.id,
            'container_name': container.name,
            'container_ip': container.ip_address,
            'container_subnet': container.subnet,
            'container_gateway': container.default_gateway,
            'device_profile': container.device_profile_name or 'none',
            'network_profile': container.network_profile_name or 'none',
            'output_dir': '/app/saved_output',
            'timestamp': datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
            'index': container.id,  # Alias for container_id
            'my_ip': container.ip_address,  # Alias for container_ip
        })
        
        # Add role-specific variables
        variables['role'] = role.name
        
        # Add IPs for ALL containers by index: {ip_0}, {ip_1}, {ip_2}, ...
        # This allows referencing any container's IP directly
        for cid, cfg in self.network.containers.items():
            variables[f'ip_{cid}'] = cfg.ip_address
            variables[f'c{cid}_ip'] = cfg.ip_address  # Alternative syntax
        
        # Add IPs for all defined roles: {server_ip}, {broker_ip}, {coordinator_ip}, ...
        # These reference the FIRST container in each role
        for role_name, role_cfg in self.config.roles.items():
            if role_cfg.container_ids:
                first_container_id = role_cfg.container_ids[0]
                role_container = self.network.get_container(first_container_id)
                if role_container:
                    variables[f'{role_name}_ip'] = role_container.ip_address
        
        # Legacy aliases for backward compatibility
        if 'server' in self.config.roles:
            server_role = self.config.roles['server']
            if server_role.container_ids:
                server_id = server_role.container_ids[0]
                server_container = self.network.get_container(server_id)
                if server_container:
                    variables['server_address'] = server_container.ip_address
                    variables['ip'] = server_container.ip_address  # Legacy alias
        
        return variables
    
    def _get_log_filename(self, container: ContainerConfig, role: RoleConfig) -> str:
        """Generate log filename for a container."""
        dev_str = (container.device_profile_name or 'default').replace(" ", "_")[:15]
        net_str = (container.network_profile_name or 'default').replace(" ", "_")[:15]
        profile_str = f"dev_{dev_str}_net_{net_str}"
        return f"{self.config.name}_{role.name}_{container.name}_{profile_str}.log"
    
    def run_command(self, container: ContainerConfig, command: str, 
                    role: Optional[RoleConfig] = None, async_exec: bool = False) -> Optional[str]:
        """
        Run a command in a container using /bin/bash.
        
        Args:
            container: Container to run command in
            command: Command to execute
            role: Role configuration
            async_exec: If True, run command asynchronously
        """
        docker = container.docker_container
        role_name = role.name if role else "unknown"
        
        if self.config.log_commands:
            self.logger.log(container.name, command, role_name)
        
        # ANSI colors for terminal output
        CYAN = '\033[96m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
        DIM = '\033[2m'
        
        # Verbose output - show command being executed
        if self.config.verbose or self.config.debug:
            # Truncate command for display if too long
            display_cmd = command[:300] + "..." if len(command) > 300 else command
            display_cmd = display_cmd.replace('\n', ' ').strip()
            
            print(f"{CYAN}[{container.name}:{role_name}]{RESET} {BOLD}CMD:{RESET} {display_cmd}")
        
        if async_exec:
            docker.sendCmd(command)
            if self.config.verbose or self.config.debug:
                print(f"{YELLOW}  → Running async (output will be in log file){RESET}")
            return None
        else:
            result = docker.cmd(command)
            
            # Show output if requested
            if self.config.show_output or self.config.debug:
                if result and result.strip():
                    # Check for common error indicators
                    result_lower = result.lower()
                    has_error = any(err in result_lower for err in 
                                   ['error', 'failed', 'exception', 'traceback', 'no such file', 
                                    'command not found', 'permission denied', 'cannot', 'fatal'])
                    
                    if has_error:
                        print(f"{RED}  ✗ OUTPUT (possible error):{RESET}")
                        for line in result.strip().split('\n'):
                            print(f"{RED}    {line}{RESET}")
                    else:
                        lines = result.strip().split('\n')
                        if self.config.debug:
                            # In debug mode, show all output
                            print(f"{GREEN}  ✓ OUTPUT ({len(lines)} lines):{RESET}")
                            for line in lines:
                                print(f"{DIM}    {line}{RESET}")
                        elif len(lines) > 10:
                            # Truncate long output
                            print(f"{GREEN}  ✓ OUTPUT ({len(lines)} lines, showing first/last 5):{RESET}")
                            for line in lines[:5]:
                                print(f"    {line}")
                            print(f"    {DIM}... ({len(lines) - 10} lines hidden) ...{RESET}")
                            for line in lines[-5:]:
                                print(f"    {line}")
                        else:
                            print(f"{GREEN}  ✓ OUTPUT:{RESET}")
                            for line in lines:
                                print(f"    {line}")
                else:
                    if self.config.debug:
                        print(f"{GREEN}  ✓ (no output){RESET}")
            elif self.config.verbose:
                # In verbose mode (not show_output), just indicate success/failure
                if result:
                    result_lower = result.lower()
                    has_error = any(err in result_lower for err in 
                                   ['error', 'failed', 'exception', 'traceback', 'no such file', 
                                    'command not found', 'permission denied', 'cannot', 'fatal'])
                    if has_error:
                        print(f"{RED}  ✗ Command may have failed - use --show-output to see details{RESET}")
                        # Show first few lines of error
                        for line in result.strip().split('\n')[:3]:
                            print(f"{RED}    {line}{RESET}")
                    else:
                        print(f"{GREEN}  ✓ OK{RESET}")
                else:
                    print(f"{GREEN}  ✓ OK{RESET}")
            
            return result
    
    def setup_tcpdump(self, container: ContainerConfig, role: RoleConfig) -> None:
        """Set up tcpdump on a container if enabled."""
        if not self.config.enable_tcpdump:
            return
        
        # Disable network offloading for accurate capture
        offload_cmd = "ethtool -K eth0 gro off tx off rx off lro off gso off tso off sg off rxvlan off txvlan off rxhash off ufo off 2>/dev/null || true"
        self.run_command(container, offload_cmd, role)
        
        intf_offload_cmd = f"ethtool -K {container.name}-eth0 gro off tx off rx off lro off gso off tso off sg off rxvlan off txvlan off rxhash off ufo off 2>/dev/null || true"
        self.run_command(container, intf_offload_cmd, role)
        
        # Start tcpdump
        pcap_file = f"/app/saved_output/{self.config.name}_{container.name}.pcap"
        tcpdump_cmd = f"tcpdump -i {self.config.tcpdump_interfaces} -w {pcap_file} &"
        self.run_command(container, tcpdump_cmd, role)
    
    def run_role(self, role_name: str) -> None:
        """Execute commands for all containers in a role."""
        if role_name not in self.config.roles:
            error(f"Role '{role_name}' not found in configuration\n")
            return
        
        role = self.config.roles[role_name]
        info(f"*** Starting role: {role_name} ({len(role.container_ids)} containers)\n")
        
        # Verbose: show role details
        if self.config.verbose:
            MAGENTA = '\033[95m'
            BOLD = '\033[1m'
            RESET = '\033[0m'
            print(f"\n{MAGENTA}{'='*60}{RESET}")
            print(f"{MAGENTA}{BOLD}ROLE: {role_name}{RESET}")
            print(f"{MAGENTA}Containers: {role.container_ids}{RESET}")
            print(f"{MAGENTA}Wait for completion: {role.wait_for_completion}{RESET}")
            print(f"{MAGENTA}Pre-commands: {len(role.pre_commands)}{RESET}")
            print(f"{MAGENTA}{'='*60}{RESET}\n")
        
        # Apply startup delay
        if role.startup_delay > 0:
            info(f"*** Waiting {role.startup_delay}s before starting {role_name}\n")
            time.sleep(role.startup_delay)
        
        for container_id in role.container_ids:
            container = self.network.get_container(container_id)
            if not container:
                error(f"Container {container_id} not found\n")
                continue
            
            # Verbose: show container info
            if self.config.verbose:
                CYAN = '\033[96m'
                RESET = '\033[0m'
                print(f"\n{CYAN}--- Container {container.name} (ID: {container_id}, IP: {container.ip_address}) ---{RESET}")
            
            # Build variables for this container
            variables = self._build_variables(container, role)
            
            # Set up tcpdump if enabled
            self.setup_tcpdump(container, role)
            
            # Ensure output directory is writable inside container
            self.run_command(container, "chmod 777 /app/saved_output 2>/dev/null || true", role)
            
            # Run pre-commands
            for pre_cmd in role.pre_commands:
                cmd = CommandTemplate(pre_cmd).render(variables)
                self.run_command(container, cmd, role)
            
            # Build main command with robust output capture
            # Wrap in subshell to ensure ALL output is captured, including early failures
            main_cmd = role.command.render(variables)
            log_file = self._get_log_filename(container, role)
            log_path = f"/app/saved_output/{log_file}"
            
            # The subshell ensures:
            # 1. Log file is created with timestamp even if command fails immediately
            # 2. All stdout/stderr is captured (including bash errors)
            # 3. Exit code is logged at the end
            full_cmd = f"( echo '=== Started: '$(date)' ===' ; {main_cmd} ; EXIT_CODE=$? ; echo '=== Finished: '$(date)' - Exit code: '$EXIT_CODE' ===' ; exit $EXIT_CODE ) > {log_path} 2>&1"
            
            # Execute command
            if role.wait_for_completion:
                self.run_command(container, full_cmd, role, async_exec=True)
                self._running_containers[container_id] = container
            else:
                # Background execution
                self.run_command(container, f"{full_cmd} &", role)
    
    def run_all_roles(self) -> None:
        """Execute all roles in the configured order."""
        order = self.config.role_order or list(self.config.roles.keys())
        
        for role_name in order:
            self.run_role(role_name)
    
    def wait_for_completion(self) -> None:
        """Wait for all async commands to complete."""
        info("*** Waiting for all containers to complete...\n")
        
        for container_id, container in self._running_containers.items():
            info(f"*** Waiting for {container.name}...\n")
            container.docker_container.waitOutput()
        
        self._running_containers.clear()
        info("*** All containers completed\n")
    
    def run(self) -> None:
        """Run the complete application."""
        info(f"*** Starting application: {self.config.name}\n")
        
        # Execute all roles
        self.run_all_roles()
        
        # Wait for completion
        self.wait_for_completion()
        
        # Save logs
        self.logger.save()
        
        info(f"*** Application {self.config.name} completed\n")
    
    def save_config(self, filename: str = "application_config.json") -> None:
        """Save the application configuration for reproducibility."""
        filepath = os.path.join(self.config.output_dir, filename)
        
        config_dict = {
            'name': self.config.name,
            'global_variables': self.config.global_variables,
            'role_order': self.config.role_order,
            'roles': {}
        }
        
        for role_name, role in self.config.roles.items():
            config_dict['roles'][role_name] = {
                'container_ids': role.container_ids,
                'command_template': role.command.template,
                'startup_delay': role.startup_delay,
                'wait_for_completion': role.wait_for_completion,
                'pre_commands': role.pre_commands,
                'environment': role.environment
            }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        info(f"*** Application config saved to {filepath}\n")


def create_application_config_from_dict(
    config_dict: dict, 
    output_dir: str,
    num_containers: int,
    verbose: bool = False,
    show_output: bool = False,
    debug: bool = False
) -> ApplicationConfig:
    """
    Create ApplicationConfig from a dictionary (e.g., parsed YAML).
    
    Supports both new explicit role format and legacy FL format.
    """
    app_config = config_dict.get('application', {})
    
    # Get experiment name
    name = app_config.get('name', config_dict.get('experiment_name', 'experiment'))
    
    # Parse global variables
    global_vars = dict(app_config.get('variables', {}))
    
    # Also include legacy top-level variables for backward compatibility
    legacy_vars = ['protocol', 'port', 'fl_method', 'alpha', 'rounds', 'max_time', 
                   'epochs', 'min_clients', 'num_client', 'client_selection',
                   'parts_dataset', 'asofed_beta']
    for var in legacy_vars:
        if var in config_dict and var not in global_vars:
            global_vars[var] = config_dict[var]
    
    # Parse server_config if present
    server_config = config_dict.get('server_config', {})
    if server_config:
        global_vars.setdefault('min_clients', server_config.get('min_client_to_start'))
        global_vars.setdefault('num_client', server_config.get('client_round'))
    
    # Parse client_config if present
    client_config = config_dict.get('client_config', {})
    if client_config:
        global_vars.setdefault('epochs', client_config.get('epochs'))
    
    # Clean up None values
    global_vars = {k: v for k, v in global_vars.items() if v is not None}
    
    # Parse roles
    roles = {}
    roles_config = app_config.get('roles', {})
    
    if roles_config:
        # New explicit role configuration
        for role_name, role_data in roles_config.items():
            container_ids = role_data.get('container_ids', [])
            
            # Handle special container_ids values
            if container_ids == "all_except_server":
                server_ids = roles_config.get('server', {}).get('container_ids', [0])
                container_ids = [i for i in range(num_containers) if i not in server_ids]
            elif container_ids == "all":
                container_ids = list(range(num_containers))
            
            # Build docker_args, include working_dir if specified
            docker_args = dict(role_data.get('docker_args', {}))
            if role_data.get('working_dir'):
                docker_args['working_dir'] = role_data.get('working_dir')
            
            roles[role_name] = RoleConfig(
                name=role_name,
                container_ids=container_ids,
                command=CommandTemplate(
                    template=role_data.get('command', ''),
                    description=role_data.get('description', '')
                ),
                startup_delay=role_data.get('startup_delay', 0.0),
                wait_for_completion=role_data.get('wait_for_completion', True),
                pre_commands=role_data.get('pre_commands', []),
                post_commands=role_data.get('post_commands', []),
                environment=role_data.get('environment', {}),
                image=role_data.get('image'),
                volumes=role_data.get('volumes', []),
                docker_args=docker_args,
                working_dir=role_data.get('working_dir')
            )
    else:
        # Legacy FL configuration - create default server/client roles
        protocol = global_vars.get('protocol', 'grpc')
        port = _get_default_port(protocol)
        global_vars['port'] = port
        
        # Build argument string from global vars
        arg_keys = ['protocol', 'rounds', 'max_time', 'fl_method', 'alpha', 
                    'min_clients', 'num_client', 'epochs', 'client_selection',
                    'parts_dataset', 'asofed_beta']
        args = ' '.join(f'--{k} {global_vars[k]}' for k in arg_keys if k in global_vars and global_vars[k] is not None)
        global_vars['extra_args'] = args
        
        # Default server role
        roles['server'] = RoleConfig(
            name='server',
            container_ids=[0],
            command=CommandTemplate(
                template="python3 -u run.py --protocol {protocol} --mode Server --port {port} --ip {container_ip} --index {container_id} " + args,
                description="FL Parameter Server"
            ),
            startup_delay=0.0,
            wait_for_completion=True
        )
        
        # Default client role
        roles['client'] = RoleConfig(
            name='client',
            container_ids=list(range(1, num_containers)),
            command=CommandTemplate(
                template="python3 -u run.py --protocol {protocol} --mode Client --my_ip {container_ip} --port {port} --ip {server_ip} --index {container_id} " + args,
                description="FL Client"
            ),
            startup_delay=20.0,  # Wait for server to start
            wait_for_completion=True
        )
    
    # Determine role order
    role_order = app_config.get('role_order', ['server', 'client'])
    
    return ApplicationConfig(
        name=name,
        output_dir=output_dir,
        roles=roles,
        global_variables=global_vars,
        role_order=role_order,
        enable_tcpdump=config_dict.get('enable_tcpdump', False),
        verbose=verbose,
        show_output=show_output,
        debug=debug
    )


def _get_default_port(protocol: str) -> int:
    """Get default port for a protocol."""
    ports = {
        'tcp': 80,
        'grpc': 50051,
        'rest': 80,
        'coap': 5683,
        'mqtt': 1883,
        'websocket': 8080,
        'amqp': 5672
    }
    return ports.get(protocol.lower(), 50051)
