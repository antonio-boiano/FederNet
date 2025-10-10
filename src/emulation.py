from mininet.net import Containernet
from mininet.node import Controller
from mininet.node import Node
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import debug, info, error
from mininet import *
from mininet.node import OVSSwitch as _OVSSwitch

class PatchedOVSSwitch(_OVSSwitch):
    # Give Mininet something sane so isOldOVS() won't crash
    OVSVersion = '2.5'   # any >= '2.0' is fine for the check
    
import ipaddress
import itertools
import math
import os
import time
import datetime
import resources.clean_containernet as clean_containernet
import yaml
import docker
import argparse
import random
from resources.performance import device_profile,network_profile

first = False

# FIXED/DEFAULT values
PWD = os.getcwd()
D_IMAGE_NAME = "fed_opt"
FL_TYPE = "fed_opt"
CLIENTS = 10
PROTOCOL = "grpc"

HOST_SINGLE_CORE_SCORE = 1079
CORE_NUM = 20
RAM_LIMIT = ""

ROUTER_DELAY_MS = 12
ROUTER_BW = 80
ROUTER_JIT = 2

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='pass the framework config file folder')
parser.add_argument('--config', help='pass the emulation config file')
parser.add_argument('--interactive', action='store_true', help='run in interactive mode')

args = parser.parse_args()
folder_name = args.name
config = args.config
interactive = args.interactive

clean_containernet.clean_session()
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Global configuration variables
alpha = None
fl_method = "None_Experiment"
clients = CLIENTS
image_name = D_IMAGE_NAME
protocol = PROTOCOL
fl_type = FL_TYPE
ntw_type = None
device_type = None
ntw_type_str = "nan"
device_type_str = "nan"
host_single_core_score = HOST_SINGLE_CORE_SCORE
device_variance = 0.2
folder_name = "./"

# New configuration variables
enable_tcpdump = True
tcp_dump_first = False
enable_mqtt = False
mqtt_config = {}
node_configs = []
executed_commands = []  # Store all executed commands for logging
mqtt_broker_container = None  # Global reference to broker container

# Parse the config file yaml
if config:
    
    info(f"Using config file: {config}\n")
    
    with open(config, "r") as stream:
        try:
            config_data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            error(exc)
    
    experiment_name = config_data.get('experiment_name', "None_Experiment")
    clients = config_data.get('clients', CLIENTS)
    image_name = config_data.get('image_name', D_IMAGE_NAME)
    protocol = config_data.get('protocol', PROTOCOL)
    fl_type = config_data.get('fl_type', FL_TYPE)
    ntw_type = config_data.get('network_type', None)
    device_type = config_data.get('device_type', None)
    host_single_core_score = config_data.get('host_single_core_score', HOST_SINGLE_CORE_SCORE)
    
    device_variance = config_data.get('device_variance', 0.2)
    
    # New configuration options
    enable_tcpdump = config_data.get('enable_tcpdump', True)
    enable_mqtt = config_data.get('enable_mqtt', False)
    mqtt_config = config_data.get('mqtt_config', {})
    node_configs = config_data.get('nodes', [])
    
    if type(device_type) is list:
        device_type_str = device_type[0]
        
    if type(ntw_type) is list:
        ntw_type_str = ntw_type[0]
        
    config_abs = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_abs)
    config_base = os.path.splitext(os.path.basename(config_abs))[0]
    folder_name = os.path.relpath(config_dir, PWD)
    
    

image_name_str = image_name.split("/")[-1] if "/" in image_name else image_name
image_name_str = image_name_str.split(":")[0] if ":" in image_name_str else image_name_str

info(f"*** Running {experiment_name} experiment\n")
info(f"*** Using {alpha} alpha\n")
info(f"*** Using {clients} clients\n")
info(f"*** Using {image_name} image\n")
info(f"*** Using {fl_type} FL type\n")
info(f"*** Using {ntw_type} network type\n")
info(f"*** Using {device_type} device type\n")
info(f"*** Using {host_single_core_score} single core score\n")
info(f"*** tcpdump enabled: {enable_tcpdump}\n")
info(f"*** MQTT enabled: {enable_mqtt}\n")

output_dir_name = "{}_{}_C{}_{}_{}_{}".format(
    timestamp, image_name_str, clients, 
    str(ntw_type_str), str(device_type_str), str(experiment_name)
)
output_dir_name = os.path.join(folder_name, output_dir_name)

class MyContainer:
    def __init__(self, _id, fl_type, router_ip, node_config=None, cpu_period=0, cpu_share=0, 
                 cpu_quota=0, nano_cpu=0, cpu=0, ram=""):
        self.fl_type = fl_type
        self.id = _id
        self.name = "{}{}".format(self.fl_type, self.id)
        self.router_ip = router_ip
        self.default_route = self.router_ip[1].compressed
        self.address = "{}/24".format(self.router_ip[100].compressed)
        self.address_2 = "{}".format(self.router_ip[100].compressed)
        self.bind_port = 1880 + self.id
        self.master = "{}@{}".format(self.name, self.address)
        
        # Node-specific configuration
        self.node_config = node_config or {}
        self.custom_image = self.node_config.get('image', image_name)
        self.custom_command = self.node_config.get('command', None)
        
        # Device constraints
        constraints = self.node_config.get('constraints', {})
        self.cpu_period = constraints.get('cpu_period', cpu_period)
        self.cpu_share = constraints.get('cpu_share', cpu_share)
        self.cpu_quota = constraints.get('cpu_quota', cpu_quota)
        self.nano_cpu = constraints.get('nano_cpu', nano_cpu)
        self.cpu = constraints.get('cpu_cores', cpu)
        self.ram = f"{constraints.get('memory_mb', 512)}m" if constraints.get('memory_mb') else ram
        

    def get_master(self):
        return self.master

    def set_master(self, new_master):
        self.master = new_master

class MyRouter:
    def __init__(self, _id, networkIP):
        self.id = _id
        self.name = "r{}".format(self.id)
        self.networkIP = networkIP
        self.mainIP = '{}/24'.format(next(self.networkIP.hosts()))
        self.eth_available = ['{}-eth{}'.format(self.name, eth) for eth in range(clients+1)]
        self.switch = None
        self.eth_used = []
        self.routing_binding = []
        self.router = net.addHost(self.name, cls=LinuxRouter, ip=self.mainIP)

    def get_eth(self):
        eth = self.eth_available.pop(0)
        self.eth_used.append(eth)
        return eth

    def add_binding(self, bind):
        self.routing_binding.append(bind)

    def add_switch(self):
        self.switch = net.addSwitch('s{}'.format(self.id))
        self.switch.cmd('ethtool -K', self.switch,
                'gro off',
                'tx off',
                'rx off')

        net.addLink(self.switch, self.router,
                    intfName=self.get_eth(),
                    params2={'ip': self.mainIP})
        return self.switch


class LinuxRouter(Node):
    def config(self, **params):
        super(LinuxRouter, self).config(**params)
        # Enable forwarding on the router
        self.cmd('sysctl net.ipv4.ip_forward=1')
        self.cmd('ethtool -K', self,
                'gro', 'off',
                'tx','off',
                'rx', 'off')
    def terminate(self):
        self.cmd('sysctl net.ipv4.ip_forward=0')

        super(LinuxRouter, self).terminate()
        
        

def add_variability(value, variation_percentage=0.3):
    variation = value * variation_percentage
    return round(value + random.uniform(-variation, variation), 4)

def get_node_config(node_id):
    """Get configuration for a specific node"""
    for node in node_configs:
        if node.get('id') == node_id:
            return node
    return {}

def core_network():
    net.addController('c0', port=6654)

    info('*** Adding routers\n')
    _routers = [MyRouter(_id=cli, networkIP=ipaddress.ip_network('10.0.{}.0/24'.format(cli)))
                for cli in range(clients+1)]

    info('*** Adding switches\n')
    _switches = [r.add_switch() for r in _routers]

    info('*** Adding router-router links\n')
    for (router1, router2) in list(itertools.combinations(_routers, 2)):
        intf_name1 = router1.get_eth()
        intf_name2 = router2.get_eth()
        params1 = '10.{}.0.1'.format(int(''.join(sorted(str(e) for e in [router1.id, router2.id]))), reverse=True)
        params2 = '10.{}.0.2'.format(int(''.join(sorted(str(e) for e in [router1.id, router2.id]))), reverse=True)

        router1.add_binding((router2.networkIP, params2, intf_name1))
        router2.add_binding((router1.networkIP, params1, intf_name2))

        # Use node-specific link configuration if available
        node1_config = get_node_config(router1.id)
        node1_link = node1_config.get('link', {})
        
        if ntw_type is not None:
            ntw_prof = network_profile(ntw_type, router2.id)
            delay = ntw_prof['delay_ms']
            bw = ntw_prof['band_mbps']
            jitter = ntw_prof['jitter_ms']
            loss_percent = ntw_prof.get('loss_percent', 0)

        elif node1_link:
            delay = node1_link.get('delay_ms', ROUTER_DELAY_MS) / 2
            bw = node1_link.get('bandwidth_mbps', ROUTER_BW)
            jitter = node1_link.get('jitter_ms', ROUTER_JIT)
            loss_percent = node1_link.get('loss_percent', 0)
        else:
            delay = add_variability(ROUTER_DELAY_MS) / 2
            bw = add_variability(ROUTER_BW)
            jitter = add_variability(ROUTER_JIT)
            loss_percent = 0
        
        net.addLink(router1.router, router2.router,
                    intfName1=intf_name1,
                    intfName2=intf_name2,
                    params1={'ip': '{}/24'.format(params1)},
                    params2={'ip': '{}/24'.format(params2)},
                    cls=TCLink,
                    delay='{}ms'.format(delay), bw=bw, jitter='{}ms'.format(jitter),loss=loss_percent)

    info('*** Adding routing\n')
    _cmd = "ip route add {to_reach} via {host} dev {eth_int}"
    for r in _routers:
        for bind in r.routing_binding:
            r.router.cmd(_cmd.format(to_reach=bind[0], host=bind[1], eth_int=bind[2]))

    return _switches, _routers

def create_mqtt_broker(ps_router):
    """Create a separate MQTT broker container in the PS subnet (10.0.0.x)"""
    global mqtt_broker_container
    
    broker_image = mqtt_config.get('broker_image', 'eclipse-mosquitto:latest')
    broker_ip = "10.0.0.1/24"  # Use .99 for the broker in PS subnet
    broker_ip_only = "10.0.0.1"
    
    info(f'*** Creating separate MQTT broker container at {broker_ip_only}\n')
    
    # Create the broker container
    mqtt_broker_container = net.addDocker(
        name='mqtt_broker',
        ip=broker_ip,
        dimage=broker_image,
        volumes=["{}/output/{}:/app/saved_output".format(PWD, output_dir_name)],
        privileged=True
    )
    
    # Disable network offload features
    mqtt_broker_container.cmd("ethtool -K eth0 gro off tx off rx off")
    
    # Link the broker to the PS switch (router 0's switch)
    net.addLink(mqtt_broker_container, ps_router.switch, cls=TCLink)
    
    # Add routing so broker can reach other networks
    mqtt_broker_container.cmd(f"ip route add 10.0.0.0/16 via {ps_router.networkIP[1].compressed} dev mqtt_broker-eth0")
    
    return mqtt_broker_container, broker_ip_only

def create_containers(fl_type, _routers, host_list=None):
    global device_type
    def setup_container(container):
        dck = net.addDocker(
            name=container.name, 
            ip=container.address, 
            dimage=container.custom_image,
            volumes=["{}/output/{}:/app/saved_output".format(PWD, output_dir_name)],
            mem_limit=container.ram,
            cpu_period=container.cpu_period,
            cpu_quota=container.cpu_quota,
            privileged=True
        )
        dck.cmd("ethtool -K eth0 gro off tx off rx off")
        return dck

    if host_list is None:
        host_list = []

    for r in _routers:
        node_config = get_node_config(r.id)
        
        if device_type is not None or (node_config != {} and 'device_type' in node_config):
                    
            if device_type is None:
                device_type = node_config['device_type']
                    
            device = device_profile(r.id, host_single_core_score, device_type, clients, variation=device_variance)
            info(f"*** Using {device} device profile\n") 
                    
            container = MyContainer(
                r.id, fl_type, r.networkIP,
                node_config={},
                cpu_period=int(device['CpuPeriod']),
                cpu_quota=int(device['CpuQuota']),
                cpu_share=int(device['CpuShares']),
                nano_cpu=int(device['NanoCPUs']),
                cpu=int(device['Cpus']),
                ram=device['Memory']
            )
            
        else:
            container = MyContainer(r.id, fl_type, r.networkIP, node_config=node_config, ram=RAM_LIMIT)
        
        host_list.append({"cls": container, "doc": setup_container(container)})

    return host_list

def add_routing(container_list_full):
    container_list = [c["doc"] for c in container_list_full]
    container_class = [c["cls"] for c in container_list_full]

    for cli, cli_cls in zip(container_list, container_class):
        cli.cmd("ip route add 10.0.0.0/16 via {} dev {}-eth0".format(cli_cls.default_route, cli_cls.name))

def log_command(node_name, command):
    """Log executed command to results"""
    global executed_commands
    log_entry = {
        'timestamp': datetime.datetime.now().isoformat(),
        'node': node_name,
        'command': command
    }
    executed_commands.append(log_entry)
    info(f"*** Command logged [{node_name}]: {command}\n")

def start_fl(container_list_full, _routers):
    def get_port():
        protocol_ports = {
            'tcp': 80,
            'grpc': 50051,
            'rest': 80,
            'coap': 5683,
            'mqtt': 1883,
            'websocket': 8080,
            'amqp': 5672
        }
        return protocol_ports.get(protocol.lower(), 50051)

    server_port = get_port()
    info(f"*** Switching to default port for {protocol.upper()} protocol: {server_port}\n")

    def add_arg(key, value):
        return f"--{key} {value}" if value is not None else None

    def get_args(file):
        with open(f"./{folder_name}/{file}", "r") as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                error(exc)
        args = [
            add_arg("protocol", config.get('protocol', protocol)),
            add_arg("rounds", config.get('rounds')),
            add_arg("fl_method", config.get('fl_method')),
            add_arg("alpha", config.get('alpha')),
            add_arg("min_clients", config.get('server_config', {}).get('min_client_to_start')),
            add_arg("num_client", config.get('server_config', {}).get('client_round')),
            add_arg("client_selection", config.get('client_selection')),
            add_arg("epochs", config.get('client_config', {}).get('epochs')),
            add_arg("parts_dataset", config.get('num_parts_dataset')),
            add_arg("asofed_beta", config.get('asofed_beta')),
        ]
        args = [arg for arg in args if arg is not None]
        return args
    
    def render_command_template(template: str, *, srv_addr, cli_cls, server_port, protocol, arguments):
        """Replace placeholders in custom command strings."""
        if not template:
            return ''
        ctx = {
            'server_address': srv_addr,
            'client_address': getattr(cli_cls, 'address_2', ''),
            'server_port': server_port,
            'protocol': protocol,
            'client_id': getattr(cli_cls, 'id', ''),
            'args': ' '.join(arguments or []),
        }
        try:
            return template.format(**ctx)
        except Exception:
            # If formatting fails, do a minimal replacement of the two core placeholders
            cmd = template.replace('{server_address}', str(ctx['server_address']))
            cmd = cmd.replace('{client_address}', str(ctx['client_address']))
            return cmd
        
    def start_fedmingle(container_list_full, file_name):
        global first, mqtt_broker_container
        try:
            container_list = [c["doc"] for c in container_list_full]
            container_class = [c["cls"] for c in container_list_full]

            ps = container_list[0]
            ps_cls = container_class[0]
            client_list = container_list[1:]
            client_class = container_class[1:]

            arguments = get_args(file_name)
            info(f"args: {arguments}\n")
            name_without_extension = file_name.rsplit(".", 1)[0]
            
            info('*** Running TRAFFIC DUMP from Parameter Server\n')

            offload_commands = "gro off tx off rx off lro off gso off tso off sg off rxvlan off txvlan off rxhash off ufo off"
            
            # Conditional tcpdump
            if enable_tcpdump:
                ps.cmd(f"ethtool -K eth0 {offload_commands} && ethtool -K {ps.name}-eth0 {offload_commands}")
                tcpdump_cmd = f"tcpdump -i any -w /app/saved_output/{name_without_extension}_ps_{ps.name}-eth0.pcap &"
                ps.cmd(tcpdump_cmd)
                log_command(ps.name, tcpdump_cmd)
                info(f"*** tcpdump enabled on {ps.name}\n")
            
            # MQTT Broker setup
            broker_ip = ps_cls.address_2  # Default to PS IP
            
            if enable_mqtt or protocol == "mqtt" or protocol == "websocket":
                broker_location = mqtt_config.get('broker_location', 'ps')
                broker_image = mqtt_config.get('broker_image', 'eclipse-mosquitto:latest')
                broker_commands = mqtt_config.get('broker_commands', [])
                
                if broker_location == 'separate':
                    info('*** Creating separate MQTT broker container\n')
                    # Create the broker in PS subnet
                    ps_router = _routers[0]  # Router 0 is the PS router
                    mqtt_broker_container, broker_ip = create_mqtt_broker(ps_router)
                    
                    # Start broker services
                    if len(broker_commands) > 0:
                        info(f"*** Using custom broker commands: {broker_commands}\n")
                        for broker_cmd in broker_commands:
                            if broker_cmd.strip():
                                mqtt_broker_container.cmd(broker_cmd)
                                log_command('mqtt_broker', broker_cmd)
                    
                    time.sleep(5)
                    info(f"*** MQTT broker running at {broker_ip}\n")
                    
                elif broker_location == 'ps':
                    info('*** Running MQTT broker on Parameter Server\n')
                    if len(broker_commands) > 0:
                        info(f"*** Using custom broker commands: {broker_commands}\n")
                        for broker_cmd in broker_commands:
                            if broker_cmd.strip():
                                ps.cmd(broker_cmd)
                                log_command(ps.name, broker_cmd)
                    else:
                        info(f"*** Using default mosquitto broker\n")
                        mosquitto_cmd = f"mosquitto -c /etc/mosquitto/mosquitto.conf -v > /app/saved_output/{name_without_extension}_mosquitto_{ps.name}.log 2>&1 &"
                        ps.cmd(mosquitto_cmd)
                        log_command(ps.name, mosquitto_cmd)
                    
                    time.sleep(5)
            
            info(f'*** Running FL Parameter Server at ip {ps_cls.address_2}\n')
            
            srv_addr = ps_cls.address_2
             
            if broker_ip != srv_addr:
                info(f'*** MQTT Broker is at {broker_ip}\n')
                srv_addr = broker_ip
            # Use custom command if specified
            if ps_cls.custom_command:
                cmd = render_command_template(
                    ps_cls.custom_command,
                    srv_addr=srv_addr,
                    cli_cls=ps_cls,
                    server_port=server_port,
                    protocol=protocol,
                    arguments=arguments,
                )
            else:
                cmd = f"python3 -u run.py --protocol {protocol} --mode Server --port {server_port} --ip {srv_addr} --index {ps_cls.id} {' '.join(arguments)}"
            
            cmd += f" > /app/saved_output/{name_without_extension}_ps_{ps.name}.log 2>&1"
            ps.sendCmd(cmd)
            log_command(ps.name, cmd)
            time.sleep(20)

            info('*** Running FL Clients\n')
            for cli, cli_cls in zip(client_list, client_class):
                if enable_tcpdump and not first:
                    tcpdump_client_cmd = f"tcpdump -i any -w /app/saved_output/{name_without_extension}_{cli.name}_{protocol}.pcap &"
                    cli.cmd(tcpdump_client_cmd)
                    log_command(cli.name, tcpdump_client_cmd)
                    first = True
                
                cli.cmd(f"ethtool -K {cli.name}-eth0 {offload_commands}")
                
                # Use custom command if specified
                if cli_cls.custom_command:
                    cmd = render_command_template(
                        cli_cls.custom_command,
                        srv_addr=srv_addr,
                        cli_cls=cli_cls,
                        server_port=server_port,
                        protocol=protocol,
                        arguments=arguments,
                    )
                else:
                    cmd = f"python3 -u run.py --protocol {protocol} --mode Client --my_ip {cli_cls.address_2} --port {server_port} --ip {srv_addr} --index {cli_cls.id} {' '.join(arguments)}"
                
                cmd += f" > /app/saved_output/{name_without_extension}_cli_{cli.name}.log 2>&1"
                cli.sendCmd(cmd)
                log_command(cli.name, cmd)

            info('*** Waiting PS Ending\n')
            ps.waitOutput()
            for cli in client_list:
                cli.waitOutput()
            
            # Save command log to file
            log_file = f"./output/{output_dir_name}/commands_executed.json"
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'w') as f:
                json.dump(executed_commands, f, indent=2)
            info(f"*** Commands logged to {log_file}\n")
            
        except Exception as e:
            error(f"Exception: {e}\n")
            net.stop()
            clean_containernet.clean_session()

    config_folder = f"./{folder_name}"
    config_files = [os.path.basename(args.config)]

    if not config_files:
        info(f"No config files in {config_folder} folder.\n")
    else:
        for i, config_file in enumerate(config_files):
            info(f"Using file {config_file} as configuration\n")
            start_fedmingle(container_list_full, config_file)

def main():
    try:
        info('\n\tFL type: {}\n'.format(FL_TYPE))
        info('\n\tNumber of clients: {}\n'.format(clients))
        info('\n\tFor FL arguments setting see {} file!\n'.format(folder_name))
        info('\n')

        switches, routers = core_network()
        ip_routers = [r.networkIP for r in routers]
        router_list = [r.router for r in routers]

        container_list_full = create_containers(FL_TYPE, routers)

        container_list = [c["doc"] for c in container_list_full]

        info('\n*** Adding container-switch links\n')
        for c, s in zip(container_list, switches):
            info(net.addLink(c, s, cls=TCLink))

        info('*** Starting network\n')
        net.start()

        add_routing(container_list_full)

        info('*** Testing connectivity\n')

        info('\n*** Waiting the network start up ({} secs)...\n'.format(ROUTER_DELAY_MS / 2))
        time.sleep(ROUTER_DELAY_MS / 2)

        if interactive:
            info('*** Starting CLI\n')
            CLI(net)
        
        info('*** Running FL Test\n')
        os.system("bash resources/disable_offload.sh >/dev/null 2>&1")
        start_fl(container_list_full, routers)

        info('*** Stopping network\n')
        net.stop()
        clean_containernet.clean_session()

    except Exception as e:
        error(f"Exception: {e}\n")
        net.stop()
        clean_containernet.clean_session()

if __name__ == '__main__':
    import json
    net = Containernet(controller=Controller, switch=PatchedOVSSwitch)
    main()
    clean_containernet.clean_session()