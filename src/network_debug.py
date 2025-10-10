from mininet.net import Containernet
from mininet.node import Controller
from mininet.node import Node
from mininet.link import TCLink
from mininet.log import debug, info, error
from mininet import *
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

# FIXED/DEFUALT values
PWD = os.getcwd() # get current working directory
D_IMAGE_NAME = "fed_opt" # "fed_mingle" # docker image name
FL_TYPE = "fed_opt"
CLIENTS = 10
PROTOCOL = "grpc"

HOST_SINGLE_CORE_SCORE = 1079 # single core score for the device from geekbench (https://browser.geekbench.com/processors/intel-xeon-silver-4210r)
CORE_NUM = 20
RAM_LIMIT = ""



ROUTER_DELAY_MS = 12 # millisecond
ROUTER_BW = 80 # Mb/s
ROUTER_JIT = 2 # millisecond



# read name for output dir
parser = argparse.ArgumentParser()
parser.add_argument('--name', help='pass the framework config file folder')
parser.add_argument('--config', help='pass the emulation config file')

args = parser.parse_args()
folder_name = args.name # "fedavg_lr", "scaffold_cs", "scaffold_pruning"
config = args.config


clean_containernet.clean_session()
timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")



#parse the config file yaml
if config:
    with open(config, "r") as stream: # config as parameter
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            error(exc)
    
    fl_method = config.get('fl_method', None)
    alpha = config.get('alpha', None)
    clients = config.get('clients', CLIENTS)
    image_name = config.get('image_name',D_IMAGE_NAME)
    protocol = config.get('protocol', PROTOCOL)
    fl_type = config.get('fl_type', FL_TYPE)
    ntw_type = config.get('network_type', None)
    device_type = config.get('device_type', None)
    host_single_core_score = config.get('host_single_core_score', HOST_SINGLE_CORE_SCORE)
    
    config_abs = os.path.abspath(args.config)
    config_dir = os.path.dirname(config_abs)
    config_base = os.path.splitext(os.path.basename(config_abs))[0]
    folder_name = os.path.relpath(config_dir, PWD)
    
    device_variance = config.get('device_variance', 0.2)

else:
    alpha = None
    fl_method = None
    clients = CLIENTS
    image_name = D_IMAGE_NAME
    protocol = PROTOCOL
    fl_type = FL_TYPE
    ntw_type = None
    device_type = None
    host_single_core_score = HOST_SINGLE_CORE_SCORE
    device_variance = 0.2 #Variance in percentage to the device profile

if type(device_type) is list:
    device_type_str = device_type[0]
    
if type(ntw_type) is list:
    ntw_type_str = ntw_type[0]
    
info(f"*** Using {fl_method} FL method")
info(f"*** Using {alpha} alpha")
info(f"*** Using {clients} clients")
info(f"*** Using {image_name} image")
info(f"*** Using {protocol} protocol")
info(f"*** Using {fl_type} FL type")
info(f"*** Using {ntw_type} network type")
info(f"*** Using {device_type} device type")
info(f"*** Using {host_single_core_score} single core score")
    

output_dir_name = "{}_{}_C{}_{}_{}_{}_A{}_{}".format(protocol,image_name,clients,str(ntw_type_str),str(device_type_str),str(fl_method),alpha,timestamp)
output_dir_name = os.path.join(folder_name,output_dir_name)



class MyContainer:
    def __init__(self, _id, fl_type, router_ip, cpu_period=0,cpu_share=0,cpu_quota=0,nano_cpu=0,cpu=0, ram=""):
        self.fl_type = fl_type
        self.id = _id
        self.name = "{}{}".format(self.fl_type, self.id)
        self.router_ip = router_ip
        self.default_route = self.router_ip[1].compressed
        self.address = "{}/24".format(self.router_ip[100].compressed)
        self.address_2 = "{}".format(self.router_ip[100].compressed)
        self.bind_port = 1880 + self.id
        self.master = "{}@{}".format(self.name, self.address)
        self.cpu_period = cpu_period
        self.cpu_share = cpu_share
        self.cpu_quota = cpu_quota
        self.nano_cpu = nano_cpu
        self.cpu = cpu
        self.ram = ram

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

def assign_cpu(core_list=None):
    if core_list is None:
        return 0
    else:
        core_per_broker = math.floor(CORE_NUM / clients+1)

        this_cpu = []
        for i in range(core_per_broker):
            this_cpu.append(core_list.pop(0))

        return ','.join(map(str, this_cpu))


def core_network():
    info('\n*** Adding controller')
    net.addController('c0', port=6654)
    info('Controller Added\n')

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

        if ntw_type is not None:
            ntw_prof = network_profile(ntw_type,router2.id)
            delay = ntw_prof['delay_ms']
            bw = ntw_prof['band_mbps']
            jitter = ntw_prof['jitter_ms']

        else:
            delay = add_variability(ROUTER_DELAY_MS) / 2 # values on RTT
            bw = add_variability(ROUTER_BW)
            jitter = add_variability(ROUTER_JIT) / 2 # values on RTT
        
        net.addLink(router1.router, router2.router,
                    intfName1=intf_name1,
                    intfName2=intf_name2,
                    params1={'ip': '{}/24'.format(params1)},
                    params2={'ip': '{}/24'.format(params2)},
                    cls=TCLink,
                    delay='{}ms'.format(delay), bw=bw, jitter='{}ms'.format(jitter))

    info('*** Adding routing\n')
    _cmd = "ip route add {to_reach} via {host} dev {eth_int}"
    for r in _routers:
        for bind in r.routing_binding:
            r.router.cmd(_cmd.format(to_reach=bind[0], host=bind[1], eth_int=bind[2]))

    return _switches, _routers


def add_variability(value, variation_percentage=0.3):
    variation = value * variation_percentage
    return round(value + random.uniform(-variation, variation),4)


def create_containers(fl_type, _routers, host_list=None):

    def setup_fedmingle(container):
        dck = net.addDocker(name=container.name, ip=container.address, dimage=D_IMAGE_NAME,
                            volumes=["{}/output/{}:/app/saved_output".format(PWD,output_dir_name)],
                            mem_limit=container.ram,
                            cpu_period=container.cpu_period,
                            cpu_quota=container.cpu_quota,
                            privileged=True)
                            #device_requests=[docker.types.DeviceRequest(device_ids=["0"], capabilities=[['compute','utility']])])
        dck.cmd("ethtool -K eth0 gro off tx off rx off")
        return dck

    if host_list is None:
        host_list = []

    for r in _routers:
        if device_type is not None:
            device = device_profile(r.id,host_single_core_score,device_type,clients,variation=device_variance)
            info(f"*** Using {device} device profile")
            container = MyContainer(r.id, fl_type, r.networkIP,
                                    cpu_period=int(device['CpuPeriod']),
                                    cpu_quota=int(device['CpuQuota']),
                                    cpu_share=int(device['CpuShares']),
                                    nano_cpu=int(device['NanoCPUs']),
                                    cpu=int(device['Cpus']),
                                    ram=device['Memory'])
        else:
            container = MyContainer(r.id, fl_type, r.networkIP, ram=RAM_LIMIT)#, cpu_limit=0.5, cpu_sched='cfs'
        host_list.append({"cls":container,"doc":setup_fedmingle(container)})

    return host_list

def add_routing (container_list_full):
    container_list = [c["doc"] for c in container_list_full]
    container_class = [c["cls"] for c in container_list_full]

    for cli, cli_cls in zip(container_list, container_class):
        cli.cmd("ip route add 10.0.0.0/16 via {} dev {}-eth0".format(cli_cls.default_route, cli_cls.name))

def start_fl(container_list_full):

    def get_port():
        protocol_ports = {
            'tcp': 80,        # Default HTTP port
            'grpc': 50051,    # Default gRPC port
            'rest': 80,       # Typically RESTful services run over HTTP
            'coap': 5683,     # Default CoAP port
            'mqtt': 1883,     # Default MQTT port
            'amqp': 5672      # Default AMQP port
            }
        return protocol_ports.get(protocol.lower(), 'Unknown protocol')

    server_port = get_port()
    info(f"*** Switching to default port for {protocol.upper()} protocol: {server_port}")

    def add_arg(key, value):
        return f"--{key} {value}" if value is not None else None

    def get_args(file):
        with open(f"./{folder_name}/{file}", "r") as stream: # config as parameter
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                error(exc)
        #TODO x MARTA
        args = [
            add_arg("protocol", config.get('protocol', protocol)), #added now
            add_arg("rounds", config.get('rounds')),
            add_arg("num_parts_dataset", config.get('num_parts_dataset')),
            add_arg("fl_method", config.get('fl_method')),
            add_arg("alpha", config.get('alpha')),
            add_arg("min_clients", config.get('server_config', {}).get('min_client_to_start')),
            add_arg("num_client", config.get('server_config', {}).get('client_round')),
            add_arg("client_selection", config.get('client_selection')),
        ]
        args = [arg for arg in args if arg is not None]
        return args
    
    def start_fedmingle(container_list_full, file_name):
        global first
        try:
            container_list = [c["doc"] for c in container_list_full]
            container_class = [c["cls"] for c in container_list_full]

            ps = container_list[0]
            client_list = container_list[1:]
            client_class = container_class[1:]

            # add FL arguments
            arguments = get_args(file_name)
            info(f"args: {arguments}\n")
            name_without_extension = file_name.rsplit(".", 1)[0]

            info('*** Running TRAFFIC DUMP from Parameter Server\n')

            offload_commands = "gro off tx off rx off lro off gso off tso off sg off rxvlan off txvlan off rxhash off ufo off"
            ps.cmd(f"ethtool -K eth0 {offload_commands} && ethtool -K {ps.name}-eth0 {offload_commands}")

            ps.cmd(f"tcpdump -i any -w /app/saved_output/{name_without_extension}_ps_{ps.name}-eth0.pcap &")

            if protocol == "mqtt":
                info('*** Running MQTT broker\n')
                ps.cmd("mosquitto -c /etc/mosquitto/mosquitto.conf &")

            info(f'*** Running FL Parameter Server at ip {container_class[0].address_2}\n')

            cmd = f"python3 run.py > /app/saved_output/{name_without_extension}_ps_{ps.name}.log --protocol {protocol} --mode Server --port {server_port} --ip {container_class[0].address_2} --index {container_class[0].id} {' '.join(arguments)}"
            ps.sendCmd(cmd)
            time.sleep(20)

            info('*** Running FL Clients\n')
            for cli, cli_cls in zip(client_list, client_class):
                # > /app/saved_output/{name_without_extension}_{cli.name}.log
                if not first:
                    cli.cmd(f"tcpdump -i any -w /app/saved_output/{name_without_extension}_{cli.name}_{protocol}.pcap &")
                    first = True
                cli.cmd(f"ethtool -K {cli.name}-eth0 {offload_commands}")
                cmd = f"python3 run.py > /app/saved_output/{name_without_extension}_cli_{cli.name}.log  --protocol {protocol} --mode Client --my_ip {cli_cls.address_2} --port {server_port} --ip {container_class[0].address_2} --index {cli_cls.id} {' '.join(arguments)}"
                cli.sendCmd(cmd)
                # time.sleep(5)

            info('*** Waiting PS Ending\n')
            ps.waitOutput()
            for cli in client_list:
                cli.waitOutput()
        except Exception as e:
            error(f"Exception: {e}")
            net.stop()
            clean_containernet.clean_session()

    # load all config files
    config_folder = f"./{folder_name}"

    config_files = [os.path.basename(args.config)]

    if not config_files:
        info(f"No config files in {config_folder} folder.")
    else:
        for i, config_file in enumerate(config_files):
            info(f"Using file {config_file} as configuration")
            start_fedmingle(container_list_full, config_file)

def main():
    try:
        info('\n\tFL type: {}'.format(FL_TYPE))
        info('\n\tNumber of clients: {}'.format(clients))
        info('\n\tFor FL arguments setting see {} file!'.format(folder_name))
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
        #net.pingAll()
        # for pairs in list(itertools.permutations(router_list + container_list, 2)):
        #     net.ping(pairs)

        #for elem in container_list[1:]:
        #    net.ping([container_list[0], elem])

        info('\n*** Waiting the network start up ({} secs)...\n'.format(ROUTER_DELAY_MS / 2))
        time.sleep(ROUTER_DELAY_MS / 2)

        os.system("bash resources/disable_offload.sh >/dev/null 2>&1")

        info('*** Running FL Test\n')
        start_fl(container_list_full)

        info('*** Stopping network\n')
        net.stop()
        clean_containernet.clean_session()

    except Exception as e:
        error(f"Exception: {e}")
        net.stop()
        clean_containernet.clean_session()

if __name__ == '__main__':
    net = Containernet(controller=Controller)
    main()
    clean_containernet.clean_session()
