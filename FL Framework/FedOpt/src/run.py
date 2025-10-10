# MAIN FILE
import sys
import argparse
import yaml
import os
from Utils.logging_fedopt import *

# sys path for Docker
#module_path = '/usr/src/app'

module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if module_path not in sys.path:
    sys.path.append(module_path)

def main():
    # add positional arguments
    parser = argparse.ArgumentParser(description='Communication Protocol Info')
    # General settings
    parser.add_argument('--ip', help='')
    parser.add_argument('--my_ip', help='')
    parser.add_argument('--port', help='')
    parser.add_argument('--protocol', help='TCP/UDP/...')
    parser.add_argument('--mode', help='Enter Server/Client')
    parser.add_argument('--fl_method', help='FedAvg/FedDyn/SCAFFOLD')
    parser.add_argument('--model', help='AlexNet/MLP')
    parser.add_argument('--device', help='cpu/cuda')
    # Dataset settings
    parser.add_argument('--dataset', help='MNIST/CIFAR10')
    parser.add_argument('--parts_dataset', help='')
    parser.add_argument('--test_size', help='')
    parser.add_argument('--iid', help='')
    parser.add_argument('--alpha', help='Dirichlet distribution')
    parser.add_argument('--train_batch', help='')
    parser.add_argument('--test_batch', help='')
    # FL client settings
    parser.add_argument('--epochs', help='')
    parser.add_argument('--lr', help='')
    parser.add_argument('--alpa_feddyn', help='')
    parser.add_argument('--asofed_beta', help='')
    # FL server settings
    parser.add_argument('--rounds', help='')
    parser.add_argument('--num_clients', help='')
    parser.add_argument('--global_lr', help='')
    parser.add_argument('--min_clients', help='')
    parser.add_argument('--max_time', help='Maximum training time')
    parser.add_argument('--max_accuracy', help='Maximum accuracy')
    # FedAsync settings
    parser.add_argument('--fedAsync_variant', help='')

    # FL optimizations settings
    parser.add_argument('--dyn_sampling', help='')
    parser.add_argument('--decay_rate', help='')
    parser.add_argument('--client_selection', help='')
    parser.add_argument('--model_pruning', help='')
    parser.add_argument('--prune_interval', help='')
    parser.add_argument('--prune_ratio', help='')
    parser.add_argument('--client_clustering', help='')
    # Verbose level
    parser.add_argument('--verbose', help='')

    # Configurations algorithms parameters
    parser.add_argument('--unweighted_mu_0', help='')
    parser.add_argument('--fedAsync_a_hinge', help='')
    parser.add_argument('--fedAsync_b_hinge', help='')
    parser.add_argument('--fedAsync_a_poly', help='')
    parser.add_argument('--fedAsync_mu_0', help='')
    parser.add_argument('--asyncFedED_lambda', help='')
    parser.add_argument('--asyncFedED_epsilon', help='')
    parser.add_argument('--asyncFedED_gamma_bar', help='')
    parser.add_argument('--asyncFedED_k', help='')
    parser.add_argument('--asofed_rho', help='')
    parser.add_argument('--asofed_global_lr', help='')
    parser.add_argument('--accuracy_file', default=None, help='Accuracy file path')
    parser.add_argument('--loss_file', default=None, help='Loss file path')

    parser.add_argument('--index', help='Client index')
    
    args = parser.parse_args()

    # set logging level
    level_str = args.verbose if args.verbose else "debug"
    logger = setup_logger("FedOpt", level_str=level_str)

    # load also config file
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, '..', 'resources', 'config.yaml')
    with open(file_path, "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    # Update YAML configuration with argparse values (grouped as before)
    config['ip'] = args.ip if args.ip else config['ip']
    config['my_ip'] = args.my_ip if args.my_ip else config['my_ip']
    config['port'] = int(args.port) if args.port else config['port']
    config['protocol'] = args.protocol if args.protocol else config['protocol']
    config['mode'] = args.mode if args.mode else config['mode']
    config['fl_method'] = args.fl_method if args.fl_method else config['fl_method']
    config['model'] = args.model if args.model else config['model']
    config['device'] = args.device if args.device else config['device']

    config['dataset'] = args.dataset if args.dataset else config['dataset']
    config['num_parts_dataset'] = int(args.parts_dataset) if args.parts_dataset else config['num_parts_dataset']
    config['test_size'] = int(args.test_size) if args.test_size else config['test_size']
    config['iid_ratio'] = float(args.iid) if args.iid else config['iid_ratio']
    config['alpha'] = float(args.alpha) if args.alpha else config['alpha']
    config['train_batch_size'] = int(args.train_batch) if args.train_batch else config['train_batch_size']
    config['test_batch_size'] = int(args.test_batch) if args.test_batch else config['test_batch_size']

    config['client_config']['epochs'] = int(args.epochs) if args.epochs else config['client_config']['epochs']
    config['client_config']['local_step_size'] = float(args.lr) if args.lr else config['client_config']['local_step_size']
    config['client_config']['alpha'] = float(args.alpa_feddyn) if args.alpa_feddyn else config['client_config'][
        'alpha']
    config['client_config']['asofed_beta'] = float(args.asofed_beta) if args.asofed_beta else config['client_config']['asofed_beta']
    config['client_config']['index'] = int(args.index) if args.index else None
    config['server_config']['rounds'] = int(args.rounds) if args.rounds else config['server_config']['rounds']
    config['server_config']['max_time']= float(args.max_time) if args.max_time else config['server_config']['max_time']
    config['server_config']['max_accuracy']= float(args.max_accuracy) if args.max_accuracy else config['server_config']['max_accuracy']
    config['server_config']['client_round'] = int(args.num_clients) if args.num_clients else config['server_config'][
        'client_round']
    config['server_config']['global_step_size'] = float(args.global_lr) if args.global_lr else config['server_config'][
        'global_step_size']
    config['server_config']['min_client_to_start'] = int(args.min_clients) if args.min_clients else config[
        'server_config']['min_client_to_start']
    config['server_config']['max_time'] = int(args.max_time) if args.max_time else config['server_config']['max_time']
    config['server_config']['fedAsync_variant'] = args.fedAsync_variant if args.fedAsync_variant else config[
        'server_config']['fedAsync_variant']

    config['dyn_sampling'] = str_to_bool(args.dyn_sampling) if args.dyn_sampling else config['dyn_sampling']
    config['decay_rate'] = float(args.decay_rate) if args.decay_rate else config['decay_rate']
    config['client_selection'] = str_to_bool(args.client_selection) if args.client_selection else config[
        'client_selection']
    config['model_pruning'] = str_to_bool(args.model_pruning) if args.model_pruning else config['model_pruning']
    config['prune_interval'] = int(args.prune_interval) if args.prune_interval else config['prune_interval']
    config['prune_ratio'] = float(args.prune_ratio) if args.prune_ratio else config['prune_ratio']
    
    config['synchronicity'] = 1 if config['fl_method'] in ["FedAvg", "FedDyn", "SCAFFOLD", "FedAvgN"] else 0
    config['client_clustering'] = str_to_bool(args.client_clustering) if args.client_clustering else config['client_clustering']
    
    # variables consistency check
    if config['server_config']["min_client_to_start"] < config['server_config']['client_round']:
        raise Exception("The number of clients to start learning must be at least equal "
                        "to the number of clients per round ")
    
    # Configuration algorithm parameters
    config["server_config"]['unweighted_mu_0'] = float(args.unweighted_mu_0) if args.unweighted_mu_0 else config["server_config"]['unweighted_mu_0']
    config["server_config"]['fedAsync_a_hinge'] = float(args.fedAsync_a_hinge) if args.fedAsync_a_hinge else config["server_config"]['fedAsync_a_hinge']
    config["server_config"]['fedAsync_b_hinge'] = float(args.fedAsync_b_hinge) if args.fedAsync_b_hinge else config["server_config"]['fedAsync_b_hinge']
    config["server_config"]['fedAsync_a_poly'] = float(args.fedAsync_a_poly) if args.fedAsync_a_poly else config["server_config"]['fedAsync_a_poly']
    config["server_config"]['fedAsync_mu_0'] = float(args.fedAsync_mu_0) if args.fedAsync_mu_0 else config["server_config"]['fedAsync_mu_0']
    config["server_config"]['asyncFedED_lambda'] = float(args.asyncFedED_lambda) if args.asyncFedED_lambda else config["server_config"]['asyncFedED_lambda']
    config["server_config"]['asyncFedED_epsilon'] = float(args.asyncFedED_epsilon) if args.asyncFedED_epsilon else config["server_config"]['asyncFedED_epsilon']
    config["server_config"]['asyncFedED_gamma_bar'] = float(args.asyncFedED_gamma_bar) if args.asyncFedED_gamma_bar else config["server_config"]['asyncFedED_gamma_bar']
    config["server_config"]['asyncFedED_k'] = int(args.asyncFedED_k) if args.asyncFedED_k else config["server_config"]['asyncFedED_k']
    config["server_config"]['asofed_rho'] = float(args.asofed_rho) if args.asofed_rho else config["server_config"]['asofed_rho']
    config["server_config"]['asofed_global_lr'] = float(args.asofed_global_lr) if args.asofed_global_lr else config["server_config"]['asofed_global_lr']
    config["accuracy_file"] = args.accuracy_file if args.accuracy_file else config["accuracy_file"]
    config["loss_file"] = args.loss_file if args.loss_file else config["loss_file"]

    # print on log the info used
    logger.info(config)

    from Communication.communication import CommunicationManager
    manager = CommunicationManager(config)
    manager.run_instance()
    logger.debug("END of the program!")


def str_to_bool(value):
    if value.lower() in ('true', 'True', 'yes'):
        return True
    elif value.lower() in ('false', 'False', 'no'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


if __name__ == "__main__":
    main()
