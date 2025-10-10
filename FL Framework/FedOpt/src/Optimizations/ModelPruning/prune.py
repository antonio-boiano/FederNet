from FedOpt.src.Communication.communication import json_to_tensor_list, json_to_state_dict
from FedOpt.src.Optimizations.ModelPruning.prune_source import *
import logging

logger = logging.getLogger("FedOpt")

class ModelPruning:
    def __init__(self, pruning_rate, model):
        # use model to analyze number of conv layer
        self.prune_layers = [] # Stores names of convolutional layers
        self.prune_channels = [] # Stores number of output channels per layer
        count = 1
        for i, layer in enumerate(model.features):
            if isinstance(layer, torch.nn.Conv2d):
                self.prune_layers.append("conv" + str(count))
                count += 1
                self.prune_channels.append(layer.out_channels)
        logger.debug("Convolutional layers founded:")
        logger.debug(f"     {self.prune_layers}")
        logger.debug(f"     {self.prune_channels}")
        self.pruning_rate = pruning_rate

    def client_fed_prune(self, model_manager):
        """
        Return the sum of the kernel weights for each layer to be pruned
        """
        network = model_manager.model # get the model
        # modify prune_step for federated application (always in independent way)
        logger.info("START Client Pruning")
        conv_count = 1  # conv count for 'indexing_prune_layers'
        dim = 0  # prune corresponding dim of filter weight [out_ch, in_ch, k1, k2] = [num output channels, num input channels, kernel size] -> if dim == 0, i prune over the dimension 0 (out_ch), otherwise over the dimension 1 (in_ch)
        array_sum_of_kernel = []
        for i in range(len(network.features)):
            if isinstance(network.features[i], torch.nn.Conv2d):
                if dim == 1:
                    dim ^= 1 # xor operation

                if 'conv%d' % conv_count in self.prune_layers:
                    kernel = network.features[i].weight.data
                    sum_of_kernel = torch.sum(torch.abs(kernel.view(kernel.size(0), -1)), dim=1)
                    array_sum_of_kernel.append(sum_of_kernel)
                    dim ^= 1

                conv_count += 1
        return array_sum_of_kernel

    def server_fed_prune(self, fed_manager, client_responses):
        """Performs federated pruning on the server side by aggregating client pruning responses and updating the server model.

        Args:
            fed_manager: The federated learning manager containing the server model and training configuration
            client_responses: Dictionary containing pruning responses from clients, with client IDs as keys

        Returns:
            channel_indexes_pruned: List of indices representing the channels that were pruned from the model

        The function:
        1. Converts client responses from JSON to tensor format
        2. Groups corresponding tensors from all clients
        3. Calculates mean tensor values across clients for each layer
        4. Applies pruning to the network based on aggregated values
        5. Updates algorithm-specific variables (FedDyn's h parameter or SCAFFOLD's control variates)"""

        all_list_of_tensors = []
        # Convert json to list of tensor
        for key in client_responses: # key is the client ID
            all_list_of_tensors.append(json_to_tensor_list(client_responses[key]))

        num_of_tensors = len(all_list_of_tensors[0]) # num_of_tensors = number of layers for each client
        if num_of_tensors != len(self.prune_layers):
            logger.error("Some tensors data is missing!")

        # Group tensors by position: groups_of_tensors[0] -> all tensors at position 0, one tensor for each client that responded (then I will calculate the mean value for each layer) 
        groups_of_tensors = [[] for _ in range(num_of_tensors)] # num_of_tensors = number of layers
        for list_of_tensor in all_list_of_tensors:
            for i in range(len(list_of_tensor)):
                groups_of_tensors[i].append(list_of_tensor[i])

        # Mean of tensors foreach group
        result = []
        for idx, group in enumerate(groups_of_tensors):
            # print(f"Tensors at position {idx}: {group}")
            result.append(torch.mean(torch.stack(group, dim=0).float(), dim=0)) # calculate the mean value for each layer across all clients

        network = fed_manager.server_model.model
        channel_indexes_pruned = self.new_network(network, result) # new_network applies the pruning to the network
        # check also parameter to be updated
        if fed_manager.server_model.name == "FedDyn":
            logger.info("Updating global variables of FedDyn due to pruning")
            fed_manager.h = {
                key: torch.zeros(params.shape, device=fed_manager.device)
                for key, params in fed_manager.server_model.model.state_dict().items()
            }
            #reduce_h(fed_manager.h, channel_indexes_pruned)
        elif fed_manager.server_model.name == "SCAFFOLD":
            logger.info("Updating global variables of SCAFFOLD due to pruning")
            fed_manager.server_c = reduce_control_variates(fed_manager.server_c, channel_indexes_pruned)
        return channel_indexes_pruned

    def new_network(self, network, tensors_mean):
        # Save the original device where the model parameters are located (e.g., CPU or CUDA)
        original_device = next(network.parameters()).device
        # Move the model to CPU for pruning since some operations may be easier or require CPU memory
        network = network.cpu()
        channel_indexes_pruned = []
        # update self.prune_channels
        # Calculate the number of channels to prune by rounding the channels to prune based on the pruning rate
        to_prune_channels = [int(round(channel * self.pruning_rate)) for channel in self.prune_channels]
        self.prune_channels = [a - b for a, b in zip(self.prune_channels, to_prune_channels)]
        logger.info(f"Number of filter that will be pruned during this phase: {to_prune_channels}")

        count = 0  # count for indexing 'prune_channels'
        conv_count = 1  # conv count for 'indexing_prune_layers'
        dim = 0  # 0: prune corresponding dim of filter weight [out_ch, in_ch, k1, k2]

        for i in range(len(network.features)):
            if isinstance(network.features[i], torch.nn.Conv2d):
                if dim == 1:
                    new, _ = get_new_conv(network.features[i], dim, channel_index, False)
                    network.features[i] = new
                    dim ^= 1

                if 'conv%d' % conv_count in self.prune_layers:
                    # adaptation of get_channel_index from original prune
                    sum_of_kernel = tensors_mean[count]
                    # vals is sum_of_kernel sorted, args are the selected index
                    # Sort the absolute values of kernel weights to identify the least important channels
                    vals, args = torch.sort(sum_of_kernel)
                    # Get the indices of the channels to prune (i.e., the smallest values)
                    channel_index = args[:to_prune_channels[count]].tolist()
                    channel_indexes_pruned.append(channel_index)
                    # Get the new Conv2d layer after pruning the selected channels
                    new_ = get_new_conv(network.features[i], dim, channel_index, False)
                    network.features[i] = new_
                    dim ^= 1 # Flip the `dim` value for future layers (to prune input or output channels)
                    count += 1 # Move to the next layer to prune
                conv_count += 1 # Move to the next convolutional layer

            # If the current layer is a BatchNorm2d and `dim == 1`, prune the BatchNorm layer as well
            elif dim == 1 and isinstance(network.features[i], torch.nn.BatchNorm2d):
                new_ = get_new_norm(network.features[i], channel_index)
                network.features[i] = new_

        # update to check last conv layer pruned
        if 'conv%d' % count in self.prune_layers:
            # Update the first linear layer (fully connected) after pruning to adjust to the new input dimensions
            logger.debug("Pruning - updating the classification input")
            channel_indexes_pruned.append(update_first_linear_layer(network, channel_index))

        # logger.debug_layers_info(network)
        if original_device.type == 'cuda':
            network.to(original_device)
        return channel_indexes_pruned

def print_layers_info(network):
    for idx, layer in enumerate(network.modules()):
        if isinstance(layer, torch.nn.Conv2d):
            logger.debug(f"Layer {idx}: {layer}")
            logger.debug(f"  In Channels: {layer.in_channels}")
            logger.debug(f"  Out Channels: {layer.out_channels}")
            logger.debug(f"  Kernel Size: {layer.kernel_size}")
            logger.debug(f"  Stride: {layer.stride}")
            logger.debug(f"  Padding: {layer.padding}")
            logger.debug(f"  Dilation: {layer.dilation}")


def update_first_linear_layer(network, channel_index):
    for idx, layer in enumerate(network.classifier):
        if isinstance(layer, torch.nn.Linear):
            network.classifier[idx],removed_index = get_new_linear_spatial(layer, channel_index, network.spatial)
            return removed_index


# Added to update bias of first classification layer
def get_new_linear_spatial(linear, channel_index, spatial_dim):
    new_channels = int((linear.in_features / spatial_dim) - len(channel_index))
    new_in_features = new_channels * spatial_dim

    new_linear = torch.nn.Linear(in_features=new_in_features,
                                 out_features=linear.out_features,
                                 bias=linear.bias is not None)

    # Adapt old weights to the new tensor
    with torch.no_grad():
        old_weights = linear.weight.data
        old_bias = linear.bias.data

        # Index of rows to be removed from tensor
        indices_to_remove = get_indices_to_remove(channel_index, spatial_dim)
        indices_to_remove_tensor = torch.tensor(indices_to_remove)

        mask = torch.ones(old_weights.size(1), dtype=torch.bool)
        mask[indices_to_remove_tensor] = False

        # Check for some error in the mask
        if mask.all():
            raise ValueError("All the mask is true, tensor is not reduced")

        # Apply mask to the original tensor
        reduced_tensor = old_weights[:, mask]

        new_linear.weight.data = reduced_tensor
        new_linear.bias.data = old_bias

    return new_linear,indices_to_remove


def get_indices_to_remove(removed_channels, spatial_dim):
    indices_to_remove = []
    for channel in removed_channels:
        indices_to_remove.extend(range(channel * spatial_dim, (channel + 1) * spatial_dim))
    return indices_to_remove


def check_model_size(local_model, msg, device):
    loaded_state_dict = json_to_state_dict(msg["server_model"],device)
    local_state_dict = local_model.state_dict()
    for key in loaded_state_dict.keys():
        if key in local_state_dict:
            if local_state_dict[key].shape != loaded_state_dict[key].shape:
                return False

    return True

def update_client_variables(model_manager,channel_indexes_pruned,old_network=None):
    # check parameter to be updated
    if model_manager.name == "FedDyn":
        logger.info("Updating local variables of FedDyn due to pruning")
        model_manager.prev_grads = reduce_prev_grads(model_manager.prev_grads, channel_indexes_pruned, old_network)
    elif model_manager.name == "SCAFFOLD":
        logger.info("Updating local variables of SCAFFOLD due to pruning")
        model_manager.server_c = reduce_control_variates(model_manager.server_c, channel_indexes_pruned)
        model_manager.client_c = reduce_control_variates(model_manager.client_c, channel_indexes_pruned)


def reduce_prev_grads(prev_grads, arrays_indexes_removed, model):
    new_prev_grads = []
    array_index = 0
    grad_index = 0  # tensor index to segment

    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            weight_size = layer.weight.numel()
            weight_tensor = prev_grads[grad_index:grad_index + weight_size].view_as(layer.weight)
            grad_index += weight_size

            if array_index == 0:
                mask_out = torch.ones(weight_tensor.size(0), dtype=torch.bool)
                mask_out[arrays_indexes_removed[0]] = False
                reduced_weight = weight_tensor[mask_out, :, :, :]
            else:
                mask_out = torch.ones(weight_tensor.size(0), dtype=torch.bool)
                mask_out[arrays_indexes_removed[array_index]] = False

                mask_in = torch.ones(weight_tensor.size(1), dtype=torch.bool)
                mask_in[arrays_indexes_removed[array_index - 1]] = False
                reduced_weight = weight_tensor[mask_out, :][:, mask_in, :, :]

            new_prev_grads.append(reduced_weight.flatten())

            if layer.bias is not None:
                bias_size = layer.bias.numel()
                bias_tensor = prev_grads[grad_index:grad_index + bias_size]
                grad_index += bias_size
                mask = torch.ones(bias_tensor.size(0), dtype=torch.bool)
                mask[arrays_indexes_removed[array_index]] = False
                reduced_bias = bias_tensor[mask]
                new_prev_grads.append(reduced_bias.flatten())

                array_index += 1
            else:
                logger.warning("Bias for convolutional layer is missing")

        elif isinstance(layer, torch.nn.Linear):
            weight_size = layer.weight.numel()
            weight_tensor = prev_grads[grad_index:grad_index + weight_size].view_as(layer.weight)
            grad_index += weight_size

            if array_index == len(arrays_indexes_removed) - 1:
                # Update input of first classification layer
                mask_in = torch.ones(weight_tensor.size(1), dtype=torch.bool)
                mask_in[arrays_indexes_removed[array_index]] = False
                reduced_weight = weight_tensor[:, mask_in]
                new_prev_grads.append(reduced_weight.flatten())
            else:
                new_prev_grads.append(weight_tensor.flatten())

            if layer.bias is not None:
                bias_size = layer.bias.numel()
                bias_tensor = prev_grads[grad_index:grad_index + bias_size]
                grad_index += bias_size
                new_prev_grads.append(bias_tensor.flatten())

            array_index += 1

    # merge list of tensor
    new_prev_grads = torch.cat(new_prev_grads)
    if grad_index == prev_grads.numel():
        logger.debug("Previous gradients resize DONE")

    return new_prev_grads


def reduce_control_variates(control_c, arrays_indexes_removed):
    updated_control_c = []
    reduction_index = 0
    i = 0
    while i < len(control_c):
        dim_tensor = control_c[i].dim()
        if dim_tensor == 4: # torch.nn.Conv2d
            weight_tensor = control_c[i]
            bias_tensor = control_c[i+1]

            if reduction_index == 0:
                mask_out = torch.ones(weight_tensor.size(0), dtype=torch.bool)
                mask_out[arrays_indexes_removed[0]] = False
                weight_tensor = weight_tensor[mask_out, :, :, :]
                updated_control_c.append(weight_tensor)
            else:
                mask_out = torch.ones(weight_tensor.size(0), dtype=torch.bool)
                mask_out[arrays_indexes_removed[reduction_index]] = False
                mask_in = torch.ones(weight_tensor.size(1), dtype=torch.bool)
                mask_in[arrays_indexes_removed[reduction_index - 1]] = False
                weight_tensor = weight_tensor[mask_out, :][:, mask_in, :, :]
                updated_control_c.append(weight_tensor)

            # update bias
            mask = torch.ones(bias_tensor.size(0), dtype=torch.bool)
            mask[arrays_indexes_removed[reduction_index]] = False
            bias_tensor = bias_tensor[mask]
            updated_control_c.append(bias_tensor)

            reduction_index += 1
            i += 2

        elif dim_tensor == 2: # torch.nn.Linear
            weight_tensor = control_c[i]
            bias_tensor = control_c[i + 1]

            if reduction_index == (len(arrays_indexes_removed) - 1):
                mask_in = torch.ones(weight_tensor.size(1), dtype=torch.bool)
                mask_in[arrays_indexes_removed[reduction_index]] = False
                weight_tensor = weight_tensor[:, mask_in]
                updated_control_c.append(weight_tensor)
                updated_control_c.append(bias_tensor)
                reduction_index+=1
            else: # unchanged linear
                updated_control_c.append(weight_tensor)
                updated_control_c.append(bias_tensor)
            i += 2
        else:
            updated_control_c.append(control_c[i])
            i += 1
    return updated_control_c


def reduce_h(h, arrays_indexes_removed):
    array_index = 0

    for key, value in h.items():
        if "features" in key: # convolutional layer
            if "weight" in key:
                weight_tensor = h[key]
                if array_index == 0:
                    # first layer, reduce only out_channels
                    mask_out = torch.ones(weight_tensor.size(0), dtype=torch.bool)
                    mask_out[arrays_indexes_removed[0]] = False
                    h[key] = weight_tensor[mask_out, :, :, :]
                else:
                    # reduce out_channels e in_channels
                    mask_out = torch.ones(weight_tensor.size(0), dtype=torch.bool)
                    mask_out[arrays_indexes_removed[array_index]] = False

                    mask_in = torch.ones(weight_tensor.size(1), dtype=torch.bool)
                    mask_in[arrays_indexes_removed[array_index - 1]] = False
                    h[key] = weight_tensor[mask_out, :][:, mask_in, :, :]
            elif "bias" in key:
                bias_tensor = h[key]

                mask = torch.ones(bias_tensor.size(0), dtype=torch.bool)
                mask[arrays_indexes_removed[array_index]] = False

                h[key] = bias_tensor[mask]
                array_index += 1
            else:
                logger.warning("Found a conv layer not pruned correctly")
        elif "classifier" in key:
            if array_index == (len(arrays_indexes_removed) - 1):
                if "weight" in key:
                    weight_tensor = h[key]
                    mask_in = torch.ones(weight_tensor.size(1), dtype=torch.bool)
                    mask_in[arrays_indexes_removed[array_index]] = False
                    h[key] = weight_tensor[:, mask_in]
                    array_index += 1