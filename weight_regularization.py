"""
Contains code for group weight orthogonalization via regularizations. Both inter-group and intra-group.
"""
import torch
import torch.nn as nn


def get_layers_to_regularize(model: nn.Module, input_shape, regularize_all=False):
    # TODO: BUG. The following method depends on the order of calls in the block forward pass. if the residual
    #  conv layer doesn't have norm after (like in wide resnet) and is called between the main block's convs then
    #  some conv layers will be missing from the list although they are followed by norm
    ordered_named_modules = []  # list of tuples
    hook_handles = []

    def add_hook(name):
        def forward_hook(module, input, output):
            if len(list(module.children())) == 0:
                ordered_named_modules.append((name, module))

        return forward_hook

    for k,v in model.named_modules():
        if len(list(v.children())) == 0:
            # Only add basic nn modules that don't contain sub-modules (e.g. conv2d/fc/avgpool)
            hook_handle = v.register_forward_hook(add_hook(k))
            hook_handles.append(hook_handle)

    # Run single input through the model to fill the module list in the order of execution
    model.cuda()
    with torch.no_grad():
        x = torch.randn(1, input_shape[0], input_shape[1], input_shape[2]).cuda()

        _ = model(x)

    # Remove hooks
    for handle in hook_handles:
        handle.remove()

    # Iterate over modules, and look for conv layers followed by group norm
    out_dict = {}
    for ii in range(len(ordered_named_modules) - 1):
        current_module_name, current_module = ordered_named_modules[ii]
        next_module_name, next_module = ordered_named_modules[ii + 1]
        if isinstance(current_module, nn.Conv2d) and isinstance(next_module, nn.GroupNorm):
            # TODO : in case of skip connections the order might not represent the actual data flow.
            #   ordered_named_modules depends on where where the skip connection conv as called
            #   (before/after/in the middle of the block).
            num_groups = next_module.num_groups
            out_dict[current_module_name] = {'num_groups': num_groups, 'c_out': current_module.weight.shape[0]}
        elif regularize_all and hasattr(current_module, 'weight') and current_module.weight.ndimension() >= 2:
            num_features = current_module.weight.shape[0]
            out_dict[current_module_name] = {'num_groups': min(32, num_features // 4), 'c_out': num_features}

    if regularize_all:
        last_module_name, last_module = ordered_named_modules[-1]
        if hasattr(last_module, 'weight') and last_module.weight.ndimension() >= 2:
            num_features = last_module.weight.shape[0]
            out_dict[last_module_name] = {'num_groups': min(32, num_features // 4), 'c_out': num_features}

    return out_dict


def selective_orth_dist(w):
    """
    Taken from https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks/blob/aa7f56901c661a124e0cfe72eb2c9dc98045ce94/imagenet/utils.py#L42
    Selective Double Soft Orthogonality Regularization
    """
    mat = w.reshape(w.shape[0], -1)
    if mat.shape[0] < mat.shape[1]:
        mat = mat.T
    return torch.norm(mat.T @ mat - torch.eye(mat.shape[1]).cuda()) ** 2


def orth_dist(w):
    """
    Taken from https://github.com/samaonline/Orthogonal-Convolutional-Neural-Networks/blob/aa7f56901c661a124e0cfe72eb2c9dc98045ce94/imagenet/utils.py#L42
    Soft Orthogonality Regularization
    """
    mat = w.reshape(w.shape[0], -1)

    return torch.norm(mat @ mat.T - torch.eye(mat.shape[0]).cuda()) ** 2


def remove_prefix(k):
    if k.startswith('module'):
        # Data parallel case
        k = k[len('module.'):]

    return k


def randomize_filter(W, layer_dict, randomize_mode):
    if randomize_mode == 'constant':
        assert 'perm' in layer_dict.keys()
        # Use pre-calculated permutation to shuffle the filter order - filters will not be normalized
        # according to the GN groups
        current_weight_filter_permutation = layer_dict['perm']
        W = W[current_weight_filter_permutation]
    elif randomize_mode == 'iter':
        # Randomly permute the output filters - this way filters will not be normalized according
        # to the GN groups
        W = W[torch.randperm(W.shape[0])]
    else:
        raise Exception(f'Unsupported randomize mode {randomize_mode}')

    return W


def group_reg_ortho_l2(model: nn.Module, reg_type: str, layers_dict: dict, randomize_mode: str = None):
    assert reg_type in ['inter', 'intra']

    total_reg_value = 0

    for k, v in model.named_modules():
        k = remove_prefix(k)
        if k in layers_dict.keys():
            W = v.weight
            num_groups = layers_dict[k]['num_groups']
            c_out = W.shape[0]
            group_size = c_out // num_groups

            if randomize_mode and isinstance(v, nn.Conv2d):  # Only shuffle conv layers weights
                W = randomize_filter(W, layers_dict[k], randomize_mode)

            if reg_type == 'intra':
                for ii in range(group_size):
                    # Create ortho matrix
                    w = W[ii::group_size]  # num_groups x cin x h x w
                    dist = orth_dist(w)

                    total_reg_value += dist
            elif reg_type == 'inter':
                # Inter-group orthogonalization
                for ii in range(num_groups):
                    # Create ortho matrix
                    w = W[ii * group_size: (ii + 1) * group_size]  # group_size x cin  x h w
                    dist = orth_dist(w)

                    total_reg_value += dist
            else:
                raise Exception(f'Unsupported mode {reg_type}')

    return total_reg_value


def l2_reg_ortho(mdl):
    """Function used for Orthogonal Regularization reg = ||W W_t - I ||^2"""
    l2_reg = 0
    for W in mdl.parameters():
        if W.ndimension() < 2:
            continue
        else:
            l2_reg += orth_dist(W)

    return l2_reg


def weights_reg(mdl, reg_type, weight_groups_dict=None, randomize_mode=None):
    if reg_type == 'SO':
        return l2_reg_ortho(mdl)
    elif reg_type in ['GSO_intra', 'GSO_inter']:
        return group_reg_ortho_l2(mdl, reg_type[len('GSO_'):], weight_groups_dict, randomize_mode=randomize_mode)
    else:
        return 0
