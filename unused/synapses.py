
def interactive_callback_plasticity_fit(**kwargs):
    import matplotlib.pyplot as plt
    fig = plt.figure(constrained_layout=3)
    gs = fig.add_gridspec(4, 3)
    initial_weights = kwargs['initial_weights'] 
    modified_weights = kwargs['modified_weights']
    initial_ratemap = kwargs['initial_ratemap']
    modified_ratemap = kwargs['modified_ratemap']
    target_ratemap =  kwargs['target_ratemap']
    plasticity_kernel =  kwargs['plasticity_kernel']
    delta_weights =  kwargs['delta_weights']
    syn_ids = np.asarray(sorted(initial_weights.keys()))
    initial_weights_array = np.asarray([initial_weights[k] for k in syn_ids])
    hist_bins = np.linspace(np.min(initial_weights_array), np.max(initial_weights_array))
    initial_weights_hist, initial_weights_bin_edges = np.histogram(initial_weights_array, bins=hist_bins)
    modified_weights_array = np.asarray([modified_weights[k] for k in syn_ids])
    hist_bins = np.linspace(np.min(modified_weights_array), np.max(modified_weights_array))
    modified_weights_hist, modified_weights_bin_edges = np.histogram(modified_weights_array, bins=hist_bins)
    hist_bins = np.linspace(np.min(delta_weights), np.max(delta_weights))
    delta_weights_hist, delta_weights_bin_edges = np.histogram(delta_weights, bins=hist_bins)
    xlimits = (min(np.min(initial_weights_array), np.min(modified_weights_array)),
               max(np.max(initial_weights_array), np.max(modified_weights_array)))
    ax = fig.add_subplot(gs[0, :])
    ax.plot(initial_weights_bin_edges[:-1], initial_weights_hist, marker='.')
    ax.set(ylabel='number of synapses')
    ax.set_xlim(xlimits)
    ax.set_yscale('log')
    ax = fig.add_subplot(gs[1, :])
    ax.plot(modified_weights_bin_edges[:-1], modified_weights_hist, marker='.', color='r')
    ax.set(xlabel='synaptic weight', ylabel='number of synapses')
    ax.set_xlim(xlimits)
    ax.set_yscale('log')
    ax = fig.add_subplot(gs[2, :])
    ax.plot(delta_weights_bin_edges[:-1], delta_weights_hist, marker='.')
    ax.set_yscale('log')
    ax.set(xlabel='synaptic weight delta', ylabel='number of synapses')
    ax = fig.add_subplot(gs[3, 0])
    ax.set_title('Initial rate map')
    p = ax.imshow(initial_ratemap, origin='lower')
    fig.colorbar(p, ax=ax)
    ax = fig.add_subplot(gs[3, 1])
    ax.set_title('Modified rate map')
    p = ax.imshow(modified_ratemap, origin='lower')
    fig.colorbar(p, ax=ax)
    ax = fig.add_subplot(gs[3, 2])
    ax.set_title('Target')
    p = ax.imshow(target_ratemap, origin='lower')
    fig.colorbar(p, ax=ax)
    plt.show()


def plasticity_fit(phi, plasticity_kernel, plasticity_inputs, source_syn_map, logger, max_iter=10, lb = -3.0, ub = 3.,
                   baseline_weight=0.0, local_random=None, interactive=False):
    
    source_gids = sorted(plasticity_inputs.keys())
    initial_weights = []
    for i, source_gid in enumerate(source_gids):
        _, initial_weight = source_syn_map[source_gid][0]
        initial_weights.append(initial_weight)
    w = np.asarray(initial_weights, dtype=np.float64)
    baseline = np.mean(w)
    A = np.column_stack([plasticity_inputs[gid].reshape((-1,)).astype(np.float64) for gid in source_gids])
    initial_ratemap = phi(np.dot(A, w))
    b = plasticity_kernel.reshape((-1,)).astype(np.float64)

    if local_random is None:
        local_random = np.random.RandomState()
    
    #res = opt.lsq_linear(A, b, bounds=(lb, ub), lsmr_tol='auto')
    def residual(x, w, A, b, phi):
        res = np.subtract(phi(np.dot(A, np.add(w, x))), b)
        return res

    for i in range(max_iter):
        try:
            dx0 = local_random.rand(len(w),)
            optres = opt.least_squares(residual, dx0, args = (w, A, b, phi), bounds=(lb,ub), 
                                       method='dogbox', xtol=1e-2, ftol=1e-2, gtol=1e-4,
                                       verbose=2 if interactive else 0)
            break
        except ValueError:
            continue
    if i >= max_iter:
        raise RuntimeError('plasticity_fit: maximum number of iterations exceeded')

    
    delta_weights = optres.x
    logger.info('Least squares fit status: %d (%s)' % (optres.status, optres.message))
    logger.info('Delta weights: min: %f max: %f' % (np.min(delta_weights), np.max(delta_weights)))
        
    syn_weights = {}
    for source_gid, delta_weight in zip(source_gids, delta_weights):
        syn_count = len(source_syn_map[source_gid])
        if syn_count > 0:
            for syn_id, initial_weight in source_syn_map[source_gid]:
                syn_weights[syn_id] = max(delta_weight + initial_weight, max(math.sqrt(initial_weight),  baseline_weight))

    if interactive:
        modified_weights = np.maximum(np.add(w, delta_weights), np.sqrt(w)) + baseline_weight
        modified_ratemap = phi(np.dot(A, modified_weights))
        logger.info('Initial rate map: min: %f max: %f' % (np.min(initial_ratemap), np.max(initial_ratemap)))
        logger.info('Target: min: %f max: %f' % (np.min(b), np.max(b)))
        logger.info('Initial weights: min: %f max: %f mean: %f' % (np.min(w), np.max(w), np.mean(w)))
        logger.info('Modified weights: min: %f max: %f mean: %f' % (np.min(modified_weights), np.max(modified_weights), np.mean(modified_weights)))
        logger.info('Modified rate map: min: %f max: %f' % (np.min(modified_ratemap), np.max(modified_ratemap)))
        initial_weights_dict = {}
        for i, source_gid in enumerate(source_gids):
            syn_id, initial_weight = source_syn_map[source_gid][0]
            initial_weights_dict[syn_id] = initial_weight
        interactive_callback_plasticity_fit(initial_weights=initial_weights_dict,
                                            modified_weights=syn_weights,
                                            delta_weights=delta_weights,
                                            initial_ratemap=initial_ratemap.reshape(plasticity_kernel.shape),
                                            modified_ratemap=modified_ratemap.reshape(plasticity_kernel.shape),
                                            plasticity_kernel=plasticity_kernel,
                                            target_ratemap=b.reshape(plasticity_kernel.shape))
        

    return syn_weights


def linear_phi(a):
    return a


def generate_structured_weights(gid, population, synapse_name, sources, dst_input_features, src_input_features,
                                src_syn_dict, spatial_mesh, plasticity_kernel=None, baseline_weight=0.0,
                                field_width_scale=1.0, local_random=None, interactive=False, max_iter=10):
    """

    :param gid:
    :param population:
    :param synapse_name:
    :param sources:
    :param dst_input_features:
    :param src_input_features:
    :param src_syn_dict:
    :param spatial_mesh:
    :param plasticity_kernel:
    :param baseline_weight:
    :param field_width_scale:
    :param local_random:
    :param interactive:
    :param max_iter:
    :return:
    """
    if gid in dst_input_features[population]:
        this_input_features = dst_input_features[population][gid]
        this_num_fields = this_input_features['Num Fields'][0]
        this_field_width = this_input_features['Field Width']
        this_x_offset = this_input_features['X Offset']
        this_y_offset = this_input_features['Y Offset']
        this_peak_rate = this_input_features['Peak Rate']
    else:
        this_input_features = None
        this_num_fields = None
        this_field_width = None
        this_x_offset = None
        this_y_offset = None
        this_peak_rate = None

    structured = False
    if (this_input_features is not None) and this_num_fields > 0:
        structured = True

    if local_random is None:
        local_random = np.random.RandomState()
        local_random.seed(int(gid))

    result = None
    if structured:

        if plasticity_kernel is None:
            plasticity_kernel = lambda x, y, x_loc, y_loc, sx, sy: gauss2d(x-x_loc, y-y_loc, sx=sx, sy=sy)
            plasticity_kernel = np.vectorize(plasticity_kernel, excluded=[2,3,4,5])

        x, y = spatial_mesh
        this_peak_locs = zip(np.nditer(this_x_offset), np.nditer(this_y_offset))
        this_sigmas    = [(width * field_width_scale) / 3. / np.sqrt(2.) for width in np.nditer(this_field_width)] # cm
        this_plasticity_kernel = reduce(np.add, [plasticity_kernel(x, y, peak_loc[0], peak_loc[1], sigma, sigma)
                                        for peak_loc, sigma in zip(this_peak_locs, this_sigmas)]) * this_peak_rate

        this_plasticity_inputs = {}
        src_gid_syn_dict = {}
        for source in sources:
            for src_gid in src_input_features[source]:
                src_gid_syn_idwgts = src_syn_dict[source][src_gid]
                src_gid_syn_dict[src_gid] = src_gid_syn_idwgts
                src_gid_input_features = src_input_features[source][src_gid]
                src_peak_rate = src_gid_input_features['Peak Rate']
                src_rate_map = src_gid_input_features['Arena Rate Map']
                norm_rate_map = src_rate_map / src_peak_rate
                this_plasticity_inputs[src_gid] = norm_rate_map

        plasticity_src_syn_dict = {}
        for source in sources:
            for src_gid in src_syn_dict[source]:
                plasticity_src_syn_dict[src_gid] = src_syn_dict[source][src_gid]
                        
        this_peak_locs = zip(np.nditer(this_x_offset), np.nditer(this_y_offset))
        logger.info('computing plasticity fit for gid %d: peak locs: %s field widths: %s' %
                        (gid, str([x for x in this_peak_locs]), str(this_field_width)))
        this_syn_weights = plasticity_fit(exp_phi, this_plasticity_kernel, this_plasticity_inputs, 
                                              plasticity_src_syn_dict, logger, baseline_weight=baseline_weight, 
                                              max_iter=max_iter, local_random=local_random, interactive=interactive)
        
        this_syn_ids = sorted(this_syn_weights.keys())
            
        result = {'syn_id': np.array(this_syn_ids).astype('uint32', copy=False),
                  synapse_name: np.array([this_syn_weights[syn_id] + baseline_weight
                                              for syn_id in this_syn_ids]).astype('float32', copy=False) }

    return result
