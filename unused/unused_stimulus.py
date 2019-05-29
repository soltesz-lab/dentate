def acquire_fields_per_cell(ncells, field_probabilities, generator):
    field_probabilities = np.asarray(field_probabilities, dtype='float32')
    field_set = [i for i in range(field_probabilities.shape[0])]
    return generator.choice(field_set, p=field_probabilities, size=(ncells,))


def get_rate_maps(cells):
    rate_maps = []
    for gid in cells:
        cell = cells[gid]
        nx, ny = cell.nx, cell.ny
        rate_map = cell.rate_map.reshape(nx, ny)
        rate_maps.append(rate_map)
    return np.asarray(rate_maps, dtype='float32')


def fraction_active(cells, threshold):
    temp_cell = cells.values()[0]
    nx, ny = temp_cell.nx, temp_cell.ny
    del temp_cell

    rate_maps = get_rate_maps(cells)
    nxx, nyy = np.meshgrid(np.arange(nx), np.arange(ny))
    coords = zip(nxx.reshape(-1, ), nyy.reshape(-1, ))

    factive = lambda px, py: calculate_fraction_active(rate_maps[:, px, py], threshold)
    return {(px, py): factive(px, py) for (px, py) in coords}


def coefficient_of_variation(cells, eps=1.0e-6):
    rate_maps = get_rate_maps(cells)
    summed_map = np.sum(rate_maps, axis=0)

    mean = np.mean(summed_map)
    std = np.std(summed_map)
    cov = np.divide(std, mean + eps)
    return cov


def peak_to_trough(cells):
    rate_maps = get_rate_maps(cells)
    summed_map = np.sum(rate_maps, axis=0)
    var_map = np.var(rate_maps, axis=0)
    minmax_eval = 0.0
    var_eval = 0.0

    return minmax_eval, var_eval


def calculate_field_distribution(pi, pr):
    p1 = (1. - pi) / (1. + (7. / 4.) * pr)
    p2 = p1 * pr
    p3 = 0.5 * p2
    p4 = 0.5 * p3
    probabilities = np.array([pi, p1, p2, p3, p4], dtype='float32')
    assert (np.abs(np.sum(probabilities) - 1.) < 1.e-5)
    return probabilities


def gid2module_dictionary(cell_lst, modules):
    module_dict = {module: {} for module in modules}
    for cells in cell_lst:
        for (gid, cell_dict) in cells:
            feature_type = cell_dict['Cell Type'][0]
            cell = make_input_cell(gid, feature_type, cell_dict)
            this_module = cell.module
            module_dict[this_module][gid] = cell
    return module_dict


def module2gid_dictionary(module_dict):
    gid_dict = dict()
    for module in module_dict:
        gid_dict.update(module_dict[module])
    return gid_dict


def calculate_fraction_active(rates, threshold):
    N = len(rates)
    num_active = len(np.where(rates > threshold)[0])
    fraction_active = np.divide(float(num_active), float(N))
    return fraction_active