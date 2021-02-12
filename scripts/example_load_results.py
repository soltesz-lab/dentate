import pprint
import h5py

input_file = 'network_clamp.optimize.h5'
opt_id = 'network_clamp.optimize'

f = h5py.File(input_file, 'r')
opt_grp = f[opt_id]

parameter_enum_dict = h5py.check_enum_dtype(opt_grp['parameter_enum'].dtype)
parameters_idx_dict = { parm: idx for parm, idx in parameter_enum_dict.items() }
parameters_name_dict = { idx: parm for parm, idx in parameters_idx_dict.items() }
    
problem_parameters = { parameters_name_dict[idx]: val
                       for idx, val in opt_grp['problem_parameters'] }
parameter_specs = [ (parameters_name_dict[spec[0]], tuple(spec)[1:])
                    for spec in iter(opt_grp['parameter_spec']) ]

problem_ids = None
if 'problem_ids' in opt_grp:
    problem_ids = set(opt_grp['problem_ids'])
    
M = len(parameter_specs)
raw_results = {}
if problem_ids is not None:
    for problem_id in problem_ids:
        data = opt_grp['%d' % 0]['results'][:].reshape((-1,M+1)) # np.array of shape [N, M+1]
        raw_results[problem_id] = {'x': data[:,1:], 'y': data[:,0],}
else:
    data = opt_grp['%d' % 0]['results'][:].reshape((-1,M+1)) # np.array of shape [N, M+1]
    raw_results[0] = {'x': data[:,1:], 'y': data[:,0],}
f.close()

pprint.pprint(raw_results)

