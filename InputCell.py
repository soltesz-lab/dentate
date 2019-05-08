from abc import ABCMeta, abstractmethod
import numpy as np

def instantiate_place_cell(context, gid, module, nfields, this_width=None, **kwargs):
    if this_width is None:
        this_width = context.module_width
    cell_args = {}
    cell_field_width = []
    mod_jitter = kwargs.get('jitter', 0.0)
    for n in xrange(nfields):
        curr_width = this_width + context.local_random.uniform(-mod_jitter, mod_jitter)
        cell_field_width.append(curr_width)
    cell_args['Nx'] = context.nx
    cell_args['Ny'] = context.ny
    cell_args['Field Width'] = cell_field_width
    place_cell = PlaceCell(gid, nfields=nfields, module=module, **cell_args)
    return place_cell

def instantiate_grid_cell(context, gid, module, nfields):
    orientation = context.grid_orientation[module]
    spacing = context.module_width

    delta_spacing     = context.local_random.uniform(-10., 10.)
    delta_orientation = context.local_random.uniform(-10., 10.)

    cell_args = {}
    cell_args['Grid Spacing']     = np.array([spacing + delta_spacing], dtype='float32')
    cell_args['Grid Orientation'] = np.array([orientation + delta_orientation], dtype='float32')
    cell_args['Nx'] = context.nx
    cell_args['Ny'] = context.ny

    grid_cell = GridCell(gid, nfields=nfields, module=module, **cell_args)
    return grid_cell

def acquire_fields_per_cell(ncells, field_probabilities, generator):
    field_probabilities = np.asarray(field_probabilities, dtype='float32')
    field_set = [i for i in range(field_probabilities.shape[0])]
    return generator.choice(field_set, p=field_probabilities, size=(ncells,))
    
    

class InputCell(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, gid, nfields=[], module=[], **kwargs):
        self.gid = gid
        self.nfields = nfields
        self.module = module

        self.x_offset = []
        self.y_offset = []
        self.cell_type = []
        self.rate_map  = []
        self.nx = kwargs.get('Nx', [])
        self.ny = kwargs.get('Ny', [])

    @abstractmethod
    def return_attr_dict(self):
        cell = {}
        cell['gid'] = np.array([self.gid], dtype='int32')
        cell['Num Fields'] = np.array([self.nfields], dtype='uint8')
        cell['Module'] = np.array([self.module], dtype='uint8')
        cell['X Offset'] = np.array(self.x_offset, dtype='float32')
        cell['Y Offset'] = np.array(self.y_offset, dtype='float32')
        cell['Nx'] = np.array([self.nx], dtype='int32')
        cell['Ny'] = np.array([self.ny], dtype='int32')
        cell['Rate Map'] = np.array(self.rate_map).reshape(-1,).astype('float32')
        return cell

    @abstractmethod
    def generate_spatial_ratemap(self, interp_x, interp_y, **kwargs):
        pass



class GridCell(InputCell):
    def __init__(self, gid, nfields=[], module=[], **kwargs):
        super(GridCell, self).__init__(gid, nfields=nfields, module=module, **kwargs)
        self.cell_type = [0]
        self.grid_spacing = kwargs.get('Grid Spacing', [])
        self.grid_orientation = kwargs.get('Grid Orientation', [])
        
    def return_attr_dict(self):
        cell = super(GridCell, self).return_attr_dict()
        cell['Cell Type'] = np.array([self.cell_type], dtype='uint8')
        cell['Grid Spacing'] = np.array(self.grid_spacing, dtype='float32')
        cell['Grid Orientation'] = np.array(self.grid_orientation, dtype='float32')
        return cell

    def generate_spatial_ratemap(self, interp_x, interp_y, **kwargs):

        a = kwargs['a']
        b = kwargs['b']
        peak_rate = kwargs['grid_peak_rate']
        x_offset = self.x_offset
        y_offset = self.y_offset
        grid_orientation = self.grid_orientation
        grid_spacing = self.grid_spacing
        theta_k   = [np.deg2rad(-30.), np.deg2rad(30.), np.deg2rad(90.)]
        inner_sum = np.zeros_like(interp_x)
        for theta in theta_k:
            inner_sum += np.cos( ((4. * np.pi) / (np.sqrt(3.) * grid_spacing)) * \
                         (np.cos(theta - grid_orientation) * (interp_x - x_offset[0]) \
                          + np.sin(theta - grid_orientation) * (interp_y - y_offset[0])))
        transfer = lambda z: np.exp(a * (z - b)) - 1.
        max_rate = transfer(3.)
        rate_map = peak_rate * transfer(inner_sum) / max_rate
        self.rate_map = rate_map
        
        return rate_map
    
class PlaceCell(InputCell):
    def __init__(self, gid, nfields=[], module=[], **kwargs):
        super(PlaceCell, self).__init__(gid, nfields=nfields, module=module, **kwargs)
        self.cell_type = [1]
        self.field_width = kwargs.get('Field Width', [])
        self.num_fields = kwargs.get('Num Fields', None)
        
    def return_attr_dict(self):
        cell = super(PlaceCell, self).return_attr_dict()
        cell['Cell Type'] = np.array([self.cell_type], dtype='uint8')
        cell['Field Width'] = np.array(self.field_width, dtype='float32')
        return cell

    def generate_spatial_ratemap(self, interp_x, interp_y, **kwargs):

        if cell.num_fields > 0:
            peak_rate = kwargs['place_peak_rate']
            x_offset = self.x_offset
            y_offset = self.y_offset
            field_width = self.field_width
            nfields  = self.num_fields
            rate_map = np.zeros_like(interp_x)
            for n in xrange(nfields):
                current_map = peak_rate * np.exp(-((interp_x - x_offset[n]) / (field_width[n] / 3. / np.sqrt(2.))) ** 2.) * np.exp(-((interp_y  - y_offset[n]) / (field_width[n] / 3. / np.sqrt(2.))) ** 2.)
                rate_map    = np.maximum(current_map, rate_map)
            rate_map.reshape(-1,).astype('float32')
        else:
            rate_map = np.zeros( (self.nx[0] * self.ny[0],) ).astype('float32')

        self.rate_map = rate_map
        return rate_map

