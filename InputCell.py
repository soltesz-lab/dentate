from abc import ABCMeta, abstractmethod
import numpy as np

#  custom data type for type of feature selectivity
selectivity_grid = 0
selectivity_place = 1



def make_place_cell(gid, module, nfields, this_width=None, **kwargs):
    if 'Field Width' in kwargs:
        place_cell = PlaceCell(gid, nfields=nfields, module=module, **kwargs)
    else:
        local_random = kwargs.get('local_random')
        mod_jitter = kwargs.get('jitter', 0.0)
        nx = kwargs.get('nx')
        ny = kwargs.get('ny')
        if this_width is None:
            this_width = kwargs.get('module_width')
        cell_args = kwargs
        cell_field_width = []
        for n in xrange(nfields):
            curr_width = this_width + local_random.uniform(-mod_jitter, mod_jitter)
            cell_field_width.append(curr_width)
        cell_args['Nx'] = nx
        cell_args['Ny'] = ny
        cell_args['Field Width'] = cell_field_width
        place_cell = PlaceCell(gid, nfields=nfields, module=module, **cell_args)
    return place_cell

def make_grid_cell(gid, module, nfields, **kwargs):

    if ('Grid Spacing' in kwargs) and ('Grid Orientation' in kwargs):
        grid_cell = GridCell(gid, nfields=nfields, module=module, **kwargs)
    else:
        
        local_random = kwargs.get('local_random')
        grid_orientation = kwargs.get('grid_orientation')
        module_width = kwargs.get('module_width')
    
        nx = kwargs.get('nx')
        ny = kwargs.get('ny')
    
        orientation = grid_orientation[module]
        spacing = module_width

        delta_spacing     = local_random.uniform(-10., 10.)
        delta_orientation = local_random.uniform(-10., 10.)

        cell_args = kwargs
        cell_args['Grid Spacing']     = np.array([spacing + delta_spacing], dtype='float32')
        cell_args['Grid Orientation'] = np.array([orientation + delta_orientation], dtype='float32')
        cell_args['Nx'] = nx
        cell_args['Ny'] = ny
        
        grid_cell = GridCell(gid, nfields=nfields, module=module, **cell_args)
    return grid_cell


def make_input_cell(gid, features_type, features):
    cell = None
    if features_type == selectivity_grid:
        cell = make_grid_cell(gid, features['Module'], features['Num Fields'], **features)
    elif features_type == selectivity_place:
        cell = make_place_cell(gid, features['Module'], features['Num Fields'], **features)
    else:
        raise RuntimeError('make_input_cell: unknown feature type %d' % features_type)
    return cell

    
def as_scalar(x):
    if isinstance(x, (np.ndarray, np.generic) ):
        return x.item()
    else:
        return x

class InputCell(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, gid, nfields=None, module=None, **kwargs):
        self.gid = as_scalar(gid)
        self.nfields = as_scalar(nfields)
        self.module = as_scalar(module)

        self.x_offset = kwargs.get('X Offset', None)
        self.y_offset = kwargs.get('Y Offset', None)
        self.cell_type = None
        self.rate_map  = None
        self.nx = as_scalar(kwargs.get('Nx', None))
        self.ny = as_scalar(kwargs.get('Ny', None))
        self.peak_rate = as_scalar(kwargs.get('Peak Rate', None))
        
    @abstractmethod
    def return_attr_dict(self):
        cell = {}
        cell['Num Fields'] = np.array([self.nfields], dtype='uint8')
        cell['Module'] = np.array([self.module], dtype='uint8')
        cell['X Offset'] = np.asarray(self.x_offset, dtype='float32')
        cell['Y Offset'] = np.asarray(self.y_offset, dtype='float32')
        cell['Nx'] = np.array([self.nx], dtype='int32')
        cell['Ny'] = np.array([self.ny], dtype='int32')
        cell['Rate Map'] = np.asarray(self.rate_map).reshape(-1,).astype('float32')
        cell['Peak Rate'] = np.array([self.peak_rate], dtype='float32')
        return cell

    @abstractmethod
    def generate_spatial_ratemap(self, interp_x, interp_y, **kwargs):
        pass



class GridCell(InputCell):
    def __init__(self, gid, nfields=[], module=[], **kwargs):
        super(GridCell, self).__init__(gid, nfields=nfields, module=module, **kwargs)
        self.cell_type = 0
        self.grid_spacing = as_scalar(kwargs.get('Grid Spacing', None))
        self.grid_orientation = as_scalar(kwargs.get('Grid Orientation', None))
        
    def return_attr_dict(self):
        cell = super(GridCell, self).return_attr_dict()
        cell['Cell Type'] = np.array([self.cell_type], dtype='uint8')
        cell['Grid Spacing'] = np.array([self.grid_spacing], dtype='float32')
        cell['Grid Orientation'] = np.array([self.grid_orientation], dtype='float32')
        return cell

    def generate_spatial_ratemap(self, interp_x, interp_y, **kwargs):

        a = 0.7
        b = -1.5
        c = 0.9
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
        rate_map = self.peak_rate * transfer(inner_sum) / max_rate
        self.rate_map = rate_map
        
        return rate_map
    
class PlaceCell(InputCell):
    def __init__(self, gid, nfields=None, module=None, **kwargs):
        super(PlaceCell, self).__init__(gid, nfields=nfields, module=module, **kwargs)
        self.cell_type = 1
        self.field_width = kwargs.get('Field Width', None)
        if nfields is None:
            self.num_fields = as_scalar(kwargs.get('Num Fields', None))
        else:
            self.num_fields = nfields
        
    def return_attr_dict(self):
        cell = super(PlaceCell, self).return_attr_dict()
        cell['Cell Type'] = np.array([self.cell_type], dtype='uint8')
        cell['Field Width'] = np.asarray(self.field_width, dtype='float32')
        return cell

    def generate_spatial_ratemap(self, interp_x, interp_y, **kwargs):

        if self.num_fields > 0:
            x_offset = self.x_offset
            y_offset = self.y_offset
            field_width = self.field_width
            nfields  = self.num_fields
            rate_map = np.zeros_like(interp_x)
            for n in xrange(nfields):
                current_map = self.peak_rate * np.exp(-((interp_x - x_offset[n]) / (field_width[n] / 3. / np.sqrt(2.))) ** 2.) * np.exp(-((interp_y  - y_offset[n]) / (field_width[n] / 3. / np.sqrt(2.))) ** 2.)
                rate_map    = np.maximum(current_map, rate_map)
            rate_map.reshape(-1,).astype('float32')
        else:
            rate_map = np.zeros( interp_x.shape ).astype('float32')

        self.rate_map = rate_map
        
        return rate_map

