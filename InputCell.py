from abc import ABCMeta, abstractmethod
import numpy as np

def instantiate_place_cell(context, gid, module, nfields):
    if 'module_width' not in context() or 'local_random' not in context() or 'nx' not in context() or 'ny' not in context():
        raise Exception('Context does not contain the necessary attributes to continue..')
    cell_args = {}
    cell_field_width = []
    this_width = context.module_width
    for n in xrange(nfields):
        delta_spacing = context.local_random.uniform(-10, 10)
        cell_field_width.append(this_width + delta_spacing)
    cell_args['Nx'] = context.nx
    cell_args['Ny'] = context.ny
    cell_args['Field Width'] = cell_field_width
    place_cell = PlaceCell(gid, nfields=nfields, module=module, **cell_args)
    return place_cell

def instantiate_grid_cell(context, gid, module, nfields):
    if 'grid_orientation' not in context() or 'module_width' not in context() or 'local_random' not in context():
        raise Exception('Context does not contain module width, orientation, or a Random obj')
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

class PlaceCell(InputCell):
    def __init__(self, gid, nfields=[], module=[], **kwargs):
        super(PlaceCell, self).__init__(gid, nfields=nfields, module=module, **kwargs)
        self.cell_type = [1]
        self.field_width = kwargs.get('Field Width', [])
        
    def return_attr_dict(self):
        cell = super(PlaceCell, self).return_attr_dict()
        cell['Cell Type'] = np.array([self.cell_type], dtype='uint8')
        cell['Field Width'] = np.array(self.field_width, dtype='float32')
        return cell
    
