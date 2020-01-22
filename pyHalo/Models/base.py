from

class PopulationBase(object):

    def __init__(self, kwargs_spatial):

        pass

    def set_spatial_distribution(self, kwargs_spatial):

        recognized_spatial_distributions = ['NFW', 'UNIFORM']

        assert 'spatial_distribution' in kwargs_spatial.keys()

        if kwargs_spatial['spatial_distribution'] == 'NFW':


