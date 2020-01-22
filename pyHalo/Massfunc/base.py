class MassFunctionBase(object):

    """
    This class handles the distribution of key words tp each specific
    type of mass function parameterization.
    """

    def __init__(self, all_kwargs):

        allowed_parameterizations = ['power_law',
                                     'power_law_turnover',
                                     'delta_function']

        if 'mass_function_type' not in all_kwargs:
            raise Exception('must specify a mass function type. Allowed models '
                            'are '+str(allowed_parameterizations)+'.')

        if all_kwargs['mass_function_type'] not in allowed_parameterizations:
            raise Exception(str(all_kwargs['mass_function_type']) + ' not a valid mass function type.'
                                      'Valid parameterizations are: '+str(allowed_parameterizations))

        mfunc_type = all_kwargs['mass_function_type']

        self.kwargs_model = self.get_kwargs_model(all_kwargs, mfunc_type)

    def _get_kwargs(self, all_kwargs, required_kwargs, options, mfunc_type):

        kwargs_out = {}

        for key in required_kwargs:

            if key not in all_kwargs.keys():
                raise Exception(key + ' is a required key word for '
                                      'mass function type '+mfunc_type)

            kwargs_out[key] = required_kwargs[key]

        has_kwarg = False
        for key in options:
            if key in all_kwargs.keys():
                if has_kwarg:
                    raise Exception('You specified two different '
                                    'keyword arguments from the list: '+str(options)+
                                    ' that each specify a different procedure.'
                                    'Pick one option.')
                has_kwarg = True

            kwargs_out[key] = required_kwargs[key]

        return kwargs_out

    def get_kwargs_model(self, all_kwargs, mfunc_type):

        if mfunc_type == 'power_law':

            required_params = ['power_law_index', 'log_mlow',
                           'log_mhigh', 'log_m_break']

            options = ['sigma_sub', 'f_sub']

        elif mfunc_type == 'power_law_turnover':

            required_params = ['power_law_index', 'log_mlow',
                               'log_mhigh', 'log_m_break', 'break_scale',
                               'break_index', 'log_m_break']

            options = ['sigma_sub', 'f_sub']

        elif mfunc_type == 'delta_function':

            required_params = ['log_M', 'DM_mass_fraction']

            options = []

        else:
            raise Exception('mass function type '+str(mfunc_type)+' not recognized.')

        kwargs_model = self._get_kwargs(all_kwargs, required_params, options)

        return kwargs_model





