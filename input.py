def get_broadband_albedo(star):
    """
    Returns broadband albedo parameters based on the stellar type.
    
    Parameters:
        star (str): The star type, which must be one of 'F', 'G', 'K', or 'M'.
        
    Returns:
        dict: A dictionary with the following keys:
              'Asnow', 'A_75', 'A_50', 'A_25', 'A_bi',
              'A_o', 'A_l', 'c_2k', 'c_1', 'c_2', 'c_5',
              'c_20', 'c_100', 'c_200', 'm'
    """
    if star == 'F':
        Asnow = 0.66833
        A_75  = 0.59884
        A_50  = 0.53664
        A_25  = 0.47961
        A_bi  = 0.42542
        A_o   = 0.32865
        A_l   = 0.41428
        c_2k  = 0.905
        c_1   = 0.994
        c_2   = 0.993
        c_5   = 0.990
        c_20  = 0.983
        c_100 = 0.969
        c_200 = 0.960
        m     = 0.83

    elif star == 'G':
        Asnow = 0.64622
        A_75  = 0.57636
        A_50  = 0.51363
        A_25  = 0.45585
        A_bi  = 0.40093
        A_o   = 0.31948
        A_l   = 0.41484
        c_2k  = 0.901
        c_1   = 0.992
        c_2   = 0.991
        c_5   = 0.988
        c_20  = 0.979
        c_100 = 0.963
        c_200 = 0.954
        m     = 0.80

    elif star == 'K':
        Asnow = 0.60392
        A_75  = 0.53716
        A_50  = 0.47708
        A_25  = 0.42150
        A_bi  = 0.36870
        A_o   = 0.30235
        A_l   = 0.40133
        c_2k  = 0.886
        c_1   = 0.990
        c_2   = 0.988
        c_5   = 0.984
        c_20  = 0.974
        c_100 = 0.955
        c_200 = 0.944
        m     = 0.78

    elif star == 'M':
        Asnow = 0.39800
        A_75  = 0.35478
        A_50  = 0.31546
        A_25  = 0.28406
        A_bi  = 0.24317
        A_o   = 0.23372
        A_l   = 0.33165
        c_2k  = 0.790
        c_1   = 0.973
        c_2   = 0.970
        c_5   = 0.959
        c_20  = 0.936
        c_100 = 0.900
        c_200 = 0.881
        m     = 0.63

    else:
        raise ValueError("Invalid star type. Must be one of 'F', 'G', 'K', or 'M'.")

    return {
        'Asnow': Asnow,
        'A_75': A_75,
        'A_50': A_50,
        'A_25': A_25,
        'A_bi': A_bi,
        'A_o': A_o,
        'A_l': A_l,
        'c_2k': c_2k,
        'c_1': c_1,
        'c_2': c_2,
        'c_5': c_5,
        'c_20': c_20,
        'c_100': c_100,
        'c_200': c_200,
        'm': m
    }


