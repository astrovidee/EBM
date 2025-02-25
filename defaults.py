"""
defaults module
===============

This module provides default configuration settings for the EBM project.
It includes parameters for simulations, constants, and initial settings.

Attributes:
    jmxdef (int): The default number of grid cells.
    runlengthdef (int): The default length of the simulation run.
    scaleQdef (float): The default scale factor for Q.
    Adef (float): Default value for parameter A.
    Bdef (float): Default value for parameter B.
    Dmagdef (float): Default value for D magnitude.
    Toffsetdef (int): Default temperature offset.
    obldef (int): Default oblateness value.
    eccdef (float): Default eccentricity.
    perdef (float): Default period.
    star (str): Default star type.
    landdef (str): Default land configuration.
    casenamedef (str): Default case name.
    hadleyflagdef (float): Default flag for Hadley circulation.
    albedoflagdef (float): Default flag for albedo setting.
    ice_modeldef (float): Default ice model parameter.
    coldstartdef (float): Default cold start parameter.
    Cldef (float): Default land heat capacity.
    Cwdef (float): Default water heat capacity.
    nudef (int): Default value for nu.
"""

jmxdef = 60
runlengthdef = 100
scaleQdef = 1.0
Adef = 203.3
Bdef = 2.09
Dmagdef = 0.44
Toffsetdef = -30
obldef = 0
eccdef = 0.0
perdef = 102.07
star = 'G'
landdef = 'modern'
casenamedef = 'Control'
hadleyflagdef = 1.0
albedoflagdef = 0.0
ice_modeldef = 1.0
coldstartdef = 0.0
Cldef = 0.45
Cwdef = 9.8
nudef = 3

defaults = {
    'jmxdef': jmxdef,
    'runlengthdef': runlengthdef,
    'scaleQdef': scaleQdef,
    'Adef': Adef,
    'Bdef': Bdef,
    'Dmagdef': Dmagdef,
    'Toffsetdef': Toffsetdef,
    'obldef': obldef,
    'eccdef': eccdef,
    'perdef': perdef,
    'star': star,
    'landdef': landdef,
    'casenamedef': casenamedef,
    'hadleyflagdef': hadleyflagdef,
    'albedoflagdef': albedoflagdef,
    'ice_modeldef': ice_modeldef,
    'coldstartdef': coldstartdef,
    'Cldef': Cldef,
    'Cwdef': Cwdef,
    'nudef': nudef
}
