################################################################list[M],
# The contents of this file are subject to the BSD 3Clause (New) License
# you may not use this file except in
# compliance with the License. You may obtain a copy of the License at
# http://directory.fsf.org/wiki/License:BSD_3Clause

# Software distributed under the License is distributed on an "AS IS"
# basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
# License for the specific language governing rights and limitations
# under the License.

# The Original Code is part of the PyRadi toolkit.

# The Initial Developer of the Original Code is M Konnik and CJ Willers,
# Portions created by CJ Willers are Copyright (C) 2006-2015
# All Rights Reserved.

# Contributor(s): ______________________________________.
################################################################
"""
This module provides a high level model for CCD and CMOS staring array 
signal chain modelling. The work is based on a paper and Matlab code by Mikhail Konnik,
available at:

- Paper available at: http://arxiv.org/pdf/1412.4031.pdf
- Matlab code available at: https://bitbucket.org/aorta/highlevelsensorsim

See the documentation at http://nelisw.github.io/pyradi-docs/_build/html/index.html 
or pyradi/doc/rystare.rst  for more detail.
"""

#prepare so long for Python 3
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division

__version__= ""
__author__='M Konnik and CJ Willers'
__all__=['photosensor'
]

import sys
if sys.version_info[0] > 2:
    print("pyradi is not yet ported to Python 3, because imported modules are not yet ported")
    exit(-1)

import numpy as np
import scipy as sp
import scipy.signal as signal
import scipy.stats as stats
import scipy.constants as const
import re

import pyradi.ryfiles as ryfiles
import pyradi.ryutils as ryutils
import pyradi.ryplot as ryplot


######################################################################################
def photosensor(strh5):
    """This routine simulates the behaviour of a CCD/CMOS sensor, performing the conversion 
    from irradiance to electrons, then volts, and then digital numbers.
     
    The process from incident photons to the digital numbers appeared on the image is outlined. 
    First of all, the radiometry is considered. Then, the process of conversion from photons to 
    electrons is outlined. Following that, conversion from electrons to voltage is 
    described.  Finally, the ADC converts the voltage signal into digital numbers. 
    The whole process is depicted on Figure below.
     
    .. image:: _images/camerascheme_horiz.png
     :width: 812px
     :align: center
     :height: 244px
     :alt: camerascheme_horiz.png
     :scale: 100 %

    Many noise sources contribute to the resulting noise image that is produced by
    photosensors. Noise sources can be broadly classified as either
    *fixed-pattern (time-invariant)* or *temporal (time-variant)*
    noise. Fixed-pattern noise refers to any spatial pattern that does not change
    significantly from frame to frame. Temporal noise, on the other hand, changes
    from one frame to the next. 



    Note that in the sequence below we add signal and noise signals linearly together. 
    For uncorrelated noise sources, the noise power values are added in quadrature, but
    that does not apply here, because we are adding instantaneous noise values (per pixel)
    so that these noise and signal values add linearly.
     
    Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    if not strh5['rystare/sensortype'].value in ['CCD', 'CMOS']:
        print('Sensor Simulator::: please select the sensor: CMOS or CCD!')
        exit(-1)

    #defining the constants such as speed of light c, Plank's h, and others.
    strh5 = set_photosensor_constants(strh5)

    #create the various data arrays
    strh5 = check_create_datasets(strh5)

    # diagram node 2 signal stored in 'rystare/signal/photonRateIrradiance'

    # add photon noise and calculate electron counts

    if strh5['rystare/flag/darkframe'].value:  
        # if this is  complete darkness, photon signal already set to zero
        pass
    else:  
        # there is light on the sensor

        # diagram node 2 signal stored in 'rystare/signal/photonRateIrradiance'

        # adjust 'rystare/signal/light' with responsivity PRNU non-uniformity
        if strh5['rystare/detector/response/NU/flag'].value:
            strh5 = responsivity_FPN_light(strh5)

        # diagram node 4 signal stored in 'rystare/signal/photonRateIrradianceNU

        strh5 = convert_to_electrons(strh5) 
        # at this point the photon signal is converted to electrons, after all 
        #responsivity effects but datector area and integration time not yet done.

        # diagram node 5 signal stored in 'rystare/signal/electronRateIrradiance'

        strh5 = multiply_detector_area(strh5) 

        # diagram node 6 signal stored in 'rystare/signal/electronRate'

        strh5 = multiply_integration_time(strh5) 

        # diagram node 7 signal stored in 'rystare/signal/lightelectrons'

    # no dark current or dark current effects added yet

    # add dark current noise
    if strh5['rystare/flag/darkcurrent'].value:     
        strh5 = dark_current_and_dark_noises(strh5) 
 
    # get total signal by adding the electrons generated by dark signal and electrons generated by light.
    strh5['rystare/signal/electrons'][...] = strh5['rystare/signal/lightelectrons'].value + \
              strh5['rystare/signal/darkcurrentelectrons'].value 

    # diagram node 12 signal in electrons stored in 'rystare/signal/electrons'

    # Full-well check-up and saturate the pixel if there are more electrons than full-well.
    # find all of pixels that are saturated (there are more electrons that full-well of the pixel)
    strh5['rystare/signal/electrons'][...] = np.where(
        strh5['rystare/signal/electrons'].value > strh5['rystare/fullwellelectrons'].value, 
        strh5['rystare/fullwellelectrons'].value, strh5['rystare/signal/electrons'].value)

    # find all of pixels that are less than zero and truncate to zero (no negative electron count).
    strh5['rystare/signal/electrons'][...] = np.where(
        strh5['rystare/signal/electrons'].value < 0, 0, strh5['rystare/signal/electrons'].value)

    # round the number of electrons.  
    strh5['rystare/signal/electrons'][...] = np.floor(strh5['rystare/signal/electrons'].value) 

    # diagram node 13 signal in electrons  after charge well clipping in 'rystare/signal/electrons'

    # Charge-to-Voltage conversion by Sense Node
    strh5 = charge_to_voltage(strh5)

    # diagram node 17 signal in volts stored in 'rystare/signal/voltage'

    # signal's Voltage amplification by Source Follower
    strh5 = source_follower(strh5)

    # diagram node 18 signal in volts stored in 'rystare/signal/voltage'

    # calculate the source follower noise in volts.  
    if strh5['rystare/sourcefollower/noise/flag'].value:
        strh5 = source_follower_noise(strh5) 

    # diagram node 19 source follower noise volts stored in 'rystare/sourcefollower/source_follower_noise'

    #add source follower noise
    strh5['rystare/signal/voltage'][...] = strh5['rystare/signal/voltage'].value + \
         strh5['rystare/sourcefollower/source_follower_noise']

    # diagram node 20 signal in volts stored in 'rystare/signal/voltage'

    strh5 = fixed_pattern_offset(strh5)

    # diagram node 21 fixed pattern offset in volts stored in 'rystare/darkoffset/NU/value'

    # signal's amplification and de-noising by Correlated Double Sampling
    strh5 = cds(strh5)

    # diagram node 23 signal in volts stored in 'rystare/signal/voltage'

    # Analogue-To-Digital Converter
    strh5 = adc(strh5)

    # diagram node 27 ADC signal stored in 'rystare/signal/DN'

    return strh5

######################################################################################
def set_photosensor_constants(strh5):
    r"""Defining the constants that are necessary for calculation of photon energy, dark current rate, etc.

     Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
   """
    #Sensor material constants    
    strh5['rystare/material/Eg-eV'] = 0.  #bandgap still to be computed at at temperature 
    strh5['rystare/material/Eg-eV-0'] = 1.1557  #bandgap energy for 0 degrees of K. [For Silicon, eV]
    strh5['rystare/material/alpha'] = 7.02100e-04 #Si material parameter, [eV/K].
    strh5['rystare/material/beta'] = 1108. #Si material parameter, [K].

    # band gap energy, [eV], Varshni equation
    strh5['rystare/material/Eg-eV'][...] = strh5['rystare/material/Eg-eV-0'].value - (strh5['rystare/material/alpha'].value * \
        (strh5['rystare/operatingtemperature'].value ** 2)) / (strh5['rystare/material/beta'].value + strh5['rystare/operatingtemperature'].value)

    #Fundamental constants
    strh5['rystare/constants/Boltzman-Constant-eV'] = const.physical_constants['Boltzmann constant in eV/K'][0] #Boltzman constant, [eV/K].
    strh5['rystare/constants/Boltzman-Constant-JK'] = const.physical_constants['Boltzmann constant'][0] #Boltzman constant, [J/K].
    strh5['rystare/constants/q'] = const.e # charge of an electron [C], coulomb

    #other constants
    strh5['rystare/constants/k1'] = 1.090900000e-14 #a constant;

    return strh5

######################################################################################
def check_create_datasets(strh5):
    r"""Create the arrays to store the various image-sized variables.

     Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
   """
    #determine the size of the sensor.
    sensor_size = strh5['rystare/imageSizePixels'].value

    #pre-allocating the matrices for photons, electrons, voltages and DNs.
    # strh5['rystare/signal/photonRateIrradiance'] = np.zeros(sensor_size)
    strh5['rystare/signal/photonRateIrradianceNU'] = np.zeros(sensor_size)
    strh5['rystare/signal/electronRateIrradiance'] = np.zeros(sensor_size)
    strh5['rystare/signal/electronRate'] = np.zeros(sensor_size)
    # strh5['rystare/signal/photonRate'] = np.zeros(sensor_size)
    # strh5['rystare/signal/photons'] = np.zeros(sensor_size)
    strh5['rystare/quantumEfficiency'] = np.zeros(sensor_size)
    strh5['rystare/signal/electrons'] = np.zeros(sensor_size)
    strh5['rystare/signal/darkcurrentelectrons'] = np.zeros(sensor_size)
    # strh5['rystare/signal/light'] = np.zeros(sensor_size)
    strh5['rystare/signal/lightelectrons'] = np.zeros(sensor_size)
    strh5['rystare/signal/voltage'] = np.zeros(sensor_size)
    strh5['rystare/signal/DN'] = np.zeros(sensor_size) 
    strh5['rystare/detector/response/NU/value'] = np.zeros(sensor_size) 
    strh5['rystare/darkresponse/NU/value'] = np.zeros(sensor_size) 
    strh5['rystare/noise/sn_reset/resetnoise'] = np.zeros(sensor_size) 
    strh5['rystare/sourcefollower/source_follower_noise'] = np.zeros(sensor_size)
    strh5['rystare/darkoffset/NU/value'] = np.ones(sensor_size) 
    strh5['rystare/signal/voltagebeforeSF'] = np.zeros(sensor_size)

    strh5['rystare/ADC/gain'] = np.ones(sensor_size)

    strh5['rystare/sourcefollower/sigma'] = np.zeros((1,1)) 
    strh5['rystare/sensenode/volt-fullwell'] = 0.
    strh5['rystare/sensenode/volt-min'] = 0.
    strh5['rystare/sensenode/capacitance'] = 0.
    strh5['rystare/sensenode/ResetKTC-sigma'] = 0.
    strh5['rystare/darkcurrentelectronsnonoise'] = 0.  

    if (strh5['rystare/sensenode/resetnoise/factor'].value > 1.):
        print('{} {} {} {}'.format('Warning! The compensation factor', strh5['rystare/sensenode/resetnoise/factor'].value,
            '(strh5["rystare/noise/sn_reset/Factor"]) you entered for the Sense Node Reset Noise cannot be more than 1!',
            'The factor is set to 1.'))
        strh5['rystare/sensenode/resetnoise/factor'].value = 1.
    else:
        if (strh5['rystare/sensenode/resetnoise/factor'].value < 0):
            print('{} {} {} {}'.format('Warning! The compensation factor', strh5['rystare/sensenode/resetnoise/factor'].value,
            '(strh5["rystare/noise/sn_reset/Factor"]) you entered for the Sense Node Reset Noise negative!',
            'The factor is set to 0, SNReset noise is not simulated.'))
            strh5['rystare/sensenode/resetnoise/factor'].value=0


    return strh5

######################################################################################
def source_follower(strh5):
    r"""The amplification of the voltage from Sense Node by Source Follower.

    Conventional sensor use a floating-diffusion sense node followed by a
    charge-to-voltage amplifier, such as a source follower.


    .. image:: _images/source_follower.png
     :width: 541px
     :height: 793px
     :align: center
     :scale: 30 %

    Source follower is one of basic single-stage field effect transistor (FET)
    amplifier topologies that is typically used as a voltage buffer. In such a
    circuit, the gate terminal of the transistor serves as the input, the source is
    the output, and the drain is common to both input and output. At low
    frequencies, the source follower has voltage gain:

    :math:`{A_{\text{v}}} = \frac{v_{\text{out}}}{v_{\text{in}}} = \frac{g_m R_{\text{S}}}{g_m R_{\text{S}} + 1} \approx 1 \qquad (g_m R_{\text{S}} \gg 1)`


    Source follower is a voltage follower, its gain is less than 1. Source followers
    are used to preserve the linear relationship between incident light, generated
    photoelectrons and the output voltage.

    The V/V non-linearity affect shot noise (but does not affect FPN curve) and can
    cause some shot-noise probability density compression. The V/V
    non-linearity non-linearity is caused by non-linear response in ADC or source
    follower. 

    The V/V non-linearity can be simulated as a change in source follower gain
    :math:`A_{SF}` as 
    a linear function of signal:

    :math:`A_{SF_{new}} = \alpha \cdot \frac{V_{REF} - S(V_{SF}) }{V_{REF} } + A_{SF},` 

    where :math:`\alpha = A_{SF}\cdot\frac{\gamma_{nlr} -1}{ V_{FW} }` and
    :math:`\gamma_{nlr}` is a non-linearity ratio of :math:`A_{SF}`. In the simulation
    we assume :math:`A_{SF} = 1` and :math:`\gamma_{nlr} = 1.05` i.e. 5\% of
    non-linearity of :math:`A_{SF}`. Then the voltage is multiplied on the new sense
    node gain :math:`A_{SF_{new}}`:

    :math:`I_{V} = I_{V}\cdot A_{SF_{new}}` 

    After that, the voltage goes to ADC for quantisation to digital numbers.

    Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    strh5['rystare/signal/voltagebeforeSF'][...] = strh5['rystare/signal/voltage'].value

    sones =  np.ones(strh5['rystare/imageSizePixels'].value)
    strh5['rystare/sourcefollower/gainA'] = strh5['rystare/sourcefollower/gain'].value * sones

    # calculating Source Follower VV non-linearity
    if strh5['rystare/sourcefollower/nonlinearity/flag'].value:
        nonlinearity_alpha = (strh5['rystare/sourcefollower/nonlinearity/ratio'].value - 1) * \
              (strh5['rystare/sourcefollower/gain'].value  / strh5['rystare/sensenode/volt-fullwell'].value)

        strh5['rystare/sourcefollower/gainA'][...] = sones * nonlinearity_alpha * \
                ((strh5['rystare/sensenode/vrefreset'].value - strh5['rystare/signal/voltage'].value) / \
                (strh5['rystare/sensenode/vrefreset'].value)) + \
                (strh5['rystare/sourcefollower/gain'].value)
                       

    # #Source Follower signal
    #     strh5['rystare/signal/voltage'][...] = (strh5['rystare/signal/voltage'].value) * strh5['rystare/sourcefollower/gainA'].value


    strh5['rystare/signal/voltage'][...] = strh5['rystare/signal/voltage'].value * \
                                            strh5['rystare/sourcefollower/gainA'].value


    return strh5

######################################################################################
def fixed_pattern_offset(strh5):
    """Add dark fixed pattens offset
 
    Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    strh5['rystare/darkoffset/NU/value'][...] = np.zeros(tuple(strh5['rystare/imageSizePixels'].value))
    if strh5['rystare/sensortype'].value in 'CMOS':  
        #If the sensor is CMOS and the darkoffset/NU is on 
        if strh5['rystare/darkoffset/NU/flag'].value: 

            # add dark fixed pattern offset
            sigma = strh5['rystare/sensenode/volt-fullwell'].value * \
                    strh5['rystare/darkoffset/NU/spread'].value

            noisematrix = FPN_models(strh5['rystare/imageSizePixels'].value[0],
                strh5['rystare/imageSizePixels'].value[1], 'column', 
                strh5['rystare/darkoffset/NU/model'].value, sigma)

            strh5['rystare/darkoffset/NU/value'][...] = noisematrix

    return strh5

######################################################################################
def cds(strh5):
    """Reducing the noise by Correlated Double Sampling, but right now the routine just adds the noise.

    Correlated Double Sampling (CDS) is a technique for measuring photo voltage
    values that removes an undesired noise. The sensor's output is measured twice.
    Correlated Double Sampling is used for compensation of Fixed pattern
    noise caused by dark current leakage,
    irregular pixel converters and the like. It appears on the same pixels at
    different times when images are taken. It can be suppressed with noise reduction
    and on-chip noise reduction technology. The main approach is CDS, having one
    light signal read by two circuits. 

    In CDS, a circuit measures the difference between the reset voltage and the
    signal voltage for each pixel, and assigns the resulting value of charge to the
    pixel. The additional step of measuring the output node reference voltage before
    each pixel charge is transferred makes it unnecessary to reset to the same level
    for each pixel. 

    First, only the noise is read. Next, it is read in combination with the light
    signal. When the noise component is subtracted from the combined signal, the
    fixed-pattern noise can be eliminated. 

    CDS is commonly used in image sensors to reduce FPN and reset noise. CDS only
    reduces offset FPN (gain FPN cannot be reduced using CDS). CDS in CCDs, PPS,
    and photogate APS, CDS reduces reset noise, in photodiode APS it increases it
    See Janesick's book and especially El Gamal's lectures.
 
    Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    strh5['rystare/signal/voltagebeforecds'] = strh5['rystare/signal/voltage'].value
    
    strh5['rystare/signal/voltage'][...] = strh5['rystare/signal/voltage'].value + strh5['rystare/darkoffset/NU/value'].value
                     
    # diagram node 22 fixed pattern offset added to signal in volts stored in 'rystare/signal/voltage'

    strh5['rystare/signal/voltage'][...] = strh5['rystare/signal/voltage'].value * strh5['rystare/CDS/gain']


    return strh5


######################################################################################
def adc(strh5):   
    r"""An analogue-to-digital converter (ADC) transforms a voltage signal into discrete codes.


    An analogue-to-digital converter (ADC) transforms a voltage signal into discrete
    codes. An :math:`N`-bit ADC has :math:`2^N` possible output codes with the difference between code
    being :math:`V_{ADC.REF}/2^N`. The resolution of the ADC indicates the number of
    discrete values that can be produced over the range of analogue values and can
    be expressed as:

    :math:`K_{ADC} = \frac{V_{ADC.REF} - V_\mathrm {min}}{N_{max}}`
    where :math:`V_\mathrm{ADC.REF}` is the maximum voltage that can be quantified,
    :math:`V_{min}` is minimum quantifiable voltage, and :math:`N_{max} = 2^N` is the number of
    voltage intervals. Therefore, the output of an ADC can be represented as:

    :math:`ADC_{Code} = \textrm{round}\left( \frac{V_{input}-V_{min}}{K_{ADC}} \right)`

    The lower the reference voltage :math:`V_{ADC.REF}`, the smaller the range of the
    voltages one can measure.

    After the electron matrix has been converted to voltages, the sense node reset
    noise and offset FPN noise are  added, the V/V gain non-linearity is applied (if
    desired), the ADC non-linearity is applied (if necessary). Finally the result is
    multiplied by ADC gain and rounded to produce the signal as a digital number:

    :math:`I_{DN} =  \textrm{round} (A_{ADC}\cdot I_{total.V}),`

    where :math:`I_\textrm{total.V} = (V_{ADC.REF} - I_{V})` is the total voltage signal
    accumulated during one frame acquisition, :math:`V_{ADC.REF}` is the maximum voltage
    that can be quantified by an ADC, and :math:`I_V` is the total voltage signal
    accumulated by the end of the exposure (integration) time and conversion.
    Usually :math:`I_V = I_{SN.V}` after the optional V/V non-linearity is applied. In
    this case, the conversion from voltages to digital signal is linear. The
    adcnonlinearity "non-linear ADC case is considered below".

    In terms of the ADC, the following non-linearity and noise should be considered
    for the simulations of the photosensors: Integral Linearity Error, Differential
    Linearity Error, quantisation error, and ADC offset.

    The DLE indicates the deviation from the ideal 1 LSB (Least Significant Bit)
    step size of the analogue input signal corresponding to a code-to-code
    increment. Assume that the voltage that corresponds to a step of 1 LSB is
    :math:`V_{LSB}`. In the ideal case, a change in the input voltage of :math:`V_{LSB}` causes
    a change in the digital code of 1 LSB. If an input voltage that is more than
    :math:`V_{LSB}` is required to change a digital code by 1 LSB, then the ADC has DLE
    error. In this case, the digital output remains constant when the input
    voltage changes from, for example, :math:`2 V_{LSB}`  to  :math:`4 V_{LSB}`, therefore
    corresponding the digital code can never appear at the output. That is, that
    code is missing.

    .. image:: _images/dle.png
     :width: 1360px
     :height: 793px
     :align: center
     :scale: 40 %

    In the illustration above, each input step should be precisely 1/8 of reference
    voltage. The first code transition from 000 to 001 is caused by an input change
    of 1 LSB as it should be. The second transition, from 001 to 010, has an input
    change that is 1.2 LSB, so is too large by 0.2 LSB. The input change for the
    third transition is exactly the right size. The digital output remains
    constant when the input voltage changes from 4 LSB to 5 LSB, therefore the code
    101 can never appear at the output.

    The ILE is the maximum deviation of the input/output characteristic from a
    straight line passed through its end points. For each voltage in the ADC input,
    there is a corresponding code at the ADC output. If an ADC transfer function is
    ideal, the steps are perfectly superimposed on a line. However, most real ADC's
    exhibit deviation from the straight line, which can be expressed in percentage
    of the reference voltage or in LSBs. Therefore, ILE is a measure of the
    straightness of the transfer function and can be greater than the differential
    non-linearity. Taking the ILE into account is important because it cannot be
    calibrated out. 

    .. image:: _images/ILE.png
     :width: 1360px
     :height: 715px
     :align: center
     :scale: 40 %

    For each voltage in the ADC input there is a corresponding word at the ADC
    output. If an ADC is ideal, the steps are perfectly superimposed on a line. But
    most of real ADC exhibit deviation from the straight line, which can be
    expressed in percentage of the reference voltage or in LSBs.

    In our model, we simulate the Integral Linearity Error (ILE) of the ADC as a
    dependency of ADC gain :math:`A_{ADC.linear}` on the signal value. Denote
    :math:`\gamma_{ADC.nonlin}` as an ADC non-linearity ratio (e.g., :math:`\gamma_{ADC.nonlin}
    = 1.04`). The linear ADC gain can be calculated from Eq.~\ref{eq:kadc} as
    :math:`A_{ADC} = 1/K_{ADC}` and used as :math:`A_{ADC.linear}`. The non-linearity
    coefficient :math:`\alpha_{ADC}` is calculated as:

    :math:`\alpha_{ADC} = \frac{1}{V_{ADC.REF}} \left( \frac{ \log(\gamma_{ADC.nonlin}
    \cdot A_{ADC.linear} )}{\log(A_{ADC.linear})} - 1 \right)`

    where :math:`V_\mathrm{ADC.REF}` is the maximum voltage that can be quantified by an
    ADC: 

    :math:`A_{ADC.nonlin} = A_{ADC.linear}^{1-\alpha_{ADC} I_{total.V}},`

    where :math:`A_{ADC.linear}` is the linear ADC gain. The new non-linear ADC conversion
    gain :math:`A_{ADC.nonlin}` is then used for the simulations.

    Quantisation errors are caused by the rounding, since an ADC has a
    finite precision. The probability distribution of quantisation noise is
    generally assumed to be uniform. Hence we use the uniform distribution to model
    the rounding errors.

    It is assumed that the quantisation error is uniformly distributed between -0.5
    and +0.5 of the LSB and uncorrelated with the signal.  Denote :math:`q_{ADC}` the
    quantising step of the ADC. For the ideal DC, the quantisation noise is:

     :math:`\sigma_{ADC} = \sqrt{ \frac{q_{ADC}^2 }{12}}.`

    If :math:`q_{ADC} = 1` then the quantisation noise is :math:`\sigma_{ADC} = 0.29` DN. The
    quantisation error has a uniform distribution. We do not assume any particular
    architecture of the ADC in our high-level sensor model.
    This routine performs analogue-to-digital convertation of volts to DN.

    Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    strh5['rystare/signal/voltagebeforeadc'] = strh5['rystare/signal/voltage'].value 
    # maximum number of DN.
    N_max = 2 ** (strh5['rystare/ADC/num-bits'].value)  

    # gain in DN/v
    adc_gain  = N_max / \
        (strh5['rystare/sensenode/volt-fullwell'].value - strh5['rystare/sensenode/volt-min'].value)     

    strh5['rystare/ADC/gain'][...] = adc_gain * strh5['rystare/ADC/gain'].value

    # Removing the reference Voltage  
    signal = strh5['rystare/sensenode/vrefreset'].value - strh5['rystare/signal/voltage'].value   

    if strh5['rystare/ADC/nonlinearity/flag'].value:   
        # account for non-linearity
        numerator = strh5['rystare/ADC/nonlinearity/ratio'].value * adc_gain
        nonlinearity_alpha = (np.log(numerator) / np.log(adc_gain) - 1) \
               / strh5['rystare/sensenode/volt-fullwell'].value

        strh5['rystare/ADC/gain'][...] = strh5['rystare/ADC/gain'].value ** (1 - nonlinearity_alpha * signal) 

    strh5['rystare/signal/voltage'][...] = signal
    
    # diagram node 24 ADC gain stored in 'rystare/ADC/gain'

    #DN given by offset plus signal * gain
    signalDN = np.round(strh5['rystare/ADC/offset'].value + strh5['rystare/ADC/gain'].value * signal)

    # diagram node 26 & 27 ADC signal stored in 'rystare/signal/DN'

    # Truncating the numbers that are less than 0 or larger than N-max
    signalDN[signalDN <= 0] = 0      
    signalDN[signalDN >= N_max] = N_max  

    strh5['rystare/signal/DN'][...] = signalDN 

    return strh5

######################################################################################
def charge_to_voltage(strh5):
    r"""The charge to voltage conversion occurs inside this routine

    After that, a new matrix strh5['rystare/signal/voltage'] is created and the raw 
    voltage signal is stored.


    After the charge is generated in the pixel by photo-effect, it is moved
    row-by-row to the sense amplifier that is separated from the pixels in case of
    CCD. The packets of charge are being shifted to the output sense node,
    where electrons are converted to voltage. The typical sense node region is
    presented on Figure below.

    .. image:: _images/CCD-sensenoderegion.png
     :width: 812px
     :align: center
     :scale: 50 %

    Sense node is the final collecting point at the end of the horizontal
    register of the CCD sensor. The CCD pixels are made with MOS devices used as
    reverse biased capacitors. The charge is readout by a MOSFET based charge to
    voltage amplifier. The output voltage is inversely proportional to the sense
    node capacitor. Typical example is that the sense node capacitor of the order
    :math:`50fF`, which produces a gain of :math:`3.2 \mu V/ e^-`. It is also important
    to minimize the noise of the output amplifier, \textbf{typically the largest
    noise source in the system}. Sense node converts charge to voltage with typical
    sensitivities :math:`1\dots 4 \mu V/e^-`.

    The charge collected in each pixel of a sensor array is converted to voltage
    by  sense capacitor  and  source-follower amplifier.

    Reset noise is induced during such conversion. Prior to the measurement
    of each pixel's charge, the CCD sense capacitor is reset to a reference level.
    Sense node converts charge to voltage with typical sensitivities :math:`1\dots 4 \mu V/e^-`. 
    The charge collected in each pixel of a sensor array is converted to
    voltage by sense capacitor and source-follower amplifier. 
    Reset noise is induced during such conversion. Prior to the measurement of each
    pixel's charge, the CCD sense node capacitor is reset to a reference level.

    Sense Node gain non-linearity, or V/e non-linearity

    The V/:math:`e^-` non-linearity affect both FPN and shot noise and can cause
    some shot-noise probability density compression. This type of non-linearity is
    due to sense node gain non-linearity. Then sense node sensitivity became
    non-linear (see Janesick's book):

    :math:`S_{SN} ( V_{SN}/e^- ) = \frac{S(V_{SN}) }{(k_1/q)  \ln( V_{REF}/[V_{REF} - S(V_{SN})] )}`

    The V/:math:`e^-` non-linearity can be expressed as  a non-linear dependency of
    signals in electron and a sense-node voltage:

    :math:`S[e^-] = \frac{k1}{q} \ln \left[ \frac{V_{REF}}{ V_{REF} -  S(V_{SN}) } \right]`

    The V/:math:`e^-` non-linearity affects photon shot noise and skews the
    distribution, however this is a minor effect. The V/:math:`e^-` non-linearity can
    also be thought as a sense node capacitor non-linearity: when a small signal is
    measured, :math:`C_{SN}` is fixed or changes negligible; on the other hand,
    :math:`C_SN` changes significantly and that can affect the signal being
    measured.

    For the simulation purpose, the V/:math:`e^-` non-linearity can be expressed as: 

    :math:`V_{SN} = V_{REF} - S(V_{SN}) = V_{REF}\exp\left[ - \frac{\alpha\cdot S[e^-]\cdot q }{k1} \right]`

    where :math:`k1=10.909*10^{-15}` and :math:`q` is the charge of an electron, and :math:`\alpha` is the coefficient of 
    non-linearity strength.

    Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    # diagram node 13 signal in electrons  after charge well clipping in 'rystare/signal/electrons'

    # Sense node capacitance, parameter, [F] Farad
    strh5['rystare/sensenode/capacitance'][...] = strh5['rystare/constants/q'].value / strh5['rystare/sensenode/gain'].value

    #voltage on the Full Well
    strh5['rystare/sensenode/volt-fullwell'][...] = strh5['rystare/sensenode/gain'].value * strh5['rystare/fullwellelectrons'].value
    strh5['rystare/sensenode/volt-min'][...] =      strh5['rystare/sensenode/gain'].value * 1.0

    #no Sense Node Reset Noise, ony vref
    vinput = strh5['rystare/sensenode/vrefreset'].value 
    
    if strh5['rystare/sensortype'].value in ['CMOS']: 

        if strh5['rystare/sensenode/resetnoise/flag'].value:

            # Obtain the matrix for the Sense Node Reset Noise, saved in strh5['rystare/noise/sn_reset/resetnoise']
            # sigma is stored in 'rystare/sensenode/ResetKTC-sigma' 
            strh5 = sense_node_reset_noise(strh5)

            # diagram node 15 reset noise in electrons stored in 'rystare/noise/sn_reset/resetnoise'
           
            vinput = strh5['rystare/sensenode/vrefreset'].value + \
                    strh5['rystare/sensenode/resetnoise/factor'].value * strh5['rystare/noise/sn_reset/resetnoise'].value
        
            # diagram node 15b reset noise in electrons stored in 'rystare/noise/sn_reset/resetnoise'

    if strh5['rystare/sensenode/nonlinear/flag'].value:
        strh5['rystare/signal/voltage'][...] = vinput * \
                   (np.exp(- strh5['rystare/sensenode/nonlinear/alpha'].value * strh5['rystare/constants/q'].value * \
                    strh5['rystare/signal/electrons'].value / strh5['rystare/constants/k1'].value))
    else:
        strh5['rystare/signal/voltage'][...] = vinput - strh5['rystare/signal/electrons'].value * strh5['rystare/sensenode/gain'].value

    # diagram node 17 signal in volts stored in 'rystare/signal/voltage'

    return strh5


######################################################################################
def sense_node_reset_noise(strh5):
    r"""This routine calculates the noise standard deviation for the sense node reset noise.

    Sense node Reset noise (kTC noise)

    Prior to the measurement of each pixel's charge packet, the sense node capacitor
    is reset to a reference voltage level. Noise is generated at the sense node by
    an uncertainty in the reference voltage level due to thermal variations in the
    channel resistance of the MOSFET reset transistor. The reference level of the
    sense capacitor is therefore different from pixel to pixel. 

    Because reset noise can be significant (about 50 rms electrons), most
    high-performance photosensors incorporate a noise-reduction mechanism such as
    correlated double sampling (CDS).

    kTC noise occurs in CMOS sensors, while for CCD sensors the sense
    node reset noise is removed~ (see Janesick's book) by Correlated Double
    Sampling (CDS). Random fluctuations of charge on the sense node during the reset
    stage result in a corresponding photodiode reset voltage fluctuation. The sense
    node reset noise (in volt units) is given by:

    :math:`\sigma_{RESET}=\sqrt{\frac{k_B T}{C_{SN}}}`

    By the relationship Q=CV it can be shown that the kTC noise can be expressed 
    as electron count by

    :math:`\sigma_{RESET}=\sqrt{\frac{k_B T C_{SN}}{q}}`

    see also https://en.wikipedia.org/wiki/Johnson%E2%80%93Nyquist_noise

    The simulation of the sense node reset noise may be performed as an addition of
    non-symmetric probability distribution to the reference voltage :math:`V_{REF}`.
    However, the form of distribution depends on the sensor's architecture and the
    reset technique. An Inverse-Gaussian distribution can be
    used for the simulation of kTC noise that corresponds to a hard reset technique
    in the CMOS sensor, and the Log-Normal distribution can be used for soft-reset
    technique. The sense node reset noise can be simulated for each :math:`(i,j)`-th pixel
    for the soft-reset case as:

    :math:`I_{SN.reset.V}=ln\mathcal{N}(0,\sigma_{RESET}^2)`

    then added to the matrix :math:`I_{REF.V}` in Volts that corresponds to the reference voltage.

    Note: For CCD, the sense node reset noise is entirely removed by CDS.

    Note: In CMOS photosensors, it is difficult to remove the reset noise for the specific CMOS pixels 
    architectures even after application of CDS. Specifically, the difficulties
    arise in 'rolling shutter' and 'snap' readout modes.
    The reset noise is increasing after CDS by a factor of :math:`\sqrt{2}`.
    Elimination of reset noise in CMOS is quite challenging.

     Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    #in [V], see Janesick, Photon Transfer, page 166.
    strh5['rystare/sensenode/ResetKTC-sigma'][...] = \
        np.sqrt((strh5['rystare/constants/Boltzman-Constant-JK'].value) * \
            (strh5['rystare/operatingtemperature'].value) / (strh5['rystare/sensenode/capacitance'].value))

    # #randomising the state of the noise generators
    #If seed is omitted or None, current system time is used
    np.random.seed(None)

    # Soft-Reset prob distribution for the CMOS noise
    # use the exp() to get log normal, then subtract 1 to move the mean value to zero.
    # for the small values used here, it is pretty much gaussian anyway!
    strh5['rystare/noise/sn_reset/resetnoise'][...] = \
         np.exp(strh5['rystare/sensenode/ResetKTC-sigma'].value * \
            np.random.standard_normal(tuple(strh5['rystare/imageSizePixels'].value))) - 1.   

    # diagram node 15 rest noise in electrons stored in 'rystare/noise/sn_reset/resetnoise'

    return strh5


######################################################################################
def dark_current_and_dark_noises(strh5):
    r"""This routine for adding dark current signals and noise, including dark FPN and dark shot noise.

    The dark signal is calculated for all pixels in the model. It is implemented using `ones` function in 
    MATLAB as a matrix of the same size as the simulated photosensor. For each :math:`(i,j)`-th pixel we have:

    :math:`I_{dc.e^-} = t_I\cdot D_R,`

    where :math:`D_R` is the average dark current: 

    :math:`D_R = 2.55\cdot10^{15}P_A D_{FM} T^{1.5} \exp\left[-\frac{E_{gap}}{2\cdot k\cdot T}\right],` 

    where:
    :math:`D_R` is in units of [e$^{-1}$/s],
    :math:`P_A` is the pixel's area [:math:`cm^2`];
    :math:`D_{FM}` is the dark current figure-of-merit in units of [nA/:math:`cm^2`] at 300K, 
    varies significantly with sensor manufacturer, and used in this simulations as 0.5 nA/:math:`cm^2`;
    :math:`E_{gap}` is the bandgap energy of the semiconductor which also varies with temperature;
    :math:`k` is Boltzman's constant that is :math:`8.617\cdot10^{-5} [eV/K].`

    The relationship between band gap energy and temperature can be described by Varshni's empirical expression, 

    :math:`E_{gap}(T)=E_{gap}(0)-\frac{\alpha T^2}{T+\beta},` 

    where :math:`E_{gap}(0)`, :math:`\alpha` and :math:`\beta` are material constants. The energy bandgap of 
    semiconductors tends to decrease as the temperature is increased. This behaviour can be better understood 
    if one considers that the inter-atomic spacing increases when the amplitude of the atomic vibrations 
    increases due to the increased thermal energy. This effect is quantified by the linear expansion 
    coefficient of a material.

    For the Silicon: :math:`E_{gap}(0) = 1.1557 [eV]`, :math:`\alpha = 7.021*10^{-4}` [eV/K], and :math:`\beta = 1108` [K].

    It appears that fill factor does not apply to dark noise (Janesick book p168 and Konnik's code does not show this).


     Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    #Dark current generation, caused over full detector pitch area, not active area only
    # translating the size to square centimeters, as in Janesick book.
    detareacm2 = strh5['rystare/pixelPitch'].value[0] * strh5['rystare/pixelPitch'].value[0] * 1e4
    

    #average quantity of dark current that is thermally generated [e]  !!! This is Janesick equations 11.15 and 11.16  
    strh5['rystare/darkcurrentelectronsnonoise'][...] = \
                strh5['rystare/integrationtime'].value * 2.55e15 * \
                detareacm2 * strh5['rystare/darkfiguremerit'].value * (strh5['rystare/operatingtemperature'].value ** 1.5) * \
                np.exp(- strh5['rystare/material/Eg-eV'].value / (2 * strh5['rystare/constants/Boltzman-Constant-eV'].value * \
                strh5['rystare/operatingtemperature'].value))

    # diagram node 8 dc dark current stored in 'rystare/darkcurrentelectronsnonoise'

    #creating the average value electrons dark signal image
    strh5['rystare/signal/darkcurrentelectrons'][...] = strh5['rystare/darkcurrentelectronsnonoise'].value * np.ones(strh5['rystare/signal/electrons'].value.shape)

    # add shot noise, based on average value
    if strh5['rystare/flag/DarkCurrent-DShot'].value:
        strh5['rystare/signal/darkcurrentelectrons'][...] = shotnoise(strh5['rystare/signal/darkcurrentelectrons'].value)

    # diagram node 9 dc dark current stored in 'rystare/darkcurrentelectrons'

    # multiply with dark current FPN 
    if strh5['rystare/darkresponse/NU/flag'].value:
        strh5 = responsivity_FPN_dark(strh5)


    return strh5


######################################################################################
def source_follower_noise(strh5):
    r"""The source follower noise routine, calculates noise in volts.

    The pixel's source follower noise limits the read noise, however in high-end CCD and CMOS cameras the source 
    follower noise has been driven down to one electron rms.
    Pixel source follower MOSFET noise consists of three types of noise:
    -  white noise;
    -  flicker noise;
    -  random telegraph noise (RTS).
    Each type of noise has its own physics that will be briefly sketched below.
    
    *Johnson noise (white noise)*

    Similarly to the reset noise in sense node, the source-follower amplifier MOSFET has a resistance that 
    generates thermal noise whose value is governed by the Johnson white noise equation. 
    It is therefore either referred to as Johnson noise or simply as white noise, since its magnitude is independent of frequency.
    If the effective resistance is considered to be the output impedance of the source-follower amplifier, the white noise, 
    in volts, is determined by the following equation:

    :math:`N_{white} (V_{SF}) = \sqrt{4kTBR_{SF}}` 

    where :math:`k` is Boltzmann's constant (J/K), :math:`T` is temperature [K], :math:`B` refers to the noise power bandwidth [Hz], 
    and :math:`R_{SF}` is the output impedance of the source-follower amplifier.
    
    *Flicker noise*

    The flicker noise is commonly referred to as :math:`1/f` noise because of its approximate inverse dependence on frequency.
    For cameras in which pixels are read out at less than approximately 1 megahertz, and with a characteristic :math:`1/f` noise 
    spectrum, the read noise floor is usually determined by 1/f noise. Note that the noise continues to decrease at this 
    rate until it levels off, at a frequency referred to as the :math:`1/f` corner frequency. For the typical MOSFET 
    amplifier, the white noise floor occurs at approximately 4.5  :math:`nV/Hz^{1/2}`. 

    Prominent sources of :math:`1/f` noise in an image sensor are pink-coloured noise generated in the photo-diodes and 
    the low-bandwidth analogue operation of MOS transistors due to imperfect contacts between two 
    materials. Flicker noise is generally accepted to originate due to the existence of interface states in the image sensor 
    silicon that turn on and off randomly according to different time constants. All systems exhibiting 1/f behaviour 
    have a similar collection of randomly-switching states. In the MOSFET, the states are traps at the silicon-oxide 
    interface, which arise because of disruptions in the silicon lattice at the surface. The level of :math:`1/f` noise 
    in a CCD sensor depends on the pixel sampling rate and from certain crystallographic orientations of silicon 
    wafer.

    
    *Random Telegraph Signal (RTS) noise*

    As the CCD and CMOS pixels are shrinking in dimensions, the low-frequency noise increases. 
    In such devices, the low-frequency noise performance is dominated by Random Telegraph Signals (RTS) on top 
    of the 1/f noise. The origin of such an RTS is attributed to the random trapping and de-trapping of mobile charge carriers 
    in traps located in the oxide or at the interface. The RTS is observed in MOSFETs as a fluctuation in the drain 
    current. A pure two-level RTS is represented in the frequency domain by a Lorentzian spectrum.

    Mathematically the source follower's noise power spectrum can be described as:
    :math:`S_{SF}(f) = W(f)^2 \cdot \left(1 + \frac{f_c}{f}\right)+S_{RTS}(f),` 

    where :math:`W(f)` is the thermal white noise [:math:`V/Hz^{1/2}`, typically :math:`15 nV/Hz^{1/2}` ], flicker noise 
    corner frequency :math:`f_c` in [Hz] (flicker noise corner frequency is the frequency where power spectrum of white and flicker noise are equal),
    and the RTS power spectrum is given (see Janesick's book):


    :math:`S_{RTS}(f) = \frac{2\Delta I^2 \tau_{RTS}}{4+(2\pi f  \tau_{RTS})^2},` 

    where :math:`\tau_{RTS}` is the RTS characteristic time constant [sec] and :math:`\Delta I` is the source follower current modulation induced by RTS [A].

    The source follower noise can be approximated as:

    :math:`\sigma_{SF} = \frac{\sqrt{\int\limits_{0}^{\infty} S_{SF}(f) H_{CDS}(f) df }}{A_{SN}A_{SF}(1-\exp^{-t_s/\tau_D})}`

    where:
    -  :math:`\sigma_{SF}` is the source follower noise [e- rms]
    -  :math:`f` is the electrical frequency [Hz]
    -  :math:`t_s` is the CDS sample-to-sampling time [sec]
    -  :math:`\tau_D` is the CDS dominant time constant (see Janesick's Scientific CCDs book) usually set as :math:`\tau_D = 0.5t_s` [sec].

    The :math:`H_{CDS}(f)` function is the CDS transfer function is (see Janesick's book):

    :math:`H_{CDS}(f) = \frac{1}{1+(2\pi f \tau_D)^2} \cdot [2-2\cos(2\pi f t_s)]` 

    First term sets the CDS bandwidth for the white noise rejection before sampling takes place 
    through :math:`B = 1/(4\tau_D)`, where :math:`B` is defined as the noise equivalent bandwidth [Hz].

    Note: In CCD photosensors, source follower noise is typically limited by the flicker noise.

    Note: In CMOS photosensors, source follower noise is typically limited by the RTS noise.
    As a side note, such subtle kind of noises is visible only on high-end ADC like 16 bit and more. 

    Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    # this is the CDS dominant time constant usually set as :math:`\tau_D = 0.5t_s` [sec].
    tau_D = 0.5 * (strh5['rystare/CDS/sampletosamplingtime'].value)
    tau_RTN = 0.1 * tau_D

    #frequency, with delta_f as a spacing.
    numFsamp = strh5['rystare/sourcefollower/dataclockspeed'].value / strh5['rystare/sourcefollower/freqsamplingdelta'].value
    f = np.linspace(1., strh5['rystare/sourcefollower/dataclockspeed'].value, numFsamp).reshape(1,-1)
    strh5['rystare/sourcefollower/noise/spectralfreq'] = f

    #CDS transfer function
    H_CDS = (2. - 2. * np.cos(2. * np.pi * strh5['rystare/CDS/sampletosamplingtime'].value * f)) / (1. + (2. * np.pi * tau_D * f) ** 2) 
    strh5['rystare/sourcefollower/noise/cdsgain'] = H_CDS

    # RTN noise only in CMOS photosensors
    S_RTN = np.zeros(f.shape)
    if strh5['rystare/sensortype'].value in ['CMOS']:  
        S_RTN = (2. * ((strh5['rystare/sourcefollower/deltaindmodulation'].value) ** 2) * tau_RTN) / (4 + (2 * np.pi * tau_RTN *f) ** 2) 
    strh5['rystare/sourcefollower/noise/spectrumRTN'] = S_RTN



    # white and 1/f noise
    W_SF = (strh5['rystare/sourcefollower/whitenoisedensity'].value ** 2) * (1. + strh5['rystare/sourcefollower/flickerCornerHz'].value / f) 
    strh5['rystare/sourcefollower/noise/spectrumwhiteflicker'] = W_SF

    #total noise
    S_SF =  W_SF + S_RTN 
    strh5['rystare/sourcefollower/noise/spectrumtotal'] = S_SF

    #Calculating the std of SF noise:
    nomin = np.sqrt(strh5['rystare/sourcefollower/freqsamplingdelta'].value * np.dot(S_SF, H_CDS.T))
    denomin =     (1 - np.exp(- (strh5['rystare/CDS/sampletosamplingtime'].value) / tau_D)).reshape(-1,1)  #\

    #the resulting source follower std noise
    strh5['rystare/sourcefollower/sigma'][...] = nomin / denomin

    #randomising the state of the noise generators
    np.random.seed(None) #If seed is omitted or None, current system time is used
    strh5['rystare/sourcefollower/source_follower_noise'][...] =\
        strh5['rystare/sourcefollower/sigma'].value * np.random.randn(strh5['rystare/imageSizePixels'].value[0],strh5['rystare/imageSizePixels'].value[1])

    return strh5



######################################################################################################
def multiply_detector_area(strh5):
    r"""This routine multiplies  detector area

    The input to the model of the photosensor is assumed to be a matrix :math:`E_{q}\in R^{N\times M}` 
    that has been converted to electronrate irradiance, corresponding to electron rate  [e/(m2.s)].  
    The electron rate irriance is converted to electron rate into the pixel by accounting for 
    detector area:

    :math:`\Phi_{q}  =  \textrm{round} \left(  E_{q} \cdot P_A    \right),`

    where :math:`P_A` is the area of a pixel [m2].
     
     Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
         | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    #Calculating the area of the pixel (in [m^2]).
    detarea = strh5['rystare/fillfactor'].value * strh5['rystare/pixelPitch'].value[0] * strh5['rystare/pixelPitch'].value[1]

    # calculate radiant flux [W] from irradiance irradiance [W/m^2] and area
    strh5['rystare/signal/electronRate'][...]  = detarea * strh5['rystare/signal/electronRateIrradiance'].value 

    return strh5

######################################################################################################
def multiply_integration_time(strh5):
    r"""This routine multiplies with integration time
     
    The input to the model of the photosensor is assumed to be a matrix :math:`E_{q}\in R^{N\times M}` 
    that has been converted to electrons, corresponding to electron rate  [e/s].  
    The electron rate is converted to electron count into the pixel by accounting for 
    detector integration time:

    :math:`\Phi_{q}  =  \textrm{round} \left(  E_{q} \cdot t_I  \right),`

    where :math:`t_{I}` is integration     (exposure) time.

     Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
         | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    #the number of electrons accumulated during the integration time  (rounded).
    strh5['rystare/signal/lightelectrons'][...] = np.round(strh5['rystare/signal/electronRate'].value * strh5['rystare/integrationtime'].value)

    return strh5

######################################################################################################
def convert_to_electrons(strh5):
    r"""This routine converts photon rate irradiance to electron rate irradiance
     
    Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    # Converting the signal from Photons to Electrons
    # Quantum Efficiency = Quantum Efficiency Interaction X Quantum Yield Gain.
    strh5['rystare/quantumEfficiency'][...] = strh5['rystare/externalquantumeff'].value * strh5['rystare/quantumyield'].value

    # number of electrons [e] generated in detector
    strh5['rystare/signal/electronRateIrradiance'][...] = strh5['rystare/signal/photonRateIrradianceNU'].value * strh5['rystare/quantumEfficiency'].value 

    return strh5

######################################################################################################
def shotnoise(sensor_signal_in):
    r"""This routine adds photon shot noise to the signal of the photosensor that is in photons.

    The photon shot noise is due to the random arrival of photons and can be
    described by a Poisson process. Therefore, for each :math:`(i,j)`-th element of
    the matrix :math:`\Phi_{q}` that contains the number of collected photons, a photon
    shot noise  is simulated as a Poisson process :math:`\mathcal{P}` with mean
    :math:`\Lambda`:

    :math:`\Phi_{ph.shot}=\mathcal{P}(\Lambda), \,\,\,\,\mbox{ where   } \Lambda = \Phi_{q}.`

    We use the `ryutils.poissonarray` function that generates Poisson random numbers
    with mean :math:`\Lambda`.  That is, the number of collected photons in 
    :math:`(i,j)`-th pixel of the simulated photosensor in the matrix :math:`\Phi_{q}` is
    used as the mean :math:`\Lambda` for the generation of Poisson random numbers to
    simulate the photon shot noise. The input of the `ryutils.poissonarray` function will
    be the matrix :math:`\Phi_{q}` that contains the number of collected photons. The
    output will be the matrix :math:`\Phi_{ph.shot} \rightarrow \Phi_{q}`, i.e., the signal
    with added photon shot noise.  The matrix :math:`\Phi_{ph.shot}` is recalculated
    each time the simulations are started, which corresponds to the temporal nature
    of the photon shot noise.

    Args:
        | sensor_signal_in (np.array[N,M]): photon irradiance in, in photons
 
    Returns:
        | sensor_signal_out (np.array[N,M]): photon signal out, in photons

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    # Since this is a shot noise, it must be random every time we apply it, unlike the Fixed Pattern Noise (PRNU or DSNU).
    # If seed is omitted or None, current system time is used
    return ryutils.poissonarray(sensor_signal_in, seedval=None)

######################################################################################################
def responsivity_FPN_light(strh5):
    r"""Adding the PRNU light noise to the sensor signal.

    The Photo Response Non-Uniformity (PRNU) is the spatial variation in pixel
    output under uniform illumination mainly due to variations in the surface area
    of the photodiodes. This occurs due to variations in substrate material during
    the fabrication of the sensor. This occurs due to variations in substrate material during
    the fabrication of the sensor.
 
    The PRNU is signal-dependent (proportional to the input signal) and is
    fixed-pattern (time-invariant). The PRNU factor is typically :math:`0.01\dots 0.02` 
    for a given sensor, but varies from one sensor to another.

    The photo response non-uniformity (PRNU) is considered in our numerical model
    as a temporally-fixed light signal non-uniformity. The PRNU is modelled using a 
    Gaussian distribution for each     :math:`(i,j)`-th pixel of the matrix :math:`I_{e^-}`, 
    as :math:`I_{PRNU.e^-}=I_{e^-}(1+\mathcal{N}(0,\sigma_{PRNU}^2))`
    where :math:`\sigma_{PRNU}` is the PRNU factor value. 

    Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    #the random generator seed is fixed with value from input
    np.random.seed(strh5['rystare/detector/response/NU/seed'].value)

    #matrix for the PRNU
    normalisedVariation = FPN_models(
        strh5['rystare/imageSizePixels'].value[0], strh5['rystare/imageSizePixels'].value[1],
        'pixel', strh5['rystare/detector/response/NU/model'].value, strh5['rystare/detector/response/NU/spread'].value)

    # diagram node 3 NU stored in 'rystare/detector/response/NU/value'

    #np.random.randn has mean=0, variance = 1, so we multiply with variance and add to mean
    strh5['rystare/detector/response/NU/value'][...] = (1 + normalisedVariation)

    #apply the PRNU noise to the light signal of the photosensor.
    strh5['rystare/signal/photonRateIrradianceNU'][...] = strh5['rystare/signal/photonRateIrradiance'].value * strh5['rystare/detector/response/NU/value'].value
        
    return strh5


######################################################################################################
def responsivity_FPN_dark(strh5):
    r"""Add dark current noises that consist of Dark FPN and Dark shot noise.

    Pixels in a hardware photosensor cannot be manufactured exactly the same from
    perfectly pure materials. There will always be variations in the photo detector
    area that are spatially uncorrelated, surface defects at
    the :math:`SiO_2/Si` interface (see Sakaguchi paper on dark current reduction), 
    and discrete     randomly-distributed charge generation centres. These
    defects provide a mechanism for thermally-excited carriers to move between the
    valence and   conduction bands. Consequently, the average dark   signal is not
    uniform but has a spatially-random and fixed-pattern noise (FPN) structure.  The
    dark current FPN can be expressed as follows:

    :math:`\sigma_{d.FPN} = t_I D_R \cdot D_N,`

    where :math:`t_I` is the integration time, :math:`D_R` is the  average dark current,
    and :math:`D_N` is the dark current FPN factor that is typically :math:`0.1\dots 0.4` for CCD and CMOS sensors. 

    There are also so called 'outliers' or 'dark spikes'; that is, some pixels generate a dark
    signal values much higher than the mean value of the dark signal. The mechanism
    of such 'dark spikes' or 'outliers' can be described by the Poole-Frenkel
    effect (increase in emission rate from a defect in the presence of an electric field). 

    *Simulation of dark current fixed pattern noise*

    The dark current Fixed
    Pattern Noise (FPN) is simulated using non-symmetric distributions to account
    for the 'outliers' or 'hot pixels'. It is usually assumed that the dark
    current FPN can be described by Gaussian distribution. However, such an
    assumption provides a poor approximation of a complicated noise picture. 

    Studies show that a
    more adequate model of dark current FPN is to use non-symmetric probability
    distributions. The concept is to use two distributions to describe very
    'leaky' pixels that exhibit higher noise level than others. The first
    distribution is used for the main body of the dark current FPN, with a uniform
    distribution  superimposed to model 'leaky' pixels. For  simulations at
    room-temperature (:math:`25^\circ` C) authors use a
    logistic distribution, where a higher proportion of the population is
    distributed in the tails. For higher
    temperatures, inverse Gaussian and
    Log-Normal distributions have been proposed. The Log-Normal distribution works well for
    conventional 3T APS CMOS sensors with comparatively high dark current.

    In our simulations we use the Log-Normal distribution for the simulation of dark
    current FPN in the case of short integration times, and superimposing other
    distributions for long integration times. The actual simulation code implements
    various models, including Log-Normal, Gaussian, and Wald distribution to emulate
    the dark current FPN noise for short- and long-term integration times.

    The dark current FPN for each pixel of the matrix :math:`I_{dc.shot.e^-}` is computed as:

    :math:`I_{dc.FPN.e^-}  = I_{dc.shot.e^-}  + I_{dc.shot.e^-} \cdot ln\mathcal{N}(0,\sigma_{dc.FPN.e^-}^2)`

    where :math:`\sigma_{dc.FPN.e^-} = t_I D_R  D_N`, :math:`D_R` is the average dark current, 
    and :math:`D_N` is the dark current FPN factor.
    Since the dark current FPN does not change from one frame to the next,  the
    matrix :math:`ln \mathcal{N}` is calculated once and then can be re-used similar to
    the PRNU simulations.

    The experimental results confirm
    that non-symmetric models, and in particular the Log-Normal distribution, 
    adequately  describe the dark current FPN in CMOS sensors, especially in the
    case of a long integration time (longer than 30-60 seconds).  For long-exposure case, one
    needs to superimpose two (or more, depending on the sensor) probability
    distributions.
 
     Args:
        | strh5 (hdf5 file): hdf5 file that defines all simulation parameters
 
    Returns:
        | in strh5: (hdf5 file) updated data fields

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """




    #get the initial deviation from the mean    
    #handle gauss and nongaussian different
    if strh5['rystare/darkresponse/NU/model'].value in ['Janesick-Gaussian', 'AR-ElGamal']:
        if 'rystare/darkresponse/NU/filter_params' in strh5:
            filter_params = strh5['rystare/darkresponse/NU/filter_params'].value
        else:
            filter_params = None
        
        darksignalnoisematrix = FPN_models(
            strh5['rystare/imageSizePixels'].value[0], strh5['rystare/imageSizePixels'].value[1],
            'pixel', strh5['rystare/darkresponse/NU/model'].value, strh5['rystare/darkresponse/NU/spread'].value,
            filter_params=filter_params)

        strh5['rystare/darkresponse/NU/value'][...] = (1 + darksignalnoisematrix)
        #gaussian noise values may be negative, here we limit them if desired
        #only 'Janesick-Gaussian' was tested, 'AR-ElGamal' limitnegative not yet tested, so negative-going values are allowed.
        if strh5['rystare/darkresponse/NU/model'].value in ['Janesick-Gaussian']:
            if strh5['rystare/darkresponse/NU/limitnegative'].value:
                strh5['rystare/darkresponse/NU/value'][...] = limitzero(strh5['rystare/darkresponse/NU/value'].value, thr=0.6) 

    #this would be Wald and lognormal
    else:
        darksignalnoisematrix = FPN_models(
            strh5['rystare/imageSizePixels'].value[0], strh5['rystare/imageSizePixels'].value[1],
            'pixel', strh5['rystare/darkresponse/NU/model'].value, strh5['rystare/darkresponse/NU/spread'].value)
        strh5['rystare/darkresponse/NU/value'][...] = (1 + darksignalnoisematrix)

    # diagram node 10 dark current nonuniformity stored in 'rystare/darkresponse/NU/value'

    #apply the darkFPN noise to the dark_signal.
    strh5['rystare/signal/darkcurrentelectrons'][...] = strh5['rystare/signal/darkcurrentelectrons'].value * strh5['rystare/darkresponse/NU/value'].value

    # diagram node 11 dark current electrons stored in 'rystare/signal/darkcurrentelectrons'

    return strh5


######################################################################################################
def FPN_models(sensor_signal_rows, sensor_signal_columns, noisetype, noisedistribution, 
           spread, filter_params=None):
    r"""The routine contains various models on simulation of Fixed Pattern Noise.

    There are many models for simulation of the FPN: some of the models are suitable
    for short-exposure time modelling (Gaussian), while other models are more suitable 
    for log-exposure modelling of dark current FPN.

    *Gaussian model (Janesick-Gaussian)*

    Fixed-pattern noise (FPN) arises from changes in dark currents due to variations
    in pixel geometry during fabrication of the sensor. FPN increases exponentially
    with temperature and can be measured in dark conditions. 
    Column FPN is caused by offset in the integrating amplifier,
    size variations in the integrating capacitor CF, channel charge injection from
    reset circuit. FPN components that are reduced by CDS.
    Dark current FPN can be expressed as:

    :math:`\sigma_{D_{FPN}} = D\cdot D_N,` 

    where :math:`D_N` is the dark current FPN quality, which is typically between 10\%
    and 40\% for CCD and CMOS sensors (see Janesick's book), and  :math:`D = t_I D_R`. 
    There are other models of dark FPN, for instance as a autoregressive process.

    *El Gamal model of FPN with Autoregressive process*

    To capture the structure of FPN in a CMOS sensor we express :math:`F_{i,j}` as the
    sum of a column FPN component :math:`Y_j` and a pixel FPN component :math:`X_{i,j}`.
    Thus, :math:`F_{i,j} = Y_j + X_{i,j},` where the :math:`Y_j`'s and the
    :math:`X_{i,j}`'s are zero mean random variables.

    The first assumption is that the random processes
    :math:`Y_{j}` and :math:`X_{i,j}` are uncorrelated. This assumption is reasonable
    since the column and pixel FPN are caused by different device parameter
    variations. We further assume that the column (and pixel) FPN processes are
    isotropic.

    The idea to use autoregressive processes to model FPN was proposed because their
    parameters can be easily and efficiently estimated from data.
    The simplest model, namely first order isotropic
    autoregressive processes is considered.  This model can be extended to higher order
    models, however, the results suggest that additional model complexity may not be
    warranted.   

    The model assumes that the column FPN process :math:`Y_{j}` is a first order
    isotropic autoregressive process of the form: 

    :math:`Y_j = a(Y_{j-1}+Y_{j+1}) + U_j` 

    where the :math:`U_j` s are zero mean, uncorrelated random variables with the same
    variance :math:`\sigma_U` , and :math:`0 \leq a \leq 1` is a parameter that
    characterises the dependency of :math:`Y_{j}` on its two neighbours.

    The model assumes that the pixel FPN process :math:`X_{i,j}` is a two dimensional
    first order isotropic autoregressive process of the form:

    :math:`X_{i,j} = b(X_{i-1,j} + X_{i+1,j} +  X_{i,j-1} + X_{i,j+1} ) + V_{i,j}` 

    where the :math:`V_{i,j}` s are zero mean uncorrelated random variables with the
    same variance :math:`\sigma_V` , and  :math:`0 \leq b \leq 1` is a parameter that
    characterises the dependency of :math:`X_{i,j}` on its four neighbours.
 
    Args:
        | sensor_signal_rows(int): number of rows in the signal matrix
        | sensor_signal_columns(int): number of columns in the signal matrix
        | noisetype(string): type of noise to generate: ['pixel' or 'column']
        | noisedistribution(string): the probability distribution name ['AR-ElGamal', 'Janesick-Gaussian', 'Wald', 'LogNormal']
        | spread(float): spread around mean value (sigma/chi/lambda) for the probability distribution
        | filter_params([nd.array]): a vector of parameters for the probability filter
 
    Returns:
        | noiseout (np.array[N,M]): generated noise of FPN.

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    if noisedistribution in ['AR-ElGamal']: # AR-ElGamal FPN model
        if not filter_params:
            print('When using AR-ElGamal the filter parameters must be defined')
            exit(-1)
        if noisetype in ['pixel']:
            x2 = np.random.randn(sensor_signal_rows, sensor_signal_columns) # uniformly distributed White Gaussian Noise.
            #Matlab's filter operates on the first dimension of the array, while scipy.signal.lfilter by default operates on the the last dimension.
            # http://stackoverflow.com/questions/16936558/matlab-filter-not-compatible-with-python-lfilter
            noiseout = sp.signal.lfilter(1,filter_params,x2,axis=0)  # here y is observed (filtered) signal. Any WSS process y[n] can be of

        if noisetype in ['column']:
            x = sp.signal.lfilter(1, filter_params, np.random.randn(1, sensor_signal_columns))    # AR(1) model
            # numpy equivalent of matlab repmat(a, m, n) is tile(a, (m, n))
            # http://stackoverflow.com/questions/1721802/what-is-the-equivalent-of-matlabs-repmat-in-numpy
            # noiseout = repmat_(x,sensor_signal_rows,1) # making PRNU as a ROW-repeated noise, just like light FPN
            noiseout = np.tile(x, (sensor_signal_rows, 1)) # making PRNU as a ROW-repeated noise, just like light FPN

    elif noisedistribution in ['Janesick-Gaussian']:    
        #Janesick-Gaussian FPN model np.random.randn has mean=0, variance = 1
        #multiply with spread to get stddev equal to spread
        if spread is None:
            print('When using Janesick-Gaussian the spread must be defined')
            exit(-1)

        if noisetype in ['pixel']:
            noiseout = np.random.randn(sensor_signal_rows,sensor_signal_columns)  # here y is observed (filtered) signal. Any WSS process y[n] can be of

        if noisetype in ['column']:
            x = np.random.randn(1,sensor_signal_columns)   # dark FPN [e] <------ Konnik
            noiseout = np.tile(x, (sensor_signal_rows, 1))  #  making PRNU as a ROW-repeated noise, just like light FPN

        if noisetype in ['row']:
            x = np.random.randn(sensor_signal_rows,1)  #dark FPN [e] <------ Konnik
            noiseout = np.tile(x, (1, sensor_signal_columns))  #making PRNU as a ROW-repeated noise, just like light FPN
            
        noiseout = noiseout * spread    
      
    elif noisedistribution in ['Wald']:  # Wald FPN model
        if spread is None:
            print('When using Wald the spread must be defined')
            exit(-1)

        if noisetype in ['pixel']:
            noiseout = distributions_generator('wald',spread,[sensor_signal_rows,sensor_signal_columns]) + np.random.randn(sensor_signal_rows,sensor_signal_columns)

        if noisetype in ['column']:
            x = distributions_generator('wald',spread,[1,sensor_signal_columns]) + np.random.randn(1,sensor_signal_columns)
            noiseout = np.tile(x, (sensor_signal_rows, 1))  #making PRNU as a ROW-repeated noise, just like light FPN

    elif noisedistribution in ['LogNormal']:
        if spread is None:
            print('When using LogNormal the spread must be defined')
            exit(-1)

        if noisetype in ['pixel']:
            noiseout = distributions_generator('lognorm',[0.0,spread],[sensor_signal_rows,sensor_signal_columns])

        if noisetype in ['column']:
            x = distributions_generator('lognorm',[0.0,spread],[1,sensor_signal_columns])
            noiseout = np.tile(x, (sensor_signal_rows, 1))


    return noiseout

######################################################################################
def create_HDF5_image(imageName, imtype, pixelPitch, numPixels, fracdiameter=0, fracblurr=0, 
                   irrad_scale=1, irrad_min=0, wavelength=0.55e-6, steps=10):
    r"""This routine performs makes a simple illuminated circle with blurred boundaries.

    Then the  sensor's radiant irradiance in units [W/m2] are converted to  
    photon rate irradiance in units [q/m2.s)] by relating one photon's energy
    to power at the stated wavelength by :math:`Q_p=\frac{h\cdot c}{\lambda}`,
    where :math:`\lambda` is wavelength, :math:`h` is Planck's constant and :math:`c` is
    the speed of light. 

    The image file is in HDF5 format, containing the input parameters to the image creation process.
    A few minimum entries are required, but you can add any information youo wish to document the
    data.  The following minimum HDF5 entries are required by pyradi.rystare:

        | ``'image/imageName'`` (string):  the image name  
        | ``'image/PhotonRateIrradianceNoNoise'`` np.array[M,N]:  a float array with the image pixel values no noise 
        | ``'image/PhotonRateIrradiance'`` np.array[M,N]:  a float array with the image pixel values with noise
        | ``'image/pixelPitch'``:  ([float, float]):  detector pitch in m [row,col]  
        | ``'image/imageSizePixels'``:  ([int, int]): number of pixels [row,col]  
        | ``'image/imageFilename'`` (string):  the image file name  
        | ``'image/wavelength'`` (float):  where photon rate calcs are done  um
        | ``'image/imageSizeRows'`` (int):  the number of image rows
        | ``'image/imageSizeCols'`` (int):  the number of image cols
        | ``'image/imageSizeDiagonal'`` (float):  the FPA diagonal size in mm 
        | ``'image/irradianceLux'`` (float):  the maximum luminous exitance in the image lux (optional)
        | ``'image/irradianceWatts'`` (float):  the maximum exitance in the image W/m2 (optional)
        | ``'image/temperature'`` (float):  the maximum target temperature in the image K (optional)

    Args:
        | imageName (string): the image name, used to form the filename
        | imtype (string): string to define the type if image to be created ['zeros','disk','stairslin','stairslog']
        | pixelPitch ([float, float]):  detector pitch in m [row,col]
        | numPixels ([int, int]): number of pixels [row,col]
        | fracdiameter (float):  diameter of the disk as fraction of minimum image size
        | fracblurr (float):   blurr of the disk as fraction of minimum image size
        | irrad_scale (float): multiplicative scale factor (max value)
        | irrad_min (float): additive minimum value in the image
        | wavelength (float): wavelength where photon rate calcs are done in [m]
        | steps (int): number of steps in the stairs image

    Returns:
        | nothing: as a side effect an image file is written

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """

    hdffilename = 'image-{}-{}-{}.hdf5'.format(imageName, numPixels[0], numPixels[1])
    imghd5 = ryfiles.erase_create_HDF(hdffilename)
    imghd5['image/imageSizePixels'] = numPixels
    imghd5['image/wavelength'] = wavelength
    imghd5['image/pixelPitch'] = pixelPitch
    imghd5['image/imageSizeRows'] = pixelPitch[0] * numPixels[0]
    imghd5['image/imageSizeCols'] = pixelPitch[0] * numPixels[0]
    imghd5['image/imageSizeDiagonal'] = np.sqrt((pixelPitch[0] * numPixels[0]) ** 2. + (pixelPitch[1] * numPixels[1]) ** 2)

    varx = np.linspace(-imghd5['image/imageSizeCols'].value/2, imghd5['image/imageSizeCols'].value/2, imghd5['image/imageSizePixels'].value[1])
    vary = np.linspace(-imghd5['image/imageSizeRows'].value/2, imghd5['image/imageSizeRows'].value/2, imghd5['image/imageSizePixels'].value[0])
    x1, y1 = np.meshgrid(varx, vary)

    if imtype in ['zeros']:
        imghd5['image/imageName'] = imageName
        imghd5['image/imageFilename'] = hdffilename

        dset = imghd5.create_dataset('image/irradianceLux', numPixels, compression="gzip")
        dset[...] = np.zeros((x1.shape))
        dset = imghd5.create_dataset('image/PhotonRateIrradianceNoNoise', numPixels, compression="gzip")
        dset[...] = np.zeros((x1.shape))
        dset = imghd5.create_dataset('image/PhotonRateIrradiance', numPixels, compression="gzip")
        dset[...] = np.zeros((x1.shape))
        
    elif imtype in ['disk']:

        minSize = np.min(imghd5['image/imageSizeRows'].value, imghd5['image/imageSizeRows'].value)
        imghd5['image/imageName'] = imageName
        imghd5['image/imageFilename'] = hdffilename
        imghd5['image/disk_diameter'] = fracdiameter * minSize
        imghd5['image/blurr'] = fracblurr * minSize
        imghd5['image/irrad_scale'] = irrad_scale 
        imghd5['image/irrad_min'] = irrad_min 

        delta_x = varx[1] - varx[0]
        delta_y = vary[1] - vary[0]

        Uin = ryutils.circ(x1,y1,imghd5['image/disk_diameter'].value) * imghd5['image/irrad_scale'].value + imghd5['image/irrad_min'].value

        dia = max(1, 2 * round(imghd5['image/blurr'].value / np.max(delta_x,delta_y)))
        varx = np.linspace(-dia, dia, 2 * dia)
        vary = np.linspace(-dia, dia, 2 * dia)
        x, y = np.meshgrid(varx, vary)
        H = ryutils.circ(x, y, dia)
        Ein = (np.abs(signal.convolve2d(Uin, H, mode='same'))/np.sum(H))  ** 2

        #photon rate irradiance lux, with no  noise
        dset = imghd5.create_dataset('image/irradianceLux', numPixels, compression="gzip")
        dset[...] = 683 * 1.019 * np.exp(-285.51 * (wavelength*1e6 - 0.5591)**2) * Ein

        #Power in a single photon, in [Joule = Watt*s]
        P_photon = const.h * const.c / wavelength

        #convert to photon rate irradiance
        Ein = Ein / P_photon
        
        #photon rate irradiance in the image ph/(m2.s), with no photon noise
        dset = imghd5.create_dataset('image/PhotonRateIrradianceNoNoise', numPixels, compression="gzip")
        dset[...] = Ein
        #photon rate irradiance in the image ph/(m2.s), adding photon noise
        dset = imghd5.create_dataset('image/PhotonRateIrradiance', numPixels, compression="gzip")
        dset[...] = shotnoise(Ein) 
        
    elif imtype in ['stairslin','stairslog']:
        #this section creates a stair-case signal increasing from irrad_min to irrad_scale+irrad_min
        imghd5['image/imageName'] = '{}-{}'.format(imageName,steps)
        imghd5['image/imageFilename'] = hdffilename
        imghd5['image/irrad_scale'] = irrad_scale 
        imghd5['image/irrad_min'] = irrad_min 
        imghd5['image/steps'] = steps 

        size = imghd5['image/imageSizePixels'].value[1]

        if imtype in ['stairslin']:
            varx = np.linspace(0,size-1,size)
        else:
            varx = np.logspace(-1,np.log10(size-1),size)
        varx =  ((varx/(size/steps)).astype(int)).astype(float) / steps
        varx = varx / np.max(varx) 

        vary = np.linspace(-imghd5['image/imageSizeRows'].value/2, imghd5['image/imageSizeRows'].value/2, imghd5['image/imageSizePixels'].value[0])

        x, y = np.meshgrid(varx,vary)

        Ein = x * np.ones(x.shape) * imghd5['image/irrad_scale'].value + imghd5['image/irrad_min'].value 
        Ein *= np.where(y<-imghd5['image/imageSizeRows'].value/3.,0, 1)
        Ein *= np.where(y>imghd5['image/imageSizeRows'].value/3.,0, 1)

        #photon rate irradiance lux, with no  noise
        dset = imghd5.create_dataset('image/irradianceLux', numPixels, compression="gzip")
        dset[...] = 683 * 1.019 * np.exp(-285.51 * (wavelength*1e6 - 0.5591)**2) * Ein

        #Power in a single photon, in [Joule = Watt*s]
        P_photon = const.h * const.c / wavelength

        #convert to photon rate irradiance
        Ein = Ein / P_photon
        
        #photon rate irradiance in the image ph/(m2.s), with no photon noise
        dset = imghd5.create_dataset('image/PhotonRateIrradianceNoNoise', numPixels, compression="gzip")
        dset[...] = Ein
        #photon rate irradiance in the image ph/(m2.s), adding photon noise
        dset = imghd5.create_dataset('image/PhotonRateIrradiance', numPixels, compression="gzip")
        dset[...] = shotnoise(Ein) 

    elif imtype in ['stairslinIRQ','stairslogIRQ']:
        #this section creates a stair-case signal increasing from irrad_min to irrad_scale+irrad_min
        imghd5['image/imageName'] = '{}-{}'.format(imageName,steps)
        imghd5['image/imageFilename'] = hdffilename
        imghd5['image/irrad_scale'] = irrad_scale 
        imghd5['image/irrad_min'] = irrad_min 
        imghd5['image/steps'] = steps 

        size = imghd5['image/imageSizePixels'].value[1]

        if imtype in ['stairslin']:
            varx = np.linspace(0,size-1,size)
        else:
            varx = np.logspace(-1,np.log10(size-1),size)
        varx =  ((varx/(size/steps)).astype(int)).astype(float) / steps
        varx = varx / np.max(varx) 

        vary = np.linspace(-imghd5['image/imageSizeRows'].value/2, imghd5['image/imageSizeRows'].value/2, imghd5['image/imageSizePixels'].value[0])

        x, y = np.meshgrid(varx,vary)

        Ein = x * np.ones(x.shape) * imghd5['image/irrad_scale'].value + imghd5['image/irrad_min'].value 
        Ein *= np.where(y<-imghd5['image/imageSizeRows'].value/3.,0, 1)
        Ein *= np.where(y>imghd5['image/imageSizeRows'].value/3.,0, 1)

        #photon rate irradiance lux, with no  noise
        dset = imghd5.create_dataset('image/irradianceLux', numPixels, compression="gzip")
        dset[...] = 683 * 1.019 * np.exp(-285.51 * (wavelength*1e6 - 0.5591)**2) * Ein

        #Power in a single photon, in [Joule = Watt*s]
        P_photon = const.h * const.c / wavelength

        #convert to photon rate irradiance
        Ein = Ein / P_photon
        
        #photon rate irradiance in the image ph/(m2.s), with no photon noise
        dset = imghd5.create_dataset('image/PhotonRateIrradianceNoNoise', numPixels, compression="gzip")
        dset[...] = Ein
        #photon rate irradiance in the image ph/(m2.s), adding photon noise
        dset = imghd5.create_dataset('image/PhotonRateIrradiance', numPixels, compression="gzip")
        dset[...] = shotnoise(Ein) 



    else:
        print('Unknown image type {}\n no image file created'.format(imtype))

    imghd5.flush()
    imghd5.close()

######################################################################################
def define_metrics():
    r"""This simple routine defines various handy shorthand for cm and mm in the code. 

    The code defines a number of scaling factors to convert to metres and radians

    Args:
        | None

    Returns:
        | scaling factors.

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """

    m = 1
    cm = 0.01 * m
    mm = 0.001 * m
    mum = 1e-06 * m
    nm = 1e-09 * m
    rad = 1
    mrad = 0.001 * rad

    return m,cm,mm,mum,nm,rad,mrad
    
    
######################################################################################
def limitzero(a, thr=0.6):
    r"""Performs an asymetric clipping to prevent negative values.
    The lower-end values are clumped up towards the lower positive values, while 
    upper-end values are not affected.  

    This function is used to prevent negative random variables for wide sigma and low
    mean value, e.g., N(1,.5).  If the random variables are passed through this function
    The resulting distribution is not normal any more, and has no known analytical form.

    A threshold value of around 0.6 was found to work well for N(1,small) up to N(1,.5).

    Before you use this function, first check the results using the code below in the main
    body of this file.

    Args:
        | a (np.array): an array of floats, 

    Returns:
        | scaling factors.

    Raises:
        | No exception is raised.

    Author: CJ Willers
    """
    ashape = a.shape
    a = a.flatten()
    a = np.where(a<thr, thr * np.exp(( a - thr) / (1 * thr)), 0) + np.where(a >= thr, a, 0)

    return   a.reshape(ashape)

######################################################################################
def distribution_exp(distribParams, out, funcName):
    r"""Exponential Distribution

    This function is meant to be called via the `distributions_generator` function.

    :math:`\textrm{pdf} = \lambda * \exp( -\lambda * y )`

    :math:`\textrm{cdf} = 1 - \exp(-\lambda * y)`

    - Mean = 1/lambda
    - Variance = 1/lambda^2
    - Mode = lambda
    - Median = log(2)/lambda
    - Skewness = 2
    - Kurtosis = 6

    GENERATING FUNCTION:   :math:`T=-\log_e(U)/\lambda`

    PARAMETERS: distribParams[0] is lambda - inverse scale or rate (lambda>0)

    SUPPORT: y,  y>= 0

    CLASS: Continuous skewed distributions

    NOTES: The discrete version of the Exponential distribution is 
    the Geometric distribution.

    USAGE: 

    - y = randraw('exp', lambda, sampleSize) - generate sampleSize number of variates from the Exponential distribution with parameter 'lambda';
 
    EXAMPLES:

    1.   y = randraw('exp', 1, [1 1e5]);
    2.   y = randraw('exp', 1.5, 1, 1e5);
    3.   y = randraw('exp', 2, 1e5 );
    4.   y = randraw('exp', 3, [1e5 1] );

    SEE ALSO:
    GEOMETRIC, GAMMA, POISSON, WEIBULL distributions
    http://en.wikipedia.org/wiki/Exponential_distribution
    """
    if distribParams is None or len(distribParams)==0:
        distribParams = [1.]
    if checkParamsNum(funcName,'Exponential','exp',distribParams,[1]):
        _lambda = distribParams[0]
        if validateParam(funcName,'Exponential','exp','lambda','lambda',_lambda,[str('> 0')]):
            out = - np.log(np.random.rand(*out.shape)) / _lambda

    return out



######################################################################################
def distribution_lognormal(distribParams, out, funcName):
    """THe Log-normal Distribution (sometimes: Cobb-Douglas or antilognormal distribution)

    This function is meant to be called via the `distributions_generator` function.

    pdf = 1/(y*sigma*sqrt(2*pi)) * exp(-1/2*((log(y)-mu)/sigma)^2)
    cdf = 1/2*(1 + erf((log(y)-mu)/(sigma*sqrt(2))));

    - Mean = exp( mu + sigma^2/2 );
    - Variance = exp(2*mu+sigma^2)*( exp(sigma^2)-1 );
    - Skewness = (exp(1)+2)*sqrt(exp(1)-1), for mu=0 and sigma=1;
    - Kurtosis = exp(4) + 2*exp(3) + 3*exp(2) - 6; for mu=0 and sigma=1;
    - Mode = exp(mu-sigma^2);

    PARAMETERS: mu - location, sigma - scale (sigma>0)

    SUPPORT:  y,  y>0

    CLASS: Continuous skewed distribution                      

    NOTES:

    1. The LogNormal distribution is always right-skewed
    2. Parameters mu and sigma are the mean and standard deviation of y in (natural) log space.
    3. mu = log(mean(y)) - 1/2*log(1 + var(y)/(mean(y))^2)
    4. sigma = sqrt( log( 1 + var(y)/(mean(y))^2) )

    USAGE:

    - randraw('lognorm', [], sampleSize) - generate sampleSize number
      of variates from the standard Lognormal distribution with 
      location parameter mu=0 and scale parameter sigma=1 
    - randraw('lognorm', [mu, sigma], sampleSize) - generate sampleSize number
      of variates from the Lognormal distribution with 
      location parameter 'mu' and scale parameter 'sigma'

    EXAMPLES:

    1.   y = randraw('lognorm', [], [1 1e5]);
    2.   y = randraw('lognorm', [0, 4], 1, 1e5);
    3.   y = randraw('lognorm', [-1, 10.2], 1e5 );
    4.   y = randraw('lognorm', [3.2, 0.3], [1e5 1] );
    """

    if distribParams is None or len(distribParams)==0:
        distribParams = [0., 1.]

    if checkParamsNum(funcName,'Lognormal','lognorm',distribParams,[0,2]):
        mu = distribParams[0]
        sigma = distribParams[1]
        if validateParam(funcName,'Lognormal','lognorm','[mu, sigma]','sigma',sigma,[str('> 0')]):
            out = np.exp(mu + sigma * np.random.randn(*out.shape)) 
    
    return out
  

######################################################################################
def distribution_inversegauss(distribParams, out, funcName):
    """The Inverse Gaussian Distribution

    This function is meant to be called via the `distributions_generator` function.

    The Inverse Gaussian distribution is left skewed distribution whose
    location is set by the mean with the profile determined by the
    scale factor.  The random variable can take a value between zero and
    infinity.  The skewness increases rapidly with decreasing values of
    the scale parameter.

    pdf(y) = sqrt(_lambda/(2*pi*y^3)) * exp(-_lambda./(2*y).*(y/mu-1).^2)

    cdf(y) = normcdf(sqrt(_lambda./y).*(y/mu-1)) + exp(2*_lambda/mu)*normcdf(sqrt(_lambda./y).*(-y/mu-1))

    where  normcdf(x) = 0.5*(1+erf(y/sqrt(2))); is the standard normal CDF
         
    - Mean     = mu
    - Variance = mu^3/_lambda
    - Skewness = sqrt(9*mu/_lambda)
    - Kurtosis = 15*mean/scale
    - Mode = mu/(2*_lambda)*(sqrt(9*mu^2+4*_lambda^2)-3*mu)

    PARAMETERS: mu - location; (mu>0), _lambda - scale; (_lambda>0)

    SUPPORT: y,  y>0

    CLASS: Continuous skewed distribution

    NOTES:

    1. There are several alternate forms for the PDF, some of which have more than two parameters
    2. The Inverse Gaussian distribution is often called the Inverse Normal
    3. Wald distribution is a special case of The Inverse Gaussian distribution where the mean is a 
       constant with the value one.
    4. The Inverse Gaussian distribution is a special case of The Generalized Hyperbolic Distribution

    USAGE:

    - randraw('ig', [mu, _lambda], sampleSize) - generate sampleSize number of variates 
      from the Inverse Gaussian distribution with parameters mu and _lambda;

    EXAMPLES:

    1.   y = randraw('ig', [0.1, 1], [1 1e5]);
    2.   y = randraw('ig', [3.2, 10], 1, 1e5);
    3.   y = randraw('ig', [100.2, 6], 1e5 );
    4.   y = randraw('ig', [10, 10.5], [1e5 1] );
 
    SEE ALSO:     WALD distribution

    Method:

    There is an efficient procedure that utilizes a transformation yielding two roots.
    If Y is Inverse Gauss random variable, then following [1] we can write:
    V = _lambda*(Y-mu)^2/(Y*mu^2) ~ Chi-Square(1)

    i.e. V is distributed as a _lambda-square random variable with one degree of freedom.
    So it can be simply generated by taking a square of a standard normal random number.
    Solving this equation for Y yields two roots:

    y1 = mu + 0.5*mu/_lambda * ( mu*V - sqrt(4*mu*_lambda*V + mu^2*V.^2) );
    and
    y2 = mu^2/y1;

    In [2] showed that  Y can be simulated by choosing y1 with probability
    mu/(mu+y1) and y2 with probability 1-mu/(mu+y1)

    References:
    [1] Shuster, J. (1968). On the Inverse Gaussian Distribution Function, Journal of the American 
    Statistical Association 63: 1514-1516.

    [2] Michael, J.R., Schucany, W.R. and Haas, R.W. (1976). Generating Random Variates Using 
    Transformations with Multiple Roots, The American Statistician 30: 88-90.

    http://en.wikipedia.org/wiki/Inverse_Gaussian_distribution
    """

    if distribParams is None or len(distribParams)==0:
        distribParams = [0., 1.]

    if checkParamsNum(funcName,'Inverse Gaussian','ig',distribParams,[2]):
        mu = distribParams[0]
        _lambda = distribParams[1]
        if validateParam(funcName,'Inverse Gaussian','ig','[mu, _lambda]','mu',mu,[str('> 0')]) & \
            validateParam(funcName,'Inverse Gaussian','ig','[mu, _lambda]','_lambda',_lambda,[str('> 0')]):
            chisq1 = np.random.randn(*out.shape) ** 2
            out = mu + 0.5 * mu / _lambda * (mu * chisq1 - np.sqrt(4 * mu * _lambda * chisq1 + mu ** 2 * chisq1 ** 2))
            l = np.random.rand(*out.shape) >= mu / (mu + out)
            out[l] = mu ** 2.0 / out[l]

    return out


######################################################################################
def distribution_logistic(distribParams, out, funcName):
    """The Logistic Distribution

    This function is meant to be called via the `distributions_generator` function.

    The logistic distribution is a symmetrical bell shaped distribution.
    One of its applications is an alternative to the Normal distribution
    when a higher proportion of the population being modeled is
    distributed in the tails.

    pdf(y) = exp((y-a)/k)./(k*(1+exp((y-a)/k)).^2)

    cdf(y) = 1 ./ (1+exp(-(y-a)/k))

    - Mean = a
    - Variance = k^2*pi^2/3
    - Skewness = 0
    - Kurtosis = 1.2

    PARAMETERS: a - location,  k - scale (k>0);

    SUPPORT: y,  -Inf < y < Inf

    CLASS: Continuous symmetric distribution                      

    USAGE:

    - randraw('logistic', [], sampleSize) - generate sampleSize number of variates from the 
      standard Logistic distribution with location parameter a=0 and scale parameter k=1;                    
    - Logistic distribution with location parameter 'a' and scale parameter 'k';

    EXAMPLES:

    1.   y = randraw('logistic', [], [1 1e5]);
    2.   y = randraw('logistic', [0, 4], 1, 1e5);
    3.   y = randraw('logistic', [-1, 10.2], 1e5 );
    4.   y = randraw('logistic', [3.2, 0.3], [1e5 1] );

    Method:

    Inverse CDF transformation method.

    http://en.wikipedia.org/wiki/Logistic_distribution
    """

    if distribParams is None or len(distribParams)==0:
        distribParams = [0., 1.]
    if checkParamsNum(funcName,'Logistic','logistic',distribParams,[0,2]):
        a = distribParams[0]
        k = distribParams[1]
        if validateParam(funcName,'Laplace','laplace','[a, k]','k',k,[str('> 0')]):
            u1 = np.random.rand(*out.shape)
            out = a - k * np.log(1.0 / u1 - 1)

    return out


######################################################################################
def distribution_wald(distribParams, out, funcName):
    """The Wald Distribution

    This function is meant to be called via the `distributions_generator` function.

    The Wald distribution is as special case of the Inverse Gaussian Distribution
    where the mean is a constant with the value one.

    pdf = sqrt(chi/(2*pi*y^3)) * exp(-chi./(2*y).*(y-1).^2);

    - Mean     = 1
    - Variance = 1/chi
    - Skewness = sqrt(9/chi)
    - Kurtosis = 3+ 15/scale

    PARAMETERS: chi - scale parameter; (chi>0)

    SUPPORT: y,  y>0

    CLASS: Continuous skewed distributions

    USAGE:

    - randraw('wald', chi, sampleSize) - generate sampleSize number of variates from the 
      Wald distribution with scale parameter 'chi';

    EXAMPLES:

    1.   y = randraw('wald', 0.5, [1 1e5]);
    2.   y = randraw('wald', 1, 1, 1e5);
    3.   y = randraw('wald', 1.5, 1e5 );
    4.   y = randraw('wald', 2, [1e5 1] );
    """                      

    if distribParams is None or len(distribParams)==0:
        distribParams = [0.]
    if checkParamsNum(funcName,'Wald','wald',distribParams,[1]):
        chi = distribParams[0]
        if validateParam(funcName,'Wald','wald','chi','chi',chi,[str('> 0')]):
            # out = feval_(funcName,'ig',[1,chi],*out.shape)
            out = distributions_generator('ig', [1, chi], sampleSize=out.shape)

    return out


######################################################################################
def distributions_generator(distribName=None, distribParams=None, sampleSize=None): 
    """The routine contains various models for simulation of FPN (DSNU or PRNU).

    This function allows the user to select the distribution by name and pass requisite
    parameters in a list (which differs for different distrubutions).  The size of the 
    distribution is defined by a scalar or list.

    sampleSize follows Matlab conventions:

    - if None then return a single scalar value
    - if scalar int N then return NxN array
    - if tuple then return tuple-sized array


    Possible values for distribName:
        | 'exp','exponential'
        | 'lognorm','lognormal','cobbdouglas','antilognormal'
        | 'ig', 'inversegauss', 'invgauss'
        | 'logistic'
        | 'wald'

    Args:
        | distribName (string): required distribution name
        | distribParams ([float]): list of distribution parameters (see below)
        | sampleSize (None,int,[int,int]): Size of the returned random set

    Returns:
        | out (float, np.array[N,M]): set of random variables for selected distribution.

    Raises:
        | No exception is raised.

    The routine set generates various types of random distributions, and is based on the 
    code randraw  by Alex Bar Guy  &  Alexander Podgaetsky
    These programs are distributed in the hope that they will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    Author: Alex Bar Guy, comments to alex@wavion.co.il
    """

    funcName = distributions_generator.__name__

    #remove spaces from distribution name
    # print(distribName)
    pattern = re.compile(r'\s+')
    distribNameInner = re.sub(pattern, '', distribName).lower()    

    if type(sampleSize) is int:
        out = np.zeros((sampleSize, sampleSize))
    elif type(sampleSize) is list or type(sampleSize) is tuple:
        out = np.zeros(sampleSize)
    else:
        out = np.zeros(1)

    if distribName is not None:

        if distribNameInner in ['exp','exponential']:
            out = distribution_exp(distribParams=None, out=out, funcName=funcName)

        elif distribNameInner in ['lognorm','lognormal','cobbdouglas','antilognormal']:
            out = distribution_lognormal(distribParams=None, out=out, funcName=funcName)

        elif distribNameInner in ['ig', 'inversegauss', 'invgauss']:
            out = distribution_inversegauss(distribParams=None, out=out, funcName=funcName)

        elif distribNameInner in ['logistic']:
            out = distribution_logistic(distribParams=None, out=out, funcName=funcName)

        elif distribNameInner in ['wald']:
            out = distribution_wald(distribParams=None, out=out, funcName=funcName)

        else:
            print('\n distributions_generator: Unknown distribution name: {}/{} \n'.format(distribName, distribNameInner))

    if out.shape == (1,):
        out = out[0]
    return out  

##########################################################################################3
def validateParam(funcName=None, distribName=None, runDistribName=None, distribParamsName=None, paramName=None, param=None, conditionStr=None):
    """Validate the range and number of parameters

    Args:
        | funcName (string):  distribution name
        | distribName (string):  distribution name
        | runDistribName (string):  run distribution name
        | distribParamsName
        | paramName 
        | param
        | conditionStr

    Returns:
        | True if the requirements are matched

    Raises:
        | No exception is raised.    
    """
    import math

    condLogical = True

    eqCondStr = ''
    for i,strn in enumerate(conditionStr):
        if i == 0:
            eqCondStr =  eqCondStr + conditionStr[i]
        else:
            eqCondStr =  eqCondStr + ' and ' + conditionStr[i]

        eqCond = conditionStr[i][0:2]
        # print('{} {} '.format(conditionStr[i], eqCond), end='')
        #remove spaces 
        pattern = re.compile(r'\s+')
        eqCond = re.sub(pattern, '', eqCond)   
        # print('{}'.format(eqCond))

        # print(eqCond)
        # funcName=funcName, distribName='Wald', runDistribName='wald', distribParamsName='chi', paramName='chi', param=chi, conditionStr[i]='> 0')

        if eqCond in ['<']: 
            condLogical &= param < float(conditionStr[i][2:])

        elif eqCond in ['<=']: 
            condLogical &= param <= float(conditionStr[i][2:])

        elif eqCond in ['>']: 
            condLogical &= param > float(conditionStr[i][2:])

        elif eqCond in ['>=']:
            condLogical &= param >= float(conditionStr[i][2:])

        elif eqCond in ['!=']: 
            condLogical &= param != float(conditionStr[i][2:])

        elif eqCond in ['==']:

            if 'integer' in conditionStr[i][2:]:
                condLogical &= param == math.floor_(param)
            else:
                condLogical &= param == math.float(conditionStr[i][2:])

    if not condLogical:
        print('{} Variates Generation: {}({}, {});\n Parameter {} should be {}\n (run {} ({}) for help)'.format(distribName, 
            funcName,runDistribName,distribParamsName,paramName,eqCondStr,funcName,runDistribName))

    return condLogical

##########################################################################################3
def checkParamsNum(funcName,distribName,runDistribName,distribParams,correctNum):
    """See if the correct number of parameters was supplied.  More than one number may apply

    Args:
        | funcName (string):  distribution name
        | distribName (string):  distribution name
        | distribParams ([float]): list of distribution parameters (see below)
        | correctNum ([int]): list with the possible numbers of parameters

    Returns:
        | True if the requirements are matched

    Raises:
        | No exception is raised.
    """

    proceed = True
    if not len(distribParams) in correctNum:
        print('{} Variates Generation:\n {} {} {} {} {}'.format(distribName, 
            'Wrong number of parameters (run ', funcName, "('", runDistribName, "') for help) "))
        proceed = False

    # print(distribParams, correctNum, proceed)

    return proceed

################################################################
def run_example(doTest='Advanced', outfilename='Output', pathtoimage=None, 
    doPlots=False, doHisto=False, doImages=False):
    """This code provides examples of use of the pyradi.rystare model for 
    a CMOS/CCD photosensor.

    Two models are provided 'simple' and 'advanced'

    doTest can be 'Simple' or 'Advanced'


    Args:
        | doTest (string):  which example to run 'Simple', or 'Advanced'
        | outfilename (string):  filename for output files
        | pathtoimage (string):  fully qualified path to where the image is located
        | doPlots (boolean):  flag to control the creation of false colour image plots with colour bars 
        | doHisto (boolean):  flag to control the creation of image histogram plots
        | doImages (boolean):  flag to control the creation of monochrome image plots 

    Returns:
        | hdffilename (string): output HDF filename

    Raises:
        | No exception is raised.

    Author: Mikhail V. Konnik, revised/ported by CJ Willers

    Original source: http://arxiv.org/pdf/1412.4031.pdf
    """
    import os.path
    from matplotlib import cm as mcm
    import matplotlib.mlab as mlab

    import pyradi.ryfiles as ryfiles
    import pyradi.ryutils as ryutils


    if doTest in ['Simple']:
        prefix = 'PS'
    elif  doTest in ['Advanced']:
        prefix = 'PA'
    else:
        exit('Undefined test')

    [m, cm, mm, mum, nm, rad, mrad] = define_metrics()

    #open the file to create data structure and store the results, remove if exists

    hdffilename = '{}{}.hdf5'.format(prefix, outfilename)
    if os.path.isfile(hdffilename):
        os.remove(hdffilename)
    strh5 = ryfiles.open_HDF(hdffilename)

    #sensor parameters
    strh5['rystare/sensortype'] = 'CCD' # must be in capitals
    # strh5['rystare/sensortype'] = 'CMOS' # must be in capitals

    # full-frame CCD sensors has 100% fil factor (Janesick: 'Scientific Charge-Coupled Devices')
    if strh5['rystare/sensortype'].value in ['CMOS']:
        strh5['rystare/fillfactor'] = 0.5 # Pixel Fill Factor for CMOS photo sensors.
    else:
        strh5['rystare/fillfactor'] = 1.0 # Pixel Fill Factor for full-frame CCD photo sensors.

    strh5['rystare/integrationtime'] = 0.01 # Exposure/Integration time, [sec].
    strh5['rystare/externalquantumeff'] = 0.8  # external quantum efficiency, fraction not reflected.
    strh5['rystare/quantumyield'] = 1. # number of electrons absorbed per one photon into material bulk
    strh5['rystare/fullwellelectrons'] = 2e4 # full well of the pixel (how many electrons can be stored in one pixel), [e]

    # Light Noise parameters
    strh5['rystare/flag/photonshotnoise'] = True #photon shot noise.

    # photo response non-uniformity noise (PRNU), or also called light Fixed Pattern Noise (light FPN)
    strh5['rystare/detector/response/NU/flag'] = True
    strh5['rystare/detector/response/NU/seed'] = 362436069
    strh5['rystare/detector/response/NU/model'] = 'Janesick-Gaussian' 
    strh5['rystare/detector/response/NU/spread'] = 0.01 # sigma [about 1\% for CCD and up to 5% for CMOS]

    # Dark Current Noise parameters
    strh5['rystare/flag/darkcurrent'] = True
    strh5['rystare/operatingtemperature'] = 300. # operating temperature, [K]
    strh5['rystare/darkfiguremerit'] = 1. # dark current figure of merit, [nA/cm2].  For very poor sensors, add DFM
    #  Increasing the DFM more than 10 results to (with the same exposure time of 10^-6):
    #  Hence the DFM increases the standard deviation and does not affect the mean value.

    # dark current shot noise
    strh5['rystare/flag/DarkCurrent-DShot'] = True

    #dark current Fixed Pattern Noise 
    strh5['rystare/darkresponse/NU/flag'] = True
    # Janesick's book: dark current FPN quality factor is typically between 10\% and 40\% for CCD and CMOS sensors
    strh5['rystare/darkresponse/NU/seed'] = 362436128
    strh5['rystare/darkresponse/NU/limitnegative'] = True # only used with 'Janesick-Gaussian' 

    if doTest in ['Simple']:
        strh5['rystare/darkresponse/NU/model'] = 'Janesick-Gaussian' 
        strh5['rystare/darkresponse/NU/spread'] =  0.3 #0.3-0.4 sigma for dark current signal (Janesick's book)
    elif  doTest in ['Advanced']:
        strh5['rystare/darkresponse/NU/model'] = 'LogNormal' #suitable for long exposures
        strh5['rystare/darkresponse/NU/spread'] = 0.4 # lognorm_sigma.
    else:
        pass

    # #alternative model
    # strh5['rystare/darkresponse/NU/model']  = 'Wald'
    # strh5['rystare/darkresponse/NU/spread']  = 2.0 #small parameters (w<1) produces extremely narrow distribution, large parameters (w>10) produces distribution with large tail.

    # #alternative model
    # strh5['rystare/darkresponse/NU/model']  = 'AR-ElGamal'
    # strh5['rystare/darkresponse/NU/filter_params']  = [1., 0.5] # see matlab filter or scipy lfilter functions for details

    #sense node charge to voltage
    strh5['rystare/sensenode/nonlinear/flag'] = False
    strh5['rystare/sensenode/nonlinear/alpha'] = 0.05 #how many times should A_SF be increased due to non-linearity?    
    strh5['rystare/sensenode/gain'] = 5e-6 # Sense node gain, A_SN [V/e]
    strh5['rystare/sensenode/resetnoise/flag'] = True
    strh5['rystare/sensenode/resetnoise/factor'] = 0.8 # the compensation factor of the Sense Node Reset Noise: 
                                           # 1 - no compensation from CDS for Sense node reset noise.
                                           # 0 - fully compensated SN reset noise by CDS.
    strh5['rystare/sensenode/vrefreset'] = 3.1 # Reference voltage to reset the sense node. [V] typically 3-10 V.

    #source follower
    strh5['rystare/sourcefollower/gain'] = 1. # Source follower gain, [V/V], lower means amplify the noise.


    #source follower
    strh5['rystare/sourcefollower/nonlinearity/flag'] = True # VV non-linearity
    strh5['rystare/sourcefollower/nonlinearity/ratio'] = 1.05 # > 1 for lower signal, < 1 for higher signal
    strh5['rystare/sourcefollower/flickerCornerHz'] = 1e6 #flicker noise corner frequency $f_c$ in [Hz], where power spectrum of white and flicker noise are equal [Hz].
    strh5['rystare/sourcefollower/whitenoisedensity'] = 15e-9 #thermal white noise [\f$V/Hz^{1/2}\f$, typically \f$15 nV/Hz^{1/2}\f$ ]
    strh5['rystare/sourcefollower/deltaindmodulation'] = 1e-8 #[A] source follower current modulation induced by RTS [CMOS ONLY]
    strh5['rystare/sourcefollower/dataclockspeed'] = 20e6 #MHz data rate clocking speed.
    strh5['rystare/sourcefollower/freqsamplingdelta'] = 10000. #sampling spacing for the frequencies (e.g., sample every 10kHz);
    if doTest in ['Simple']:
        strh5['rystare/sourcefollower/noise/flag'] = False
    elif  doTest in ['Advanced']:
        strh5['rystare/sourcefollower/noise/flag'] = True

    #dark current Offset Fixed Pattern Noise 
    strh5['rystare/darkoffset/NU/flag'] = True
    strh5['rystare/darkoffset/NU/model'] = 'Janesick-Gaussian'
    strh5['rystare/darkoffset/NU/spread'] = 0.0005 # percentage of (V_REF - V_SN)


    # Correlated Double Sampling (CDS)
    if doTest in ['Simple']:
        strh5['rystare/CDS/sampletosamplingtime'] = 0 #not used
    elif  doTest in ['Advanced']:
        strh5['rystare/CDS/sampletosamplingtime'] = 1e-6 #CDS sample-to-sampling time [sec].
    else:
        pass
    strh5['rystare/CDS/gain'] = 1. # CDS gain, [V/V], lower means amplify the noise.

    # Analogue-to-Digital Converter (ADC)
    strh5['rystare/ADC/num-bits'] = 12. # noise is more apparent on high Bits
    strh5['rystare/ADC/offset'] = 0. # Offset of the ADC, in DN
    strh5['rystare/ADC/nonlinearity/flag'] = False
    strh5['rystare/ADC/nonlinearity/ratio'] = 1.1

    #Sensor noises and signal visualisation
    strh5['rystare/flag/plots/doPlots'] = False
    strh5['rystare/flag/plots/plotLogs'] = False

    #For testing and measurements only:
    strh5['rystare/flag/darkframe'] = False # True if no signal, only dark

    #=============================================================================

    if strh5['rystare/flag/darkframe'].value:  # we have zero light illumination    
        imagehdffilename = 'data/image-Zero-256-256.hdf5'
    else:   # load an image, nonzero illumination
        imagehdffilename = 'data/image-Disk-256-256.hdf5'

    if pathtoimage is None:
        pathtoimage = os.path.dirname(__file__) + '/' + imagehdffilename

    imghd5 = ryfiles.open_HDF(pathtoimage)

    #images must be in photon rate irradiance units q/(m2.s)
    strh5['rystare/irradianceLux'] = imghd5['image/irradianceLux'].value
    strh5['rystare/signal/photonRateIrradianceNoNoise'] = imghd5['image/PhotonRateIrradianceNoNoise'].value
    strh5['rystare/signal/photonRateIrradiance'] = imghd5['image/PhotonRateIrradiance'].value
    strh5['rystare/pixelPitch'] = imghd5['image/pixelPitch'].value
    strh5['rystare/imageName'] = imghd5['image/imageName'].value
    strh5['rystare/imageFilename'] = imghd5['image/imageFilename'].value
    strh5['rystare/imageSizePixels'] = imghd5['image/imageSizePixels'].value
    strh5['rystare/wavelength'] = imghd5['image/wavelength'].value
    strh5['rystare/imageSizeRows'] = imghd5['image/imageSizeRows'].value
    strh5['rystare/imageSizeCols'] = imghd5['image/imageSizeCols'].value
    strh5['rystare/imageSizeDiagonal'] = imghd5['image/imageSizeDiagonal'].value

    #calculate the noise and final images
    strh5 = photosensor(strh5) # here the Photon-to-electron conversion occurred.

    with open('{}{}.txt'.format(prefix,outfilename), 'wt') as fo: 
        fo.write('{:26}, {:.5e}, {:.5e}\n'.format('SignalPhotonRateIrradiance',np.mean(strh5['rystare/signal/photonRateIrradiance'].value), np.var(strh5['rystare/signal/photonRateIrradiance'].value)))
        fo.write('{:26}, {:.5e}, {:.5e}\n'.format('signalphotonRateIrradianceNU',np.mean(strh5['rystare/signal/photonRateIrradianceNU'].value), np.var(strh5['rystare/signal/photonRateIrradianceNU'].value)))
        fo.write('{:26}, {:.5e}, {:.5e}\n'.format('signalelectronRateIrradiance',np.mean(strh5['rystare/signal/electronRateIrradiance'].value), np.var(strh5['rystare/signal/electronRateIrradiance'].value)))
        fo.write('{:26}, {:.5e}, {:.5e}\n'.format('SignalelectronRate',np.mean(strh5['rystare/signal/electronRate'].value), np.var(strh5['rystare/signal/electronRate'].value)))
        fo.write('{:26}, {:.5e}, {:.5e}\n'.format('signallightelectrons',np.mean(strh5['rystare/signal/lightelectrons'].value), np.var(strh5['rystare/signal/lightelectrons'].value)))
        fo.write('{:26}, {:.5e}, {:.5e}\n'.format('signalDark',np.mean(strh5['rystare/signal/darkcurrentelectrons'].value), np.var(strh5['rystare/signal/darkcurrentelectrons'].value)))
        fo.write('{:26}, {:.5e}, {:.5e}\n'.format('source_follower_noise',np.mean(strh5['rystare/sourcefollower/source_follower_noise'].value), np.var(strh5['rystare/sourcefollower/source_follower_noise'].value)))
        fo.write('{:26}, {:.5e}, {:.5e}\n'.format('SignalElectrons',np.mean(strh5['rystare/signal/electrons'].value), np.var(strh5['rystare/signal/electrons'].value)))
        fo.write('{:26}, {:.5e}, {:.5e}\n'.format('voltagebeforeSF',np.mean(strh5['rystare/signal/voltagebeforeSF'].value), np.var(strh5['rystare/signal/voltagebeforeSF'].value)))
        fo.write('{:26}, {:.5e}, {:.5e}\n'.format('voltagebeforecds',np.mean(strh5['rystare/signal/voltagebeforecds'].value), np.var(strh5['rystare/signal/voltagebeforecds'].value)))
        fo.write('{:26}, {:.5e}, {:.5e}\n'.format('voltagebeforeadc',np.mean(strh5['rystare/signal/voltagebeforeadc'].value), np.var(strh5['rystare/signal/voltage'].value)))
        fo.write('{:26}, {:.5e}, {:.5e}\n'.format('SignalVoltage',np.mean(strh5['rystare/signal/voltage'].value), np.var(strh5['rystare/signal/voltagebeforeadc'].value)))
        fo.write('{:26}, {:.5e}, {:.5e}\n'.format('SignalDN',np.mean(strh5['rystare/signal/DN'].value), np.var(strh5['rystare/signal/DN'].value)))

    if doPlots:
        lstimgs = ['rystare/signal/photonRateIrradiance','rystare/signal/photons','rystare/signal/electrons','rystare/signal/voltage',
                   'rystare/signal/DN','rystare/signal/light','rystare/signal/darkcurrentelectrons', 'rystare/detector/response/NU/value',
                   'rystare/darkresponse/NU/value','rystare/quantumEfficiency']
        # ryfiles.plotHDF5Images(strh5, prefix=prefix, colormap=mcm.gray,  lstimgs=lstimgs, logscale=strh5['rystare/flag/plots/plotLogs'].value) 
        ryfiles.plotHDF5Images(strh5, prefix=prefix, colormap=mcm.jet,  lstimgs=lstimgs, logscale=strh5['rystare/flag/plots/plotLogs'].value) 

    if doHisto:
        lstimgs = ['rystare/signal/photonRateIrradiance','rystare/signal/photons','rystare/signal/electrons','rystare/signal/voltage',
                   'rystare/signal/DN','rystare/signal/light','rystare/signal/darkcurrentelectrons',
                   'rystare/detector/response/NU/value','rystare/darkresponse/NU/value','rystare/quantumEfficiency']
        ryfiles.plotHDF5Histograms(strh5, prefix, bins=100, lstimgs=lstimgs)

    if doImages:
        lstimgs = ['rystare/signal/photonRateIrradiance','rystare/signal/photonRate', 'rystare/signal/photons','rystare/signal/electrons','rystare/signal/voltage',
                    'rystare/signal/DN','rystare/signal/light','rystare/signal/darkcurrentelectrons', 'rystare/noise/sn_reset/resetnoise','rystare/sourcefollower/source_follower_noise',
                    'rystare/detector/response/NU/value','rystare/darkresponse/NU/value','rystare/quantumEfficiency']
        ryfiles.plotHDF5Bitmaps(strh5, prefix, format='png', lstimgs=lstimgs)

    strh5.flush()
    strh5.close()

    return hdffilename

################################################################
def get_summary_stats(hdffilename):
    """Return a string with all the summary input and results data.

    Args:
        | hdffilename (string):  filename for input HDF file

    Returns:
        | Returns a string with summmary data.

    Raises:
        | No exception is raised.

    Author: CJ Willers

     """
    import StringIO
    
    output = StringIO.StringIO()

    if hdffilename is not None:
        strh5 = ryfiles.open_HDF(hdffilename)
        print(hdffilename)

        print('Image file name             : {}'.format(strh5['rystare/imageFilename'].value))
        print('Image name                  : {}'.format(strh5['rystare/imageName'].value))
        print('Sensor type                 : {} '.format(strh5['rystare/sensortype'].value))
        print('Pixel pitch                 : {} m'.format(strh5['rystare/pixelPitch'].value))
        print('Image size diagonal         : {} m'.format(strh5['rystare/imageSizeDiagonal'].value))
        print('Image size pixels           : {} '.format(strh5['rystare/imageSizePixels'].value))
        print('Fill factor                 : {}'.format(strh5['rystare/fillfactor'].value))
        print('Full well electrons         : {} e'.format(strh5['rystare/fullwellelectrons'].value))
        print('Integration time            : {} s'.format(strh5['rystare/integrationtime'].value))
        print('Wavelength                  : {} m'.format(strh5['rystare/wavelength'].value))
        print('Operating temperature       : {} K'.format(strh5['rystare/operatingtemperature'].value))
        print('Max image irradiance         : {} lux'.format(np.max(strh5['rystare/irradianceLux'].value)))
        print('{:28}: q/(m2.s) mean={:.5e}, var={:.5e}'.format('PhotonRateIrradianceNoNoise',np.mean(strh5['rystare/signal/photonRateIrradianceNoNoise'].value), np.var(strh5['rystare/signal/photonRateIrradianceNoNoise'].value)))
        print('{:28}: q/(m2.s) mean={:.5e}, var={:.5e}'.format('SignalPhotonRateIrradiance',np.mean(strh5['rystare/signal/photonRateIrradiance'].value), np.var(strh5['rystare/signal/photonRateIrradiance'].value)))
        print('{:28}: q/(m2.s) mean={:.5e}, var={:.5e}'.format('SignalPhotonsNU',np.mean(strh5['rystare/signal/photonRateIrradianceNU'].value), np.var(strh5['rystare/signal/photonRateIrradianceNU'].value)))
        print('{:28}: e mean={:.5e}, var={:.5e}'.format('signalLight',np.mean(strh5['rystare/signal/lightelectrons'].value), np.var(strh5['rystare/signal/lightelectrons'].value)))
        print('{:28}: e mean={:.5e}, var={:.5e}'.format('signalDarkNoNoise',np.mean(strh5['rystare/darkcurrentelectronsnonoise'].value), np.var(strh5['rystare/darkcurrentelectronsnonoise'].value)))
        print('{:28}: e mean={:.5e}, var={:.5e}'.format('signalDark',np.mean(strh5['rystare/signal/darkcurrentelectrons'].value), np.var(strh5['rystare/signal/darkcurrentelectrons'].value)))
        print('{:28}: e mean={:.5e}, var={:.5e}'.format('SignalElectrons',np.mean(strh5['rystare/signal/electrons'].value), np.var(strh5['rystare/signal/electrons'].value)))
        print('{:28}: v mean={:.5e}, var={:.5e}'.format('source_follower_noise',np.mean(strh5['rystare/sourcefollower/source_follower_noise'].value), np.var(strh5['rystare/sourcefollower/source_follower_noise'].value)))
        print('{:28}: v mean={:.5e}, var={:.5e}'.format('SignalVoltage',np.mean(strh5['rystare/signal/voltage'].value), np.var(strh5['rystare/signal/voltage'].value)))
        print('{:28}: DN mean={:.5e}, var={:.5e}'.format('SignalDN',np.mean(strh5['rystare/signal/DN'].value), np.var(strh5['rystare/signal/DN'].value)))
        print('{:28}: DN mean={:.5e}, var={:.5e}'.format('ADC gain',np.mean(strh5['rystare/ADC/gain'].value), np.var(strh5['rystare/ADC/gain'].value)))
        print('ADC                         : DN/v bits={} offset={} DN'.format(strh5['rystare/ADC/num-bits'].value,strh5['rystare/ADC/offset'].value ))
        print('Dark current figure-merit   : {} '.format(strh5['rystare/darkfiguremerit'].value))
        print('Quantum external efficiency : {} '.format(strh5['rystare/externalquantumeff'].value))
        print('Quantum yield               : {} '.format(strh5['rystare/quantumyield'].value))
        print('Sense node gain             : {} w/e'.format(strh5['rystare/sensenode/gain'].value))
        print('Sense reset Vref            : {} v'.format(strh5['rystare/sensenode/vrefreset'].value))
        print('Source follower gain        : {} '.format(strh5['rystare/sourcefollower/gain'].value))
        print('k1 constant                 : {} '.format(strh5['rystare/constants/k1'].value))
        if strh5['rystare/darkoffset/NU/flag'].value:
            print('Dark offset model           : {}'.format(strh5['rystare/darkoffset/NU/model'].value))
            print('Dark offset spread          : {} e'.format(strh5['rystare/darkoffset/NU/spread'].value))
        print('Dark response model         : {}'.format(strh5['rystare/darkresponse/NU/model'].value))
        print('Dark response seed          : {}'.format(strh5['rystare/darkresponse/NU/seed'].value))
        print('Dark response spread        : {} '.format(strh5['rystare/darkresponse/NU/spread'].value))
        print('Dark response neg lim flag  : {}'.format(strh5['rystare/darkresponse/NU/limitnegative'].value))
        if strh5['rystare/detector/response/NU/flag'].value:
            print('Detector response mode      : {}'.format(strh5['rystare/detector/response/NU/model'].value))
            print('Detector response seed      : {}'.format(strh5['rystare/detector/response/NU/seed'].value))
            print('Detector response spread    : {}'.format(strh5['rystare/detector/response/NU/spread'].value))
        print('Material Eg 0 K             : {} eV'.format(strh5['rystare/material/Eg-eV-0'].value))
        print('Material Eg                 : {} eV'.format(strh5['rystare/material/Eg-eV'].value))
        print('Material alpha              : {}  '.format(strh5['rystare/material/alpha'].value))
        print('Material beta               : {}  '.format(strh5['rystare/material/beta'].value))
        print('Sense node reset noise factr: {}  '.format(strh5['rystare/sensenode/resetnoise/factor'].value))
        print('Sense node reset kTC sigma  : {} v'.format(strh5['rystare/sensenode/ResetKTC-sigma'].value))
        print('Sense node capacitance      : {} fF'.format(1e15 * strh5['rystare/sensenode/capacitance'].value))
        print('Sense node signal full well : {} V'.format(strh5['rystare/sensenode/volt-fullwell'].value))
        print('Sense node signal minimum   : {} V'.format(strh5['rystare/sensenode/volt-min'].value))
        print('Sense node reset noise      : {} '.format(strh5['rystare/sensenode/resetnoise/flag'].value))
        print('CDS gain                    : {} '.format(strh5['rystare/CDS/gain'].value))
        print('CDS sample-to-sampling time : {} s'.format(strh5['rystare/CDS/sampletosamplingtime'].value))
        print('Delta induced modulation    : {} '.format(strh5['rystare/sourcefollower/deltaindmodulation'].value))
        print('Data clock speed            : {} Hz'.format(strh5['rystare/sourcefollower/dataclockspeed'].value))
        print('Frequency sampling delta    : {} Hz'.format(strh5['rystare/sourcefollower/freqsamplingdelta'].value))
        print('White noise density         : {} V/rtHz'.format(strh5['rystare/sourcefollower/whitenoisedensity'].value))
        print('Flicker corner              : {} Hz'.format(strh5['rystare/sourcefollower/flickerCornerHz'].value))
        print('{:28}: v mean={:.5e}, var={:.5e}'.format('Source follower sigma',np.mean(strh5['rystare/sourcefollower/sigma'].value), np.var(strh5['rystare/sourcefollower/sigma'].value)))
        
        strh5.flush()
        strh5.close()

    # Retrieve file contents
    contents = output.getvalue()

    # Close object and discard memory buffer 
    output.close()

    return contents




################################################################
################################################################
##
##

if __name__ == '__init__':
    pass

if __name__ == '__main__':

    import os.path

    import pyradi.ryfiles as ryfiles
    import pyradi.ryutils as ryutils

    doAll = True

    #----------  create test images ---------------------
    if doAll:
        pass

        numPixels = [256, 256]  # [ROW, COLUMN] size
        pixelPitch = [5e-6, 5e-6] # pixels pitch, in [m], [ROW, COLUMN] 

        #create image with all pixels set to zero
        create_HDF5_image(imageName='Zero', imtype='zeros', pixelPitch=pixelPitch, numPixels=numPixels)

        #create an image with a blurred disk
        create_HDF5_image(imageName='Disk', imtype='disk', pixelPitch=pixelPitch,
                numPixels=numPixels, fracdiameter=0.7, fracblurr=0.2, irrad_scale=0.1)

        #create an image with linear stairs
        create_HDF5_image(imageName='Stairslin-10', imtype='stairslin', pixelPitch=[5e-6, 5e-6],
                numPixels=[250, 250], irrad_scale=37.04e-3, irrad_min=74.08e-6)

        #create an image with log stairs
        create_HDF5_image(imageName='Stairslin-40', imtype='stairslin', pixelPitch=[5e-6, 5e-6],
                numPixels=[100,520], irrad_scale=37.04e-3, irrad_min=74.08e-6, steps=40)


    #----------  test the limitzero function ---------------------
    if doAll:
        import numpy as np
        import pyradi.ryplot as ryplot
        from scipy.stats import norm

        thres = 0.6
        rv = norm.rvs(loc=1, scale=0.4, size=10000)
        hist, bins = np.histogram(rv, bins=100)

        a = np.linspace(-10, 2, 100)
        with ryplot.savePlot(1,1,1,figsize=(12,4), saveName=['limitzero01.png']) as p:
            p.plot(1,a,limitzero(a, thres),'Clipping function','Input','Output')

        with ryplot.savePlot(2,1,1,figsize=(12,6), saveName=['limitzero02.png']) as q:
            q.plot(1,(bins[1:]+bins[:-1])/2.,hist /(bins[1]-bins[0]),'Histrograms','Value','Count',label=['No clip'])
            for sigma in [0.2, 0.3, 0.4, 0.5]:
            # for sigma in [0.1]:
                rv = norm.rvs(loc=1, scale=sigma, size=10000)
                histl, binsl = np.histogram(limitzero(rv,thres), bins=100)
                q.plot(1,(binsl[1:]+binsl[:-1])/2.,histl/(binsl[1]-binsl[0]),label=['Clipped, $\sigma$={}'.format(sigma)])   


    #----------  how to use example code ---------------------
    if True:
        hdffilename = run_example(doTest='Advanced')
        get_summary_stats(hdffilename)
        print(20*'- ')
        hdffilename = run_example(doTest='Simple')
        get_summary_stats(hdffilename)
