# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 12:03:50 2012

@author: Ricardos
"""

import matplotlib.pyplot as plt
import numpy as np

"""
################################################################################

This module was built to give the user a simple but reliable tool to simulate or 
to understand main parameters used to design a infrared photodetector. This tool
fits in the PYRAD pack as being the final toolkit to control all the IR radiation
cycle: from the emssion by a sourceof interest up to the detection by a photovoltaic 
IR photodetector.

All the work done in this module was based in classical equations found in the 
literature but the used references are:
    [1] DERENIAK,E.L. 'INFRARED DETECTORS AND SYSTEMS'. John Wiley and Sons, 1996.
    [2] ROGALSKI,A. 'INFRARED DETECTORS'. Gordon and Breach Science Publishers, 2000.
    
This code was built by
Ricardo Augusto Tavares Santos, D.Sc.
Instituto Tecnológico de Aeronáutica - Laboratório de Guerra Eletronica - Brazil

Nelis Willers, M.Sc.
DPSS/CSIR - South Africa

################################################################################
"""

# OPERATIONAL PARAMETERS
"""
In this step, all the parameters referring to the semiconductor material used 
to build the photodetector must be defined. Must be remembered that each material
has its own paramenters, ans each time the material is changed, the parameters
must eb changed as well.
In the same step, the important classical constants and semiconductors basic 
parameters are also calculated.
Finally, there is loop to establish if the radiance comes from a source or from 
only the background.
"""

T_source=0.1                             #source temperature in K
T_bkg=280                                #source temperature in K
T_detector=80                            # detector temperature in K
lambda_initial=1.0e-6                    # wavelength in meter- can start in 0
lambda_final=5.5e-6                      # wavelength in meter
A_det=(100e-6)**2                         # detector area in m2
A_source=0.000033                        # source area in m2
A_bkg=2*np.pi*(0.0055)**2                # bkg area in m2 - this area must be considered equals to the window area
d=0.01                                   # distance between source and detector or between window and detector
delta_f=1                              # measurement or desirable bandwidth - Hertz
nr=3.42                                  # refraction index of the semiconcuctor being simulated
E0=0.24                                  # semiconductor bandgap at room temp in Ev
alfa=6e-4                                # first fitting parameter for the Varshini's Equation
B=500                                    # second fitting parameter for the Varshini's Equation
n1=1                                     # refraction index of the air
n2=3.42                                  # refraction index of the semiconductor being analyzed
theta1=0.01                              # radiation incident angle in degrees - angle between the surface's normal and the radiation path
lx=5e-4                                  # detector thickness in meter
transipedance=10e7                       # transimpedance value used during the measurement
e_mob=120                                # electron mobility - m2/V.s
h_mob=1                                  # hole mobility - m2/V.s  
tau_e=1e-10                              # electron lifetime - s
tau_h=1e-6                               # hole lifetime - s
m0=9.11e-31                              # electron mass - kg
me=0.014*m0                              # used semiconductor electron effective mass 
mh=0.43*m0                               # used semiconductor hole effective mass 
na=1e18                                  # positive or negative dopping - m-3
b=1                                      # b=1 when the diffusion current is dominantand b=2 when the recombination current dominates - Derinaki's book page 251



# IMPORTANT CONSTANTS
q=1.6e-19                                # electron charge
etha2=0.45                               # quantum efficieny table 3.3 dereniak's book
h=6.626068e-34                           # planck cte. - joule.s
c=3e8                                    # light velocity - m/s
k=1.38e-23                               # boltzmann cte. - Joules/k
sigma=5.670373e-8                        # stefan-boltzmann cte - W/(m2 K4)
sigma_photon=1.52e15                     # boltzmann constant for photons- photons/(s.m2.K3)
epsilon=1                                # source emissivity



if T_source> T_bkg:
    r=np.sqrt(A_source/np.pi)            # source radius if it is a circle and plane source
else:
    r=np.sqrt(A_bkg/np.pi)               # source radius if it is a circle and plane source


# DEFINING THE WAVELENGTH VECTOR
lambda_vector=np.linspace(lambda_initial,lambda_final,1000)

# OPTICS TRANSMITTANCE PLUS THE ATMOSPHERIC TRANSMITTANCE
# IN THIS CASE THE SPECTRAL TRANSMITTANCE OR THE AVERAGE TRANSMITTANCE VALUE MUST
#BE USED OR ASSUMED EQUALS TO 1 IF IT IS DESCONSIDERED

final_trans=1

# DEFINIG THE BIAS TO BE USED IN THE SIMULATIONS (IF NECESSARY)

V=np.linspace(-250e-3,100e-3,np.size(lambda_vector))

# CALCULATING THE FREQUENCY RANGE GIVEN THE WAVELENGTH'S RANGE AND CHANGING IT TO
# ENERGY

f=c/lambda_vector                              # frequency in Hz
E=h*f                                          # Einstein's equation in Joules
E=E/q                                          # Energy in Ev


# CALCULATING THE SEMICONDUCTOR BANDGAP

Eg=(E0-(alfa*(T_detector**2/(T_detector+B))))  # Varshini's Equation to calculate the bandgap dependant on the temp - eV

# CALCULATING THE SEMICONDUCTOR'S OPTICAL REFLECTANCE

theta2=np.arcsin(np.sin(theta1)*n1/n2)         # Snell's equation
RS=np.abs((n1*np.cos(theta1)-n2*np.cos(theta2))/(n1*np.cos(theta1)+n2*np.cos(theta2)))**2   # Reflectance for perpendicular polarization
RP=np.abs((n1*np.cos(theta2)-n2*np.cos(theta1))/(n1*np.cos(theta1)+n2*np.cos(theta2)))**2   # Reflectance for parallel polarization
R=(RS+RP)/2;


################################################################################
# CALCULATING THE QUANTUM EFFICIENCY

def QE(E,Eg,lx,k,T_detector,lambda_vector):
    """
    Calculate the spectral quantum efficiency for a given semiconductor material.

    Args:
        |E: energy in Ev;
        |Eg: bandgap energy in Ev;
        |lx: detector thickness in m;
        |k: Boltzmann constant;
        |T_detector: detector's temperature in K;
        |lambda_vector: wavelength in m.
        
    Returns:
        | etha(etha_vector,absorption coefficient).
    """

    a_vector=[]
    etha_vector=[]
    for i in range(0,np.size(lambda_vector)):      # Calculating the absorption coefficient and QE
        if E[i]>Eg:
            a=(1.9e4*np.sqrt(E[i]-Eg))+800
            a_vector=np.r_[a_vector,a]             # Absorption coef - eq. 3.5 - [1] 
            etha1=(1-R)*(1-np.exp(-a*lx))           # QE - eq. 3.3 - [1]
            etha_vector=np.r_[etha_vector,etha1]
        else:
            a=800*np.exp((E[i]-Eg)/(k*T_detector))
            a_vector=np.r_[a_vector,a]
            etha1=(1-R)*(1-np.exp(-a*lx))
            etha_vector=np.r_[etha_vector,etha1]
            
    plt.figure()
    plt.plot(lambda_vector*1e6,a_vector)
    plt.xlabel('Wavelength (microns)'); plt.ylabel('Absorption Coefficient'); plt.title('Spectral Absorption Coefficient')
    plt.figure()
    plt.plot(lambda_vector*1e6,etha_vector)
    plt.xlabel('Wavelength (microns)'); plt.ylabel('Quantum Efficiency'); plt.title('Spectral Quantum Efficiency')
    plt.show()
            
    etha=np.mat('etha_vector;a_vector')

    return etha
 
    
################################################################################
# CALCULATING THE RADIANCE FROM A SOURCE AND FROM THE BACKGROUND
def Irradiance(epsilon,sigma_photon,T_source,T_bkg,lambda_vector,A_source,A_bkg,h,c,k):
    """
    This module calculates the quantity of energy produced by a source and the 
    background for a specific temperature in terms of number of photons and in 
    terms of Watt for the inverval among the wavelengths defined by lambda_inicial
    and lambda_final. Must be understood that this amount of energy is only a fraction
    from the total calculated using the Stefann-Boltzmann Law.
    
    After to calculate the Radiance, the irradiance calculation is done in order
    to be able to calculate the photocurrente generated by a photodetector. So,
    it is necessary to calculate how much energy is reaching the detector given 
    the energy emited by the source plus the backgorund and considering the
    distance among them and the detector. This is the irradiance calculation.
    
    All the equations used here are easilly found in the literature.
    
    
    
    Args:
        |epsilon: source emissivity (non-dimensional);
        |sigma_photon: boltzmann constant for photons in photons/(s.m2.K3);
        |T_source: source´s temperature in K;
        |T_kbg: background´s temperature in K;
        |lambda_vector: wavenlength in m;
        |A_source: soource´s area in m2;
        |A_bkg: background´s area in m2.
        |h: Planck´s constant in J.s;
        |c= light velocity in m/s;
        |k= Boltzmann´s constant in J/K;
    Returns:
        |Irradiance_vector(Detector_irradiance_source,Detector_irradiance_bkg,Detector_total)
        
    """

    # TOTAL PHOTON RADIANCE FROM THE SOURCE AND BACKGROUND

    L_source=epsilon*sigma_photon*T_source**3/np.pi     # source total radiance in one hemisphere (adapted for photons) - eq. 2.9 - Infrared Detectors and Systems - Dereniak and Boreman 
    L_bkg=epsilon*sigma_photon*T_bkg**3/np.pi           # bkg total radiance in one hemisphere (adapted for photons) - eq. 2.9 - Infrared Detectors and Systems - Dereniak and Boreman 

    # TOTAL EMITTED POWER FOR BKG AND SOURCE

    Total_source_energy=sigma*T_source**4               # Stefann-Boltzmann Law 
    Total_bkg_energy=sigma*T_bkg**4                     # Stefann-Boltzmann Law

    # EMITTED ENERGY RATE

    M_bkg_band_vector=((2*np.pi*h*c**2)/((lambda_vector)**5*(np.exp((h*c)/(lambda_vector*k*T_bkg))-1)))        # Planck Distribution - W/m2 - eq. 2.83 - Infrared Detectors and Systems - Dereniak and Boreman 
    M_source_band_vector=((2*np.pi*h*c**2)/((lambda_vector)**5*E,Eg,lx,k,T_detector(np.exp((h*c)/(lambda_vector*k*T_source))-1)))  # Planck Distribution - W/m2 - eq. 2.83 - Infrared Detectors and Systems - Dereniak and Boreman 

    M_total=(M_bkg_band_vector+M_source_band_vector)

    Band_source_energy=np.trapz(M_source_band_vector,lambda_vector)        # rate between the band and the total
    Band_bkg_energy=np.trapz(M_bkg_band_vector,lambda_vector)
    Total_energy=np.trapz(M_total,lambda_vector)

    plt.figure()
    plt.plot(lambda_vector*1e6,M_bkg_band_vector,'r',lambda_vector,M_bkg_band_vector,'b')
    plt.title('Background Radiation');plt.xlabel('Wavelength (m)');plt.ylabel('Radiance (W/m2)')
    plt.figure()
    plt.plot(lambda_vector*1e6,M_source_band_vector,'r',lambda_vector,M_source_band_vector,'b')
    plt.title('Source Radiation');plt.ylabel('Radiance (W/m2)');plt.xlabel('Wavelength (m)')
    plt.figure()
    plt.plot(lambda_vector*1e6,M_total)
    plt.title('Total Radiation');plt.xlabel('Wavelength (m)');plt.ylabel('Radiance (W/m2)')
    plt.show()

    rate_source=Band_source_energy/Total_source_energy
    rate_bkg=Band_bkg_energy/Total_bkg_energy

    # BAND PHOTON RADIANCE FROM THE SOURCE AND FROM THE BACKGROUND

    Radiance_final_source=L_source*rate_source
    Radiance_final_bkg=L_bkg*rate_bkg
    Radiance_total=Radiance_final_source+Radiance_final_bkg
    
    print ("Radiance_final_source=  "),Radiance_final_source
    print ("Radiance_final_bkg=  "),Radiance_final_bkg
    print ("Radiance_final_total=  "),Radiance_total
    
    
    # IRRADIANCE CALCULUS
    Detector_irradiance_source=(Radiance_final_source*A_source)/(d**2)     # photons/m2

    Detector_irradiance_bkg=(Radiance_final_bkg*A_bkg)/(d**2);epsilon,sigma_photon,T_source,T_bkg,lambda_vector,A_source,A_bkg,h,c,k

    Detector_irradiance_total=Detector_irradiance_source+Detector_irradiance_bkg;

    print ("Detector_irradiance_source=  "),Detector_irradiance_source
    print ("Detector_irradiance_bkg=  "),Detector_irradiance_bkg
    print ("Detector_irradiance_total=  "),Detector_irradiance_total

    Irradiance_vector=np.r_[Detector_irradiance_source,Detector_irradiance_bkg,Detector_irradiance_total]

    return Irradiance_vector  
    
    
################################################################################
# PHOTOCURRENT CALCULUS
def Photocurrent(E,Eg,lx,T_detector,lambda_vector,epsilon,sigma_photon,T_source,T_bkg,A_source,A_bkg,h,c,k,A_det,q,etha2):
    """
    The photocurrent is the the current generated by a photodetector given its 
    quantum efficiency, irradiance and area.
    
    The result is given in current or tension (dependant on the transipedance used
    in the calculation or measurement)
    
    Args:
        
        |E: energy in Ev;
        |Eg: bandgap energy in Ev;
        |lx: detector thickness in m;
        |T_detector: detector's temperature in K;
        |lambda_vector: wavenlength in m;
        |epsilon: source emissivity (non-dimensional);
        |sigma_photon: boltzmann constant for photons in photons/(s.m2.K3);
        |T_source: source´s temperature in K;
        |T_kbg: background´s temperature in K;
        |A_source: soource´s area in m2;
        |A_bkg: background´s area in m2.
        |h: Planck´s constant in J.s;
        |c= light velocity in m/s;
        |k= Boltzmann´s constant in J/K;
        |A_det: detector´s area in m2;
        |q: electron fundamental charge in C;
        |Etha: quantum efficiency;|k: Boltzmann constant;
        |etha2: average theoretical QE given by the literature;
        
    Returns:
        |Photocurrent_vector(I1_wide,I1_wide_thoeretical,V1_wide)
    """
    etha_vector=QE(E,Eg,lx,k,T_detector,lambda_vector)
    etha_vector=etha_vector[0,:]
    
    Irradiance_vector=Irradiance(epsilon,sigma_photon,T_source,T_bkg,lambda_vector,A_source,A_bkg,h,c,k)
    Detector_irradiance_total=Irradiance_vector[2]
    
    Detector_irradiance_bkg=Irradiance_vector[1]
    
    avg_etha=np.mean(etha_vector)

    I1_wide=avg_etha*Detector_irradiance_total*A_det*q      # Photocurrent - eq. 3.10 - Infrared Detectors and Systems - Dereniak and Boreman
    print ("I1_wide=  "),I1_wide

    I1_wide_theoretical=etha2*Detector_irradiance_total*A_det*q
    print ("I1_wide_theoretical=  "),I1_wide_theoretical
    
    I1_bkg=avg_etha*Detector_irradiance_bkg*A_det*q      # Photocurrent - eq. 3.10 - Infrared Detectors and Systems - Dereniak and Boreman
    print ("I1_bkg=  "),I1_bkg
    
    I1_bkg_theoretical=etha2*Detector_irradiance_bkg*A_det*q      # Photocurrent - eq. 3.10 - Infrared Detectors and Systems - Dereniak and Boreman
    print ("I1_bkg_theoretical=  "),I1_bkg_theoretical

    V1_wide=np.mean(transipedance*I1_wide)
    print ("V1_wide=  "),V1_wide
    
    Photocurrent_vector=np.r_[I1_wide,I1_wide_theoretical,I1_bkg,I1_bkg_theoretical]
    
    return Photocurrent_vector

e_mob=120                                # electron mobility - m2/V.s
h_mob=1                                  # hole mobility - m2/V.s  
tau_e=1e-10                              # electron lifetime - s
tau_h=1e-6                               # hole lifetime - s
m0=9.11e-31                              # electron mass - kg
me=0.014*m0                              # used semiconductor electron effective mass 
mh=0.43*m0                               # used semiconductor hole effective mass 
na=6e20                                  # positive or negative dopping - m-3
################################################################################
# IxV Characteristic Calculation
def IXV(e_mob,tau_e,me,mh,na,V,b,alfa,B,E,Eg,lx,T_detector,lambda_vector,epsilon,sigma_photon,T_source,T_bkg,A_source,A_bkg,h,c,k,A_det,q,etha2):
    """
    This module provides the diode curve for a given irradiance.
    
    Args:
        |e_mob: electron mobility in m2/V.s;
        |tau_e: electron lifetime in s;
        |me: electron effective mass in kg;
        |mh: hole effective mass in kg;
        |na: dopping concentration in m-3;
        |V: bias in V;
        |b: diode equation non linearity factor;
        |alfa: first fitting parameter for the Varshini's Equation
        |B: second fitting parameter for the Varshini's Equation
        |Eg: energy bandgap in Ev;
        |lx: detector thickness in m;
        |T_detector: detector's temperature in K;
        |lambda_vector: wavenlength in m;
        |epsilon: source emissivity (non-dimensional);
        |sigma_photon: boltzmann constant for photons in photons/(s.m2.K3);
        |T_source: source´s temperature in K;
        |T_kbg: background´s temperature in K;
        |A_source: soource´s area in m2;
        |A_bkg: background´s area in m2.
        |h: Planck´s constant in J.s;
        |c= light velocity in m/s;
        |k= Boltzmann´s constant in J/K;
        |A_det: detector´s area in m2;
        |q: electron fundamental charge in C;
        |Etha: quantum efficiency;|k: Boltzmann constant;
        |etha2: average theoretical QE given by the literature;
                
    Returns:
        |IXV_vector(I_vector1,I_vector2)
        
    """

    # SATURATION REVERSE CURRENT CALCULATION
    # note: in this procedure the bandgap calculation must be a specific equation for the used semiconductor. It means, for each different semiconductor material used the equation must be changed.

    Photocurrent_vector=Photocurrent(E,Eg,lx,T_detector,lambda_vector,epsilon,sigma_photon,T_source,T_bkg,A_source,A_bkg,h,c,k,A_det,q,etha2)
    I_bkg1=Photocurrent_vector[2]
    I_bkg2=Photocurrent_vector[3]

    # DIFFUSION lENGTH
    Le=np.sqrt(k*T_detector*e_mob*tau_e/q)                  # diffusion length - m - calculated using dereniak's book page 250
    print ("Le=  "),Le 

    # BANDGAP CALCULATION
    Eg=(0.24-(6e-4*(T_detector**2/(T_detector+500))))*q     # semiconductor bandgap - J - Varshini's Equation


    ni=(np.sqrt(4*(2*np.pi*k*T_detector/h**2)**3*np.exp(-Eg/(k*T_detector))*(me*mh)**1.5)) # intrinsic carrier concentration - dereniak`s book eq. 7.1 - m-3
    nd=(ni**2/na)                                           # donnors concentration in m-3
    
    # REVERSE CURRENT USING DERENIAK'S MODEL
    I0_deriniak=A_det*q*(Le/tau_e)*nd
    print ("I0_deriniak=  "),I0_deriniak

    # REVERSE CURRENT USING ROGALSKI'S MODEL
    De=k*T_detector*e_mob/q                                 # carrier diffusion coefficient - rogalski's book pg. 164
    I0_rogalski=q*De*nd*A_det/Le                            # reverse saturation current - rogalski's book eq. 8.118
    print ("I0_rogalski=  "),I0_rogalski

    # IxV CHARACTERISTIC
    I_vector=[]
    for i in range(0,np.size(lambda_vector)):
        I=I0_deriniak*(np.exp(q*V[i]/(b*k*T_detector))-1)     # diode equation from dereniak'book eq. 7.23
        I_vector=np.r_[I_vector,I]

    I_vector1=I_vector-I_bkg1
    I_vector2=I_vector-I_bkg2
    
    IXV_vector=np.mat['I_vector1;I_vector2']
    
    plt.figure()
    plt.plot(V,I_vector1,V,I_vector2)
    plt.title('IxV Characteristic');plt.ylabel('Current (A)');plt.xlabel('Bias (V)');plt.legend(('Model 1 - Average Calculated QE','Model 2 - QE from the literature'))
    plt.show()
    
    return IXV_vector
    

################################################################################
# REPONSIVITY PREDICTION
def Responsivity(lambda_vector,q,E,Eg,lx,k,T_detector):
    """
    Responsivity quantifies the amount of output seen per watt of radiant optical
    power input [1]. But, for this application it is interesting to define spectral
    responsivity that is the output per watt of monochromatic radiation. This is
    calculated in this function [1].
    
    Args:
        |lambda_vector: wavelength in m;
        |q: electron fundamental charge in C;
        |E: energy in Ev;
        |Eg: bandgap energy in Ev;
        |lx: detector thickness in m;
        |k: Boltzmann constant;
        |T_detector: detector's temperature in K;
        |lambda_vector: wavelength in m.
        
    
    Returns:
        |Responsivity_vector(total_responsivity,spectral_responsivity)
        
    """

    etha_vector=QE(E,Eg,lx,k,T_detector,lambda_vector)
    etha_vector=etha_vector[0,:]   

    Responsivity=[]
    for i in range(0,np.size(lambda_vector)):
        Responsivity_vector1=(q*lambda_vector[i]*etha_vector[i])/(h*c)              #  responsivity model from dereniak's book eq. 7.114
        Responsivity=np.r_[Responsivity,Responsivity_vector1]

    R_total=np.trapz(Responsivity,lambda_vector*1e4)
    
    plt.figure()
    plt.plot(lambda_vector*1e6,Responsivity)
    plt.title('Spectral Responsivity');plt.xlabel('Wavelength (microns)');plt.ylabel('Responsivity (A/W)');
    plt.show()
    
    Responsivity_vector=np.mat('R_total;Responsivity')
    
    return Responsivity_vector
    
    
################################################################################
# DETECTIVITY PREDICTION
def Detectivity(lambda_vector,q,E,Eg,lx,k,T_detector,epsilon,sigma_photon,T_source,T_bkg,A_source,A_bkg,h,c):
    """
    Detectivity can be interpreted as an SNR out of a detector when 1 W of radiant
    power is incident on the detector, given an area equal to 1 cm2 and noise-
    equivalent bandwidth of 1 Hz. The spectral responsivity is the rms signal-to-
    noise output when 1 W of monochromatic radiant flux is incident on 1 cm2
    detector area, within a noise-equivalent bandwidth of 1 Hz. Its maximum value
    (called the peak spectral D*) corresponds to the largest potential SNR.
    
    Args:
        |lambda_vector: wavelength in m;
        |q: electron fundamental charge in C;
        |E: energy in Ev;
        |Eg: bandgap energy in Ev;
        |lx: detector thickness in m;
        |k: Boltzmann constant;
        |T_detector: detector's temperature in K;
        |epsilon: source emissivity (non-dimensional);
        |sigma_photon: boltzmann constant for photons in photons/(s.m2.K3);
        |T_source: source´s temperature in K;
        |T_kbg: background´s temperature in K;
        |h: Planck´s constant in J.s;
        |c= light velocity in m/s;
        |k= Boltzmann´s constant in J/K;
        |A_source: soource´s area in m2;
        |A_bkg: background´s area in m2.
        
    Returns
        |Detectivity_vector(total_detectivity,spectral_detectivity)
        
    """
    
    etha_vector=QE(E,Eg,lx,k,T_detector,lambda_vector)
    etha_vector=etha_vector[0,:]

    Irradiance_vector=Irradiance(epsilon,sigma_photon,T_source,T_bkg,lambda_vector,A_source,A_bkg,h,c,k)
    Detector_irradiance_total=Irradiance_vector[2]

    Detectivity=[]
    for i in range(0,np.size(lambda_vector)):
        Detectivity_vector1=((lambda_vector[i])/(h*c))*np.sqrt(etha_vector[i]/(2*Detector_irradiance_total))    #  detectivity model from dereniak's book - changing for cm - eq. 7.134
        Detectivity=np.r_[Detectivity,Detectivity_vector1]

    D_total=np.trapz(Detectivity,lambda_vector*1e4)
    
    plt.figure()
    plt.plot(lambda_vector*1e6,Detectivity)
    plt.title('Spectral Detectivity');plt.xlabel('Wavelength (microns)');plt.ylabel('Detectivity (cm Hz^1/2 W^-1)')
    plt.show()
    
    Detectivity_vector=np.mat('D_total;Detectivity')
    
    return Detectivity_vector
    
    
################################################################################
# NEP PREDICITION
def NEP(lambda_vector,q,E,Eg,lx,k,T_detector,epsilon,sigma_photon,T_source,T_bkg,A_source,A_bkg,h,c):
    """
    NEP is the radiant power incident on detector that yields SNR=1 [1].
    
    Args:
        |lambda_vector: wavelength in m;
        |q: electron fundamental charge in C;
        |E: energy in Ev;
        |Eg: bandgap energy in Ev;
        |lx: detector thickness in m;
        |k: Boltzmann constant;
        |T_detector: detector's temperature in K;
        |epsilon: source emissivity (non-dimensional);
        |sigma_photon: boltzmann constant for photons in photons/(s.m2.K3);
        |T_source: source´s temperature in K;
        |T_kbg: background´s temperature in K;
        |h: Planck´s constant in J.s;
        |c= light velocity in m/s;
        |k= Boltzmann´s constant in J/K;
        |A_source: soource´s area in m2;
        |A_bkg: background´s area in m2.
        
    Returns
        |NEP(spectral_nep)
        
    """
    
    D=Detectivity(lambda_vector,q,E,Eg,lx,k,T_detector,epsilon,sigma_photon,T_source,T_bkg,A_source,A_bkg,h,c)
    Det=D[:,1]
    
    NEP=1/Det

    plt.figure()
    plt.plot(lambda_vector*1e6,NEP)
    plt.title('Spectral Noise Equivalent Power');plt.xlabel('Wavelength (microns)');plt.ylabel('NEP (W)');
    plt.show()
