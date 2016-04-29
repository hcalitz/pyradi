# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 12:03:50 2012

@author: Ricardos
"""

import matplotlib.pyplot as plt
import numpy as np

# OPERATIONAL PARAMETERS
T_source=0.1                            #source temperature in K
T_bkg=280                                #source temperature in K
T_detector=80                            # detector temperature in K
lambda_initial=1.0e-6                    # wavelength in meter- can start in 0
lambda_final=5.5e-6                      # wavelength in meter
A_det=(50e-6)**2                         # detector area in m2
A_source=0.000033                        # source area in m2
A_bkg=2*np.pi*(0.0055)**2                # bkg area in m2 - this area must be considered equals to the window area
d=0.01                                   # distance between source and detector or between window and detector
delta_f=1.9                              # measurement or desirable bandwidth - Hertz
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
na=6e20                                  # positive or negative dopping - m-3
b=1                                      # b=1 when the diffusion current is dominantand b=2 when the recombination current dominates - Derinaki's book page 251



# IMPORTANT CONSTANTS
q=1.6e-19                         # electron charge
etha2=0.35                               # InSb quantum efficieny table 3.3 dereniak's book
h=6.626068e-34                           # planck cte. - joule.s
c=3e8                                    # light velocity - m/s
k=1.38e-23                          # boltzmann cte. - Joules/k
sigma=5.670373e-8                        # stefan-boltzmann cte - W/(m2 K4)
sigma_photon=1.52e15                     # boltzmann constant for photons- photons/(s.m2.K3)
epsilon=0.9                                # source emissivity



if T_source>T_bkg:
    r=np.sqrt(A_source/np.pi)            # source radius if it is a circle and plane source
else:
    r=np.sqrt(A_bkg/np.pi)               # source radius if it is a circle and plane source


# DEFINING THE WAVELENGTH VECTOR
lambda_vector=np.linspace(lambda_initial,lambda_final,1000)

# OPTICAL TRANSMITTANCE
# IN THIS CASE THE SPECTRAL TRANSMITTANCE OR THE AVERAGE TRANSMITTANCE VALUE MUST BE USED

final_trans=1

# DEFINIG THE BIAS TO BE USED IN THE SIMULATIONS

V=np.linspace(-250e-3,100e-3,np.size(lambda_vector))

# CALCULATING THE QUANTUM EFFICIENCY

f=c/lambda_vector                        # frequency in Hz

Eg=(E0-(alfa*(T_detector**2/(T_detector+B))))  # Varshini's Equation to calculate the bandgap dependant on the temp - eV

E=h*f                                    # Einstein's equation in Joules
E=E/q                                    # Energy in Ev

theta2=np.arcsin(np.sin(theta1)*n1/n2)         # Snell's equation
RS=np.abs((n1*np.cos(theta1)-n2*np.cos(theta2))/(n1*np.cos(theta1)+n2*np.cos(theta2)))**2   # Reflectance for perpendicular polarization
RP=np.abs((n1*np.cos(theta2)-n2*np.cos(theta1))/(n1*np.cos(theta1)+n2*np.cos(theta2)))**2   # Reflectance for parallel polarization
R=(RS+RP)/2;


a_vector=[];
etha_vector=[]
for i in range(0,np.size(lambda_vector)):      # Calculating the absorption coefficient and QE
    if E[i]>Eg:
        a=(1.9e4*np.sqrt(E[i]-Eg))+800
        a_vector=np.r_[a_vector,a]             # Absorption coef - eq. 3.5 - Infrared Detectors and Systems - Dereniak and Boreman 
        etha=(1-R)*(1-np.exp(-a*lx))           # QE - eq. 3.4 - Infrared Detectors and Systems - Dereniak and Boreman
        etha_vector=np.r_[etha_vector,etha]
    else:
        a=800*np.exp((E[i]-Eg)/(k*T_detector))
        a_vector=np.r_[a_vector,a]
        etha=(1-R)*(1-np.exp(-a*lx))
        etha_vector=np.r_[etha_vector,etha]
 
plt.figure()
plt.plot(lambda_vector*1e6,a_vector)
plt.xlabel('Wavelength (microns)'); plt.ylabel('Absorption Coefficient'); plt.title('Spectral Absorption Coefficient')
plt.figure()
plt.plot(lambda_vector*1e6,etha_vector)
plt.xlabel('Wavelength (microns)'); plt.ylabel('Quantum Efficiency'); plt.title('Spectral Quantum Efficiency')
plt.show()

# CALCULATING THE RADIANCE FROM A SOURCE AND FROM THE BACKGROUND
# VERY OFTEN, WHILE MEASURING ANY DETECTOR, THE DETECTOR IS INSERTED IN A DEWAR IN ORDER TO OBTAIN THE CORRECT TEMPERATURE
# AND VACUUM PARAMETERS. THIS DEWAR IS COMPOSED BY A METAL TUBE AND A OPTICAL WINDOWN (NORMALLY MADE USING SAPPHIRE OR SILICON).
# THIS WINDOWN MUST BU TRANSPARENT (TRANSMITTANCE ALMOST EQUALS TO 1 IN THE DESIRED/ANALYZED WAVELENGTHS), AND WILL DETERMINE
# THE BACKGROUND EMITTING AREA TO BE CONSIDERED IN THE CALCULUS. IN THIS CASE, THE WINDOWN AREA WILL BE THE BACKGROUND EMITTING AREA,
# AND ITS DISTANCE FROM THE DETECTOR WILL BE THE BACKGROUND DISTANCE.

# TOTAL PHOTON RADIANCE FROM THE SOURCE AND BACKGROUND

L_source=epsilon*sigma_photon*T_source**3/np.pi     # source total radiance in one hemisphere (adapted for photons) - eq. 2.9 - Infrared Detectors and Systems - Dereniak and Boreman 
L_bkg=epsilon*sigma_photon*T_bkg**3/np.pi           # bkg total radiance in one hemisphere (adapted for photons) - eq. 2.9 - Infrared Detectors and Systems - Dereniak and Boreman 


# TOTAL EMITTED POWER FOR BKG AND SOURCE

Total_source_energy=sigma*T_source**4               # Stefann-Boltzmann Law 
Total_bkg_energy=sigma*T_bkg**4                     # Stefann-Boltzmann Law

print ("Total_source_energy=  "),Total_source_energy
print ("Total_bkg_energy=  "),Total_bkg_energy


# EMITTED ENERGY RATE

M_bkg_band_vector=((2*np.pi*h*c**2)/((lambda_vector)**5*(np.exp((h*c)/(lambda_vector*k*T_bkg))-1)))        # Planck Distribution - W/m2 - eq. 2.83 - Infrared Detectors and Systems - Dereniak and Boreman 
M_source_band_vector=((2*np.pi*h*c**2)/((lambda_vector)**5*(np.exp((h*c)/(lambda_vector*k*T_source))-1)))  # Planck Distribution - W/m2 - eq. 2.83 - Infrared Detectors and Systems - Dereniak and Boreman 

M_total=(M_bkg_band_vector+M_source_band_vector)

Band_source_energy=np.trapz(M_source_band_vector,lambda_vector)        # rate between the band and the total
Band_bkg_energy=np.trapz(M_bkg_band_vector,lambda_vector)
Total_energy=np.trapz(M_total,lambda_vector)

print ("Band_source_energy=  "),Band_source_energy
print ("Band_bkg_energy=  "),Band_bkg_energy
print ("Total_energy=  "),Total_energy

rate_source=Band_source_energy/Total_source_energy
rate_bkg=Band_bkg_energy/Total_bkg_energy

print ("rate_source=  "),rate_source
print ("rate_bkg=  "),rate_bkg


# BAND PHOTON RADIANCE FROM THE SOURCE AND FROM THE BACKGROUND

Radiance_final_source=L_source*rate_source
Radiance_final_bkg=L_bkg*rate_bkg
Radiance_total=Radiance_final_source+Radiance_final_bkg

print ("Radiance_final_source=  "),Radiance_final_source
print ("Radiance_final_bkg=  "),Radiance_final_bkg
print ("Radiance_final_total=  "),Radiance_total

# IRRADIANCE CALCULUS
Detector_irradiance_source=(Radiance_final_source*A_source)/(d**2)     # photons/m2

Detector_irradiance_bkg=(Radiance_final_bkg*A_bkg)/(d**2);

Detector_irradiance_total=Detector_irradiance_source+Detector_irradiance_bkg;

print ("Detector_irradiance_source=  "),Detector_irradiance_source
print ("Detector_irradiance_bkg=  "),Detector_irradiance_bkg
print ("Detector_irradiance_total=  "),Detector_irradiance_total

# PHOTOCURRENT CALCULUS
avg_etha=np.mean(etha_vector)

I1_wide=avg_etha*Detector_irradiance_total*A_det*q
print ("I1_wide=  "),I1_wide

I1_wide_theoretical=etha2*Detector_irradiance_total*A_det*q
print ("I1_wide_theoretical=  "),I1_wide_theoretical

I_bkg1=avg_etha*Detector_irradiance_bkg*A_det*q
I_bkg2=etha2*Detector_irradiance_bkg*A_det*q

print ("I_bkg1=  "),I_bkg1
print ("I_bkg2=  "),I_bkg2


V1_wide=np.mean(transipedance*I1_wide);

# SATURATION REVERSE CURRENT CALCULATION
# note: in this procedure the bandgap calculation must be a specific equation for the used semiconductor. It means, for each different semiconductor material used the equation must be changed.
Le=np.sqrt(k*T_detector*e_mob*tau_e/q)                  # diffusion length - m - calculated using dereniak's book page 250
print ("Le=  "),Le 
Eg=(0.24-(6e-4*(T_detector**2/(T_detector+500))))*q     # semiconductor bandgap - J
ni=(np.sqrt(4*(2*np.pi*k*T_detector/h**2)**3*np.exp(-Eg/(k*T_detector))*(me*mh)**1.5)) # intrinsic carrier concentration - dereniak`s book eq. 7.1 - m-3
nd=(ni**2/na)                                           # donnors concentration in m-3
print ("nd=  "),nd

# REVERSE CURRENT USING DERENIAK'S MODEL
I0_deriniak=A_det*q*(Le/tau_e)*nd                       # reverse saturation current - dereniak's book eq. 7.34 - Ampère
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

# REPONSIVITY PREDICTION
Responsivity=[]
for i in range(0,np.size(lambda_vector)):
    Responsivity_vector=(q*lambda_vector[i]*etha_vector[i])/(h*c)              #  responsivity model from dereniak's book eq. 7.114
    Responsivity=np.r_[Responsivity,Responsivity_vector]

R_total=np.trapz(Responsivity,lambda_vector*1e4) 
print ("Total Responsivity=  "),R_total

# DETECTIVITY PREDICTION
Detectivity=[]
for i in range(0,np.size(lambda_vector)):
    Detectivity_vector=((lambda_vector[i])/(h*c))*np.sqrt(etha_vector[i]/(2*Detector_irradiance_bkg))    #  detectivity model from dereniak's book - changing for cm - eq. 7.134
    Detectivity=np.r_[Detectivity,Detectivity_vector]

D_total=np.trapz(Detectivity,lambda_vector*1e4)
print ("Total Detectivity=  "),D_total

# NEP PREDICITION
NEP=1/Detectivity;

plt.figure()
plt.plot(lambda_vector*1e6,M_bkg_band_vector,'r',lambda_vector,M_bkg_band_vector,'b')
plt.title('Background Radiation');plt.xlabel('Wavelength (m)');plt.ylabel('Radiance (W/m2)')
plt.figure()
plt.plot(lambda_vector*1e6,M_source_band_vector,'r',lambda_vector,M_source_band_vector,'b')
plt.title('Source Radiation');plt.ylabel('Radiance (W/m2)');plt.xlabel('Wavelength (m)')
plt.figure()
plt.plot(lambda_vector*1e6,M_total)
plt.title('Total Radiation');plt.xlabel('Wavelength (m)');plt.ylabel('Radiance (W/m2)')
plt.figure()
plt.plot(V,I_vector1,V,I_vector2)
plt.title('IxV Characteristic');plt.ylabel('Current (A)');plt.xlabel('Bias (V)')
plt.figure()
plt.plot(lambda_vector*1e6,Responsivity)
plt.title('Spectral Responsivity');plt.xlabel('Wavelength (microns)');plt.ylabel('Responsivity (A/W)');
plt.figure()
plt.plot(lambda_vector*1e6,Detectivity)
plt.title('Spectral Detectivity');plt.xlabel('Wavelength (microns)');plt.ylabel('Detectivity (cm Hz^1/2 W^-1)')
plt.figure()
plt.plot(lambda_vector*1e6,NEP)
plt.title('Spectral Noise Equivalent Power');plt.xlabel('Wavelength (microns)');plt.ylabel('NEP (W)');
plt.show()
