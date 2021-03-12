#######################################################
# Capacitor Model Functions:                          #
# --------------------------                          #
#BASIC FUNCTIONS FOR CAPACITOR ENERGY CALCULATIONS    #
#THESE VALUES ARE AVAILABLE IN THE PROVIDED DATASETS  #
#BUT MAY BE REPRODUCED USING THE GEOMETRIC PROPERTIES #
#TO ENSURE CACULATIONS ARE CONSISTENT.                #
#######################################################


import numpy as np

##########################################
#Electrostatic Constants                 #
##########################################
X_air = .0005          #electric susceptability of air
e_0   = 8.854*1e-12    #electric permitivitty of vacuum
e     = (1.+X_air)*e_0 #electric permitivitty of cloudy air

##########################################
# Capacitor:                             #
##########################################
def capacitor_discharge(rho,d,a,sig,charge_type):
    '''
    Returns the energy neutralized for a capacitor. (Eq. 1)
       -) rho = critical space charge density
       -) d   = plate separation (separation between charge centers)
       -) a   = plate area (flash plan view area)
       -) e   = permitivitty of cloudy air
    '''
    if charge_type == 'surface':
        num = sig**2. * d * (a)
    elif charge_type == 'space':
        num = rho**2. * d**3. * a
    den = 2. * e
    return (num/den)

###########################################
# Runaway Breakeven Electric Field        #
###########################################
def e_br(z):
    '''
    Compute critical electric field from initiation altitudes. (Eq. 5)
    Note: does not assume RBE as breakdown mechanism, simply
          allows for approximation of electric field upon
          flash initiation.

          args: z = initiation altitude in km
                     MUST CONVERT FROM INDEX TO ALTITUDE
                     USING GRID SPACING:

                     z(km) = (z*125) * 1e-3

          returns: e_critical in kV/m
    '''
    rho_a = 1.208 * np.exp(-(z/8.4))
    #For threshold field of 201 kV/m [Marshall et al. 1995]
    #return 167.*rho_a

    #For Threshold field of 281 kV/m [Marshall et al. 2005]
    return 232.6*rho_a

###########################################
#COMPUTE ALL FLASH ENERGIES:              #
###########################################
def compute_energy(cap_dist,cap_area,flash_breakdown,charge_type):
    '''
    Compute flash energy for dataset.
    Critical space charge density is computed prior to energy calculation.
    (Eq. 6 and 7 in Text)
    '''
    sig_crit = (e*flash_breakdown) #between plates
#     sig_crit = (2*e*flash_breakdown) # at d/2 between plates.
    rho_crit     = (sig_crit)/cap_dist
    if charge_type == 'surface':
        flash_energy = capacitor_discharge(rho_crit,cap_dist,cap_area,sig_crit,'surface')
    else:
        flash_energy = capacitor_discharge(rho_crit,cap_dist,cap_area,sig_crit,'space')
    #Multiply by two to account for mirror charges
#     return(flash_energy*2)
    #No mirror charge if we assume capacitor is shielded, and z_init is too far from surface to have or need to 
    #account for the surface conducting plane.
    return(flash_energy) 
