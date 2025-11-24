import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lng 
from scipy import integrate

# ============================
# Geometrical data
# ============================

D_barr_ext = 2.5       #m
D_vess_int = 3.0       #m
t_th_ins = 5.0         #cm 
k_th_ins = 1.4         #W/mK

# ============================
# Primary fluid
# ============================

T_in = 214                 #°C
T_out_avg = 254            #°C
T_out_max = 270            #°C
P_int = 75                 #bar
m_flr = 3227               #kg/s
Cp = 4534                  #J/(kg·K)
rho = 852.5                #kg/m³
mu = 1.259e-4              #Pa·s
k = 0.658                  #W/(m·K)

# ============================
# Containment (CPP) water
# ============================

T_cpp = 70                      #°C
P_cpp = 75                      #bar
Cp_cpp = 4172.5                 #J/(kg·K)
rho_cpp = 981.2                 #kg/m³
mu_cpp = 4.06e-4                #Pa·s
k_cpp = 0.666                   #W/(m·K)
beta_cpp = 5.57e-4             #1/K
DeltaT = 30                     #K

# ============================
# Steel properties
# ============================

E = 177                        #GPa
nu = 0.3                  
alpha_l = 1.7e-5               #1/K
k_st = 48.1                    #W/(m·K)
mu_st = 24                     #1/m
sigma_y = np.array([240,232.5,222,216,210,204,199.5,195,190.5,186,181.5,177,171,165,157.5,147])
sigma_in = np.array([160,155,148,144,140,136,133,130,127,124,121,118,114,110,105,98])

# ============================
# Radiation source
# ============================

Phi_0 = 1.5e13                 #photons/(cm²·s)
E_y = 6.0                      #MeV
E_y_J = E_y * 1.60218e-13      #Joules
B = 1.4                        #Build-up factor

# ============================
# Computed additional data
# ============================

t = 0.22                                    #m #First guess
R_int = D_vess_int/2                        #m
R_ext = R_int + t                           #m
G = E/(2*(1+nu))                            #MPa
rho_ii = (R_ext**2)/(R_ext**2 - R_int**2)
rho_i = (R_int**2)/(R_ext**2 - R_int**2)
Phi_0 = Phi_0 * 1e4                         #photons/(m²·s)
v = m_flr/(rho*np.pi*(D_vess_int**2)/4)     #m/s
Mar_criterion = R_int/t                     #Satisfied

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PURELY MECHANICAL PROBLEM
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ============================
# Mariotte Solution for a thin-walled cylinder and sphere (R_int = R_ext = R)
# ============================

sigma_rM_cyl = -P_int/2                        #Compressive
sigma_tM_cyl = R_int*P_int/t                   
sigma_zM_cyl = R_int*P_int/(2*t)

sigma_tM_sph = R_int*P_int/(2*t)

# ============================ 
# Comparison stress - Guest-Tresca Theory
# ============================
sigma_cTR_M = np.max([abs(sigma_tM_cyl - sigma_rM_cyl), abs(sigma_zM_cyl - sigma_rM_cyl), abs(sigma_tM_cyl - sigma_zM_cyl)])
print("\nGuest-Tresca Equivalent Stress - Mariotte solution: %.3f Mpa" %sigma_cTR_M)

# ============================ 
# Comparison stress - Von Mises Theory
# ============================
sigma_cVM_M = np.sqrt(0.5*((sigma_rM_cyl - sigma_tM_cyl)**2 + (sigma_tM_cyl - sigma_zM_cyl)**2 + (sigma_zM_cyl - sigma_rM_cyl)**2))
print("Von Mises Equivalent Stress - Mariotte solution: %.3f Mpa" %sigma_cVM_M)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ============================ 
# Hydrostatic Stress 
# ============================
sigma_h = (sigma_rM + sigma_tM + sigma_zM)/3
if (sigma_h <= (8*sigma_y)/9):
    print("No Plastic Deformation due to Hydrostatic Stress: %.3f Mpa" %sigma_h)

# ============================ 
# First guess on Stress intensity
# ============================ 
sigma_int_1st = min((2*sigma_y)/3, sigma_ult/3) #BCC Mild steel hyp.  ===> Sigma_y is now a vector, and sigma_ult should be too: how is this computed then? I might have an idea from a file on WeBeep
print("First guess on allowable stress intensity: %.3f Mpa" %sigma_int_1st)
sigma_int_2nd = 117.9 #Mpa #Design Stress Intensity at 400°F (204.4°C) for carbon steel Line No. 13 (SA-333) from ASME code pag.274

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#tmp = 0
r = np.linspace(R_int, R_ext, 1000)

# ============================ 
# Lamé Solution 
# ============================ 
sigma_rL = lambda r: -(((R_int**2*R_ext**2)/(R_ext**2 - R_int**2))*(P_int-P_cpp)/r**2)+(P_int*(R_int**2))-(P_cpp*(R_ext**2))/(R_ext**2 - R_int**2)
sigma_tL = lambda r: (P_int*(R_int**2))-(P_cpp*(R_ext**2))/(R_ext**2 - R_int**2) + (((R_int**2*R_ext**2)/(R_ext**2 - R_int**2))*(P_int-P_cpp)/r**2)

flag_eps = int(input("Enter the stress/strain condition (1: Plane Stress, 0: Plane Strain, -1: Hydrostatic): "))
def sigma_zL_fun(flag_eps):
    if flag_eps == 1:                           #Plane Stress
        eps_z_a = 2*nu*rho_ii*P_cpp/E
        eps_z_b = -2*nu*rho_i*P_int/E
    elif flag_eps == 0:                         #Plane Strain
        eps_z_a = 0
        eps_z_b = 0
    elif flag_eps == -1:                        #Hydrostatic
        eps_z_a = (2*nu-1)*rho_ii*P_cpp/E
        eps_z_b = (1-2*nu)*rho_i*P_int/E
         
    sigma_zL_a = E*eps_z_a - 2*nu*rho_ii*P_cpp  #a) P_int = 0
    sigma_zL_b = E*eps_z_b + 2*nu*rho_i*P_int   #b) P_cpp = 0
    return sigma_zL_a + sigma_zL_b              #Superposition Principle
sigma_zL = sigma_zL_fun(flag_eps)

# ============================ 
# Comparison stress - Guest-Tresca Theory 
# ============================ 
sigma_cTR_L = np.max([abs(sigma_tL(r) - sigma_rL(r)), abs(sigma_zL - sigma_rL(r)), abs(sigma_tL(r) - sigma_zL)])
print("\nGuest-Tresca Equivalent Stress - Lamé solution: %.3f Mpa" %sigma_cTR_L)

# ============================ 
# Comparison stress - Von Mises Theory 
# ============================ 
sigma_cVM_L = max(np.sqrt(0.5*((sigma_rL(r) - sigma_tL(r))**2 + (sigma_tL(r) - sigma_zL)**2 + (sigma_zL - sigma_rL(r))**2))) #The max should be the worst case, in theory
print("Von Mises Equivalent Stress - Lamé solution: %.3f Mpa" %sigma_cVM_L)

# ======================================
# Plotting the stress profiles: Mariotte
# ======================================
plt.figure(figsize=(15,10))
plt.subplot(1,2,1)
plt.xlim(R_int, R_ext)
plt.axhline(y=sigma_rM_cyl, label='Radial (r) Stress Mariotte', color='blue')
plt.axhline(y=sigma_tM_cyl, label=r'Hoop ($\theta$) Stress Mariotte', color='red')
plt.axhline(y=sigma_zM_cyl, label='Axial (z) Stress Mariotte', color='green')
plt.xlabel('Radius (m)')
plt.ylabel('Stress (MPa)')
plt.title('Stress Distribution in a thin-walled cylinder - Mariotte Solution')
plt.legend()
plt.grid()

# ======================================
# Plotting the stress profiles: Lamé
# ======================================
plt.subplot(1,2,2)
plt.plot(r, sigma_rL(r), label='Radial (r) Stress Lamé')
plt.plot(r, sigma_tL(r), label=r'Hoop ($\theta$) Stress Lamé')
plt.axhline(y=sigma_zL, color='green', label='Axial (z) Stress Lamé')
plt.xlabel('Radius (m)')
plt.ylabel('Stress (MPa)')
plt.title('Stress Distribution in the cylinder wall - Lamé Solution')
plt.legend()
plt.grid()
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PURELY THERMAL PROBLEM - POWER IMPOSED (should be right????) - NO THERMAL SHIELD
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ======================================
# Radiation-induced heating
# ======================================
Phi = lambda r: Phi_0*np.exp(-mu_st*(r-R_int))     #1/(m²·s)
I = lambda r: E_y_J*Phi(r)*B                       #W/(m²)
q_0 = B*Phi_0*E_y_J*mu_st                          #W/(m³)
q_iii = lambda r: q_0*np.exp(-mu_st*(r-R_int))     #W/(m³)

# ======================================
# Plotting the volumetric heat source profile 
# ======================================
plt.figure(figsize=(10,10))
plt.plot(r, q_iii(r), 'k', label='Radial (r) Volumetric heat source Profile')
plt.axvline(x=r[0], color='red', linestyle='dashed', linewidth='0.5')
plt.axhline(y=q_iii(r[0]), color='red', linestyle='dashed', linewidth='0.5')
plt.plot(r[0], q_iii(r[0]), '--or', label='Vessel Inner Surface Value')
plt.axvline(x=r[-1], color='red', linestyle='dashed', linewidth='0.5')
plt.axhline(y=q_iii(r[-1]), color='red', linestyle='dashed', linewidth='0.5')
plt.plot(r[-1], q_iii(r[-1]), '--or', label='Vessel-Insulation Interface Value')
plt.xlabel('Radius (m)')
plt.ylabel(r'$q_0$ (W/m$^3$)')
plt.title('Volumetric heat source profile across the vessel wall')
plt.legend()
plt.grid()
plt.show()

print("\nVolumetric heat source at the vessel inner surface: %.3f W/m³" %q_iii(r[0]))
print("Volumetric heat source at the vessel-insulation interface: %.3f W/m³" %q_iii(r[-1]))

# ======================================
# Dimensionless numbers and heat transfer coefficients
# ======================================
Pr = (Cp*mu)/k                                                                              #Prandtl number                                        
Re = (rho*v*D_vess_int)/mu                                                                  #Reynolds number
Nu_1 = 0.023*(Re**0.8)*(Pr**0.4)                                                            #Dittus-Boelter equation for forced convection
Gr = (rho**2)*9.81*beta_cpp*DeltaT*(D_vess_int**3)/(mu**2)                                  #Grashof number
Nu_2 = 0.13*((Gr*Pr)**(1/3))                                                                #McAdams correlation for natural convection
h_1 = (Nu_1*k)/D_vess_int                                                                   #W/(m²·K)
h_2 = (Nu_2*k)/D_vess_int                                                                   #W/(m²·K)
R_th_2 = (1/(2*np.pi*k_th_ins))*np.log((R_ext + t_th_ins)/(R_ext))                          #Thermal Resistance of the insulation layer
R_th_out = 1/(h_2*2*np.pi*(R_ext + t_th_ins))                                               #Thermal Resistance of the outer convective layer
u_2 = 1/(R_th_2 + R_th_out)                                                                 #Overall heat transfer coefficient outside the vessel

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
NB: The thermal resistances are computed per unit length of the vessel
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
q_0_flag = int(input("\nIs there a volumetric heat source q_0? (1: Yes, 0: No): "))
q_0 = q_0*q_0_flag

# ======================================
# T profile constants for the vessel: general and under adiabatic outer wall approximation (dT/dx = 0 at r = R_ext)
# ======================================
adiab_flag = int(input("\nApply Adiabatic Outer Wall approximation? (1: Yes, 0: No): "))

if adiab_flag == 0:
    C1 = ((q_0/(k_st*mu_st**2))*(np.exp(-mu_st*t)-1)-(q_0/mu_st)*((1/h_1)+(np.exp(-mu_st*t)/u_2))-(T_in-T_cpp))/(t+(k_st/h_1)+(k_st/u_2))
elif adiab_flag == 1:
    C1 = -((q_0/(k_st*mu_st))*np.exp(-mu_st*t))
C2 = T_in + (q_0/(h_1*mu_st)) + C1*(k_st/h_1) + (q_0/(k_st*mu_st**2))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
I used T_in as T1 when computing C1 and C2, but T of the primary fluid changes along z...Three more logical ways to proceed would be:

1) Use a mean logarithmic temperature difference, which has been computed, instead of T_in - T_cpp. This would allow to account for the 
   change of T along z in an approximate way.
2) Discretize the thermal problem along z, obtaining an array of T1 values which will result in an array of C1 and C2 values. Then,
   compute T(r) for each discretized z and solve the problem in a 2D scheme.
3) Check which one is the worst case scenario (which results in the highest thermal stresses) between T_in and T_out_max, and use that 
   value as T1. This would be a conservative approach, but would not account for the real T profile along z.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# ======================================
# T profiles across the vessel wall, average Ts, maxima and their positions
# ======================================
T_vessel = lambda r: -((q_0/(k_st*mu_st**2))*np.exp(-mu_st*(r-R_int))) + C1*(r-R_int) + C2
T_vessel_avg = (1/t)*integrate.quad(T_vessel, R_int, R_ext)[0]
T_vessel_max = max(T_vessel(r))
r_T_vessel_max = r[np.argmax(T_vessel(r))]
#T_vessel_avg_2 = (q_0/(k_st*mu_st**2))*((np.exp(-mu_st*t)-1)/(mu_st*t))+ C1*(t/2) + C2                                          #Analytical Integration Result

# ======================================
# Thermal power fluxes (kW/m²) on the inner and outer vessel surface
# ======================================
DeltaT_LM1 = ((T_in-T_vessel(r[0]))-(T_out_avg-T_vessel(r[0])))/(np.log((T_in-T_vessel(r[0]))/(T_out_avg-T_vessel(r[0]))))       #Log Mean Temperature Difference to account for T change along z, instead of just using Tin-T_wall
q_s1 = h_1*DeltaT_LM1/1000                                                                                                       #kW/m²
q_s2 = u_2*(T_vessel(r[-1])-T_cpp)/1000                                                                                          #kW/m²

print("\nThermal power flux on the inner vessel surface: %.3f kW/m²" %q_s1)
print("Thermal power flux on the outer vessel surface: %.3f kW/m²" %q_s2)

# ======================================
# Plotting the wall T profiles
# ======================================
if adiab_flag == 0:
    print("\nAverage Vessel Temperature (numerical integration): %.3f °C" %T_vessel_avg)
    #print("Average Vessel Temperature (analytical integration): %.3f °C" %T_vessel_avg_2)
    print("Maximum Vessel Temperature: %.3f °C at r = %.3f m" %(T_vessel_max, r_T_vessel_max))

    plt.figure(figsize=(10,10))
    plt.plot(r, T_vessel(r), label='Radial (r) T Profile')
    plt.axvline(x=r_T_vessel_max, color='red', linestyle='dashed', linewidth='0.5')
    plt.axhline(y=T_vessel_max, color='red', linestyle='dashed', linewidth='0.5')
    plt.axvline(x=R_int, color='black', linewidth='1.5', label='Vessel Inner Surface')
    plt.axvline(x=R_ext, color='black', linewidth='1.5', label='Vessel Outer Surface')
    plt.plot(r_T_vessel_max, T_vessel_max,'--or',label='Max T')
    plt.axhline(y=T_vessel_avg, color='green', label='Wall Average T')
    plt.xlabel('Radius (m)')
    plt.ylabel('T (K)')
    plt.title('Wall Temperature Profile, Average and Maximum ')
    plt.legend()
    plt.grid()
    plt.show()

elif adiab_flag == 1:
    print("\nAverage Vessel Temperature under Adiabatic Outer Wall approximation (numerical integration): %.3f °C" %T_vessel_avg)
    #print("Average Vessel Temperature under Adiabatic Outer Wall approximation (analytical integration): %.3f °C" %T_vessel_avg_2)
    print("Maximum Vessel Temperature under Adiabatic Outer Wall approximation: %.3f °C at r = %.3f m" %(T_vessel_max, r_T_vessel_max))

    # ======================================
    # Under Adiabatic Outer Wall Approximation
    # ======================================
    plt.figure(figsize=(10,10))
    plt.plot(r, T_vessel(r), label='Radial (r) T Profile')
    plt.axvline(x=r_T_vessel_max, color='red', linestyle='dashed', linewidth='0.5')
    plt.axhline(y=T_vessel_max, color='red', linestyle='dashed', linewidth='0.5')
    plt.axvline(x=R_int, color='black', linewidth='1.5', label='Vessel Inner Surface')
    plt.axvline(x=R_ext, color='black', linewidth='1.5', label='Vessel Outer Surface')
    plt.plot(r_T_vessel_max, T_vessel_max,'--or', label='Max T')
    plt.axhline(y=T_vessel_avg, color='green', label='Wall Average T')
    plt.xlabel('Radius (m)')
    plt.ylabel('T (K)')
    plt.title('Wall Temperature Profile, Average and Maximum under AOW Approximation ')
    plt.legend()
    plt.grid()
    plt.show()

# ======================================
# Thermal stresses computation
# ======================================
sigma_r_th = lambda r: (E*alpha_l/(1-nu))*(1/(r**2)) * (((((r**2)-(R_int**2))/((R_ext**2)-(R_int**2)))*integrate.quad(lambda r: T_vessel(r)*r, R_int, R_ext)[0]) - integrate.quad(lambda r: T_vessel(r)*r, R_int, R_ext)[0])
sigma_t_th = lambda r: (E*alpha_l/(1-nu))*(1/(r**2)) * (((((r**2)+(R_int**2))/((R_ext**2)-(R_int**2)))*integrate.quad(lambda r: T_vessel(r)*r, R_int, R_ext)[0]) + integrate.quad(lambda r: T_vessel(r)*r, R_int, R_ext)[0] - T_vessel(r)*(r**2)) #In the second integral there should be r instead of R_ext but I'm not sure that makes sense
sigma_t_th_SIMP = lambda r: (E*alpha_l/(1-nu))*(T_vessel_avg - T_vessel(r))   #Simplified formula assuming average T
sigma_z_th = lambda r: sigma_r_th(r) + sigma_t_th(r)                          #Superposition principle under the hypothesis of long, hollow cylinder with load-free ends

sigma_th_max = max(sigma_t_th(r))
r_sigma_th_max = r[np.argmax(sigma_t_th(r))]
print("\nMaximum Thermal Hoop Stress: %.3f Mpa at r = %.3f m" %(sigma_th_max, r_sigma_th_max))
sigma_th_max_SIMP = max(sigma_t_th_SIMP(r))
r_sigma_th_max_SIMP = r[np.argmax(sigma_t_th_SIMP(r))]
print("Maximum Thermal Hoop Stress (Simplified formula): %.3f Mpa at r = %.3f m" %(sigma_th_max_SIMP, r_sigma_th_max_SIMP))

# ======================================
# Plotting the thermal stress profiles
# ======================================
plt.figure(figsize=(10,10))
plt.plot(r, sigma_r_th(r), label='Radial (r) Thermal Stress Profile')
plt.plot(r, sigma_t_th(r), label='Hoop (θ) Thermal Stress Profile')
plt.plot(r, sigma_t_th_SIMP(r), label='Simplified Hoop (θ) Thermal Stress Profile')
plt.plot(r, sigma_z_th(r), color='green', label='Axial (z) Thermal Stress Profile')
plt.axvline(x=r_sigma_th_max, color='red', linestyle='dashed', linewidth='0.5')
plt.axhline(y=sigma_th_max, color='red', linestyle='dashed', linewidth='0.5')
plt.plot(r_sigma_th_max, sigma_th_max,'--or', label='Max Hoop Stress')
plt.axvline(x=r_sigma_th_max_SIMP, color='cyan', linestyle='dashed', linewidth='0.5')
plt.axhline(y=sigma_th_max_SIMP, color='cyan', linestyle='dashed', linewidth='0.5')
plt.plot(r_sigma_th_max_SIMP, sigma_th_max_SIMP,'--oc', label='Simplified Max Hoop Stress')
plt.xlabel('Radius (m)')
plt.ylabel('Thermal Stress (MPa)')
plt.title('Wall Thermal Stress Profiles and Maximum Hoop Stress')
plt.legend()
plt.grid()
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
I'm positive there's something wrong in the thermal stress computation...to be checked, as the expected profiles when T(r) is linear are very different from the ones obtained here.
For example, the maximum thermal hoop stress should be at the inner surface. 
Finally, the maximum thermal hoop stress computed via the formula provided in the homework file is yet to be implemented, using the design curves and the sigma_T coefficient.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""