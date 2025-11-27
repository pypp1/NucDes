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
beta_cpp = 5.57e-4              #1/K
DeltaT = 30                     #K

# ============================
# Steel properties
# ============================
E = 177*1e3                    #MPa
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
P_int_MPa = P_int/10                        #MPa
P_cpp_MPa = P_cpp/10                        #MPa
Phi_0 = Phi_0 * 1e4                         #photons/(m²·s)
v = m_flr/(rho*np.pi*(D_vess_int**2)/4)     #m/s
Mar_criterion = R_int/t                     #To check for the validity of the Mariotte solution

# ======================================
# Simpson composite integration function
# =====================================
def simpcomp(f, a, b, N):
    """ Formula di Cavalieri-Simpson composita
    Input:
        f:   funzione da integrare (lambda function)
        a:   estremo inferiore intervallo di integrazione
        b:   estremo superiore intervallo di integrazione
        N:   numero di sottointervalli (N = 1 formula di integrazione semplice)
    Output:
        I:   integrale approssimato """
    h = (b-a)/N                                     # ampiezza sottointervalli
    x = np.linspace(a, b, N+1)                      # griglia spaziale
    xL, xR = x[:-1], x[1:]                          # liste dei nodi "sinistri" e "destri"
    xM = 0.5*(xL + xR)                              # punti medi
    I = (h/6.0)*(f(xL)+4*f(xM)+f(xR)).sum()         # integrale approssimato
    return I

# =============================================================================================================================================================
# PURELY MECHANICAL PROBLEM
# =============================================================================================================================================================
dr = int(input("\nEnter the no. of points for the r vector (eg: 1000): "))
r = np.linspace(R_int, R_ext, dr)

Def_P_flag = int(input("\nAssume default pressures (75 bar = 7.5 MPa)? (1: Yes, 0: No): "))
if Def_P_flag == 0:
    P_int = int(input("\nSet the internal pressure (bar): "))
    P_int_MPa = P_int/10
    P_cpp = int(input("Set the external pressure (bar): "))
    P_cpp_MPa = P_cpp/10

# ============================
# Mariotte Solution for a thin-walled cylinder and sphere (R_int = R_ext = R)
# ============================
if Mar_criterion > 5:
    Mariotte_flag = int(input("\nThe cylinder wall can be considered thin. Are you interested in visualizing the Mariotte solution for stress? (1: Yes, 0: No): "))
    sigma_rM_cyl = -P_int_MPa/2                        #Compressive
    sigma_tM_cyl = R_int*P_int_MPa/t                   
    sigma_zM_cyl = R_int*P_int_MPa/(2*t)
    #sigma_tM_sph = R_int*P_int_MPa/(2*t)

    if Mariotte_flag == 1:

        # ======================================
        # Plotting the stress profiles: Mariotte
        # ======================================
        plt.figure(figsize=(15,10))
        plt.axvline(x=R_int, color='black', linewidth='3', label='Vessel Inner Surface')
        plt.axvline(x=R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
        plt.axhline(y=sigma_rM_cyl, label='Radial (r) Stress Mariotte', color='blue')
        plt.axhline(y=sigma_tM_cyl, label=r'Hoop ($\theta$) Stress Mariotte', color='red')
        plt.axhline(y=sigma_zM_cyl, label='Axial (z) Stress Mariotte', color='green')
        plt.xlabel('Radius (m)')
        plt.ylabel('Stress (MPa)')
        plt.title('Stress Distribution in a thin-walled cylinder - Mariotte Solution')
        plt.legend()
        plt.grid()
        plt.show()

    elif Mariotte_flag == 0:
        print("Skipping Mariotte solution.")
else:
    print("\nThe cylinder can't be considered thin. Skipping Mariotte solution.")
    Mariotte_flag = 0

# ============================ 
# General Lamé Solution 
# ============================
def sigmaL_func(r):
    if P_int != 0 and P_cpp == 0:                                                                                   # P_int only
        sigma_rL = lambda r: - rho_i * P_int_MPa * ((R_ext**2)/(r**2) - 1)
        sigma_tL = lambda r: rho_i * P_int_MPa * ((R_ext**2)/(r**2) + 1)

    elif P_int == 0 and P_cpp != 0:                                                                                 # P_out only
        sigma_rL = lambda r: rho_ii * P_cpp_MPa * ((R_int**2)/(r**2) - 1)
        sigma_tL = lambda r: - rho_ii * P_cpp_MPa * (1 + (R_int**2)/(r**2)) 

    elif P_int != 0 and P_cpp != 0:                                                                                 # Both P_int and P_out: more general case
        sigma_rL = lambda r: (- rho_i * P_int_MPa * ((R_ext**2)/(r**2) - 1)) + (rho_ii * P_cpp_MPa * ((R_int**2)/(r**2) - 1))
        sigma_tL = lambda r: (- rho_ii * P_cpp_MPa * (1 + (R_int**2)/(r**2))) + (rho_ii * P_cpp_MPa * (1 + (R_int**2)/(r**2)))

    if P_int == P_cpp:                                                                                              #Hydrostatic Stress Condition
        print("\nInteral and external pressures are equal: hydrostatic stress condition is verified. Skipping.")
        eps_z_a = (2*nu-1)*rho_ii*P_cpp_MPa/E
        eps_z_b = (1-2*nu)*rho_i*P_int_MPa/E

    elif P_int != P_cpp:
        flag_eps = int(input("\nEnter the stress/strain condition (1: Plane Stress, 0: Plane Strain): "))

        if flag_eps == 1:                                                                                           #Plane Stress
            eps_z_a = 2*nu*rho_ii*P_cpp_MPa/E
            eps_z_b = -2*nu*rho_i*P_int_MPa/E
                
        elif flag_eps == 0:                                                                                         #Plane Strain
            eps_z_a = 0
            eps_z_b = 0 

    sigma_zL_a = E*eps_z_a - 2*nu*rho_ii*P_cpp_MPa  #a) P_int = 0
    sigma_zL_b = E*eps_z_b + 2*nu*rho_i*P_int_MPa   #b) P_cpp = 0
    return (sigma_rL(r), sigma_tL(r), sigma_zL_a + sigma_zL_b)              #Superposition Principle

sigma_L = sigmaL_func(r)
sigma_rL = sigma_L[0]  
sigma_tL = sigma_L[1]
sigma_zL = sigma_L[2]

if Mariotte_flag == 1:
    Lame_flag = int(input("\nThe Mariotte solution for a thin cylinder has been visualized. Are you interested in visualizing the more general Lamé solution? (1: Yes, 0: No): "))
    
elif Mariotte_flag == 0:
    print("Visualizing general Lamé solution.")
    Lame_flag = 1
    
if Lame_flag == 1:
    
    # ======================================
    # Plotting the stress profiles: Lamé
    # ======================================
    plt.figure(figsize=(15,10))
    plt.axvline(x=R_int, color='black', linewidth='3', label='Vessel Inner Surface')
    plt.axvline(x=R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
    plt.plot(r, sigma_rL, label='Radial (r) Stress Lamé')
    plt.plot(r, sigma_tL, label=r'Hoop ($\theta$) Stress Lamé')
    plt.axhline(y=sigma_zL, color='green', label='Axial (z) Stress Lamé')
    plt.xlabel('Radius (m)')
    plt.ylabel('Stress (MPa)')
    plt.title('Stress Distribution in the cylinder wall - Lamé Solution')
    plt.legend()
    plt.grid()
    plt.show()

elif Lame_flag == 0:
    print("Skipping Lamé solution.")

# =============================================================================================================================================================
# PURELY THERMAL PROBLEM - POWER IMPOSED - NO THERMAL SHIELD
# =============================================================================================================================================================
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
hs_flag = int(input("\nDo you want to visualize the volumetric heat source q0 inside the wall? (1: Yes, 0: No): "))
if hs_flag == 1:
    plt.figure(figsize=(10,10))
    plt.axvline(x=R_int, color='black', linewidth='3', label='Vessel Inner Surface')
    plt.axvline(x=R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
    plt.plot(r, q_iii(r), 'g', label='Radial (r) Volumetric heat source profile')
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

# =============================================================================================================================================================
# NB: The thermal resistances are computed per unit length of the vessel (l = 1)
# =============================================================================================================================================================
q_0_flag = int(input("\nDo you want to account for the presence of the volumetric heat source q0 inside the wall? (1: Yes, 0: No): "))
q_0 = q_0*q_0_flag
Disc_flag = int(input("Do you want to use a discretization approach along z? (1: Yes, 0: No): "))

# ======================================
# 1D Approach: no discretization along z
# ======================================
if Disc_flag == 0:
    print("No discretization along z. Assuming constant temperature of the primary fluid T1.")
    T1_flag = int(input("\nWhat temperature do you want to use as T1 to compute C1 and C2? (0: T_in, 1: T_out_avg, 2: T_out_max, 3: T_avg_log, 4: T_avg): "))
    adiab_flag = int(input("Apply Adiabatic Outer Wall approximation? (1: Yes, 0: No): "))

    if T1_flag == 0:
        T1 = T_in
    elif T1_flag == 1:
        T1 = T_out_avg
    elif T1_flag == 2:
        T1 = T_out_max
    elif T1_flag == 3:
        T1 = (T_in - T_out_avg)/np.log(T_in/T_out_avg)
    elif T1_flag == 4:
        T1 = (T_in + T_out_avg)/2

    # ======================================
    # T profile constants for the vessel: general and under adiabatic outer wall approximation (dT/dx = 0 at r = R_ext)
    # ======================================
    if adiab_flag == 0:
        C1 = ((q_0/(k_st*mu_st**2))*(np.exp(-mu_st*t)-1)-(q_0/mu_st)*((1/h_1)+(np.exp(-mu_st*t)/u_2))-(T1-T_cpp))/(t+(k_st/h_1)+(k_st/u_2))
    elif adiab_flag == 1:
        C1 = -((q_0/(k_st*mu_st))*np.exp(-mu_st*t))
    C2 = T1 + (q_0/(h_1*mu_st)) + C1*(k_st/h_1) + (q_0/(k_st*mu_st**2))
    # =============================================================================================================================================================
    # This approach uses a constant T1 to compute C1 and C2, but T of the primary fluid changes along z...More logical ways to proceed would be:

    # 1) Discretize the thermal problem along z, obtaining an array of T1 values which will result in an array of C1 and C2 values. Then,
    # compute T(r) for each discretized z and solve the problem in a 2D scheme. This is done later on.
    # 2) Check which one is the worst case scenario (which results in the highest thermal stresses) between T_in and T_out_max, and use that 
    # value as T1. While keeping a constant T1, this would still be a conservative approach. The choice to adopt this method is left to the user.
    # =============================================================================================================================================================

    # ======================================
    # T profiles across the vessel wall, average T, maximum and its position
    # ======================================
    T_vessel = lambda r: -((q_0/(k_st*mu_st**2))*np.exp(-mu_st*(r-R_int))) + C1*(r-R_int) + C2
    T_vessel_avg = (1/t)*integrate.quad(T_vessel, R_int, R_ext)[0]
    T_vessel_max = max(T_vessel(r))
    r_T_vessel_max = r[np.argmax(T_vessel(r))]
    #T_vessel_avg_2 = (q_0/(k_st*mu_st**2))*((np.exp(-mu_st*t)-1)/(mu_st*t))+ C1*(t/2) + C2                                          #Analytical Integration Result

    # ======================================
    # Thermal power fluxes (kW/m²) on the inner and outer vessel surface
    # ======================================
    DeltaT_1 = T1 - T_vessel(r[0])
    str1 = "\nT1 = T_in has been assumed: a logarithmic mean DeltaT could thus be useful to account for the T profile along z in an approximate way, even though the vessel wall temperature is not constant."
    str2 = "The heat flux computed with the regular DeltaT will still be displayed."
    str3 = "Do you want to adopt such an approach? (1: Yes, 0: No): "
    prompt = "\n".join([str1, str2, str3]) + " "
    
    if T1_flag == 0:
        LogDelta_flag = int(input(prompt))
        if LogDelta_flag == 1:
            DeltaT_LM1 = ((T1-T_vessel(r[0]))-(T_out_avg-T_vessel(r[0])))/(np.log((T1-T_vessel(r[0]))/(T_out_avg-T_vessel(r[0]))))        #Log Mean Temperature Difference to account for T change along z, instead of just using T1-T_wall
            q_s1_log = h_1*DeltaT_LM1/1000                                                                                                #kW/m²
            print("\nThermal power flux on the inner vessel surface - Logarithmic Mean DeltaT Approach: %.3f kW/m²" %q_s1_log)
    q_s1 = h_1*DeltaT_1/1000                                                                                                              #kW/m²
    q_s2 = u_2*(T_vessel(r[-1])-T_cpp)/1000                                                                                               #kW/m²

    print("\nThermal power flux on the inner vessel surface: %.3f kW/m²" %q_s1)
    print("Thermal power flux on the outer vessel surface: %.3f kW/m²" %q_s2)
    # ======================================
    # CHECK IF THIS IS CORRECT: UNDER ADIABATIC OUTER WALL APPROXIMATION, SHOULDN'T THE THERMAL HEAT FLUX BE NULL AT R_EXT?
    # ======================================

    # ======================================
    # Plotting the wall T profiles
    # ======================================
    T_pl_flag = int(input("\nDo you want to visualize the T profile across the vessel's wall? (1: Yes, 0: No): "))
    
    if adiab_flag == 0:
        print("\nVessel inner surface Temperature: %.3f °C" %T_vessel(r[0]))
        print("Vessel outer surface Temperature: %.3f °C" %T_vessel(r[-1]))
        print("Average Vessel Temperature (numerical integration): %.3f °C" %T_vessel_avg)
        #print("Average Vessel Temperature (analytical integration): %.3f °C" %T_vessel_avg_2)
        print("Maximum Vessel Temperature: %.3f °C at r = %.3f m" %(T_vessel_max, r_T_vessel_max))
        
        if T_pl_flag == 1:
            # ======================================
            # No Adiabatic Outer Wall Approximation
            # ======================================
            plt.figure(figsize=(10,10))
            plt.axvline(x=R_int, color='black', linewidth='3', label='Vessel Inner Surface')
            plt.axvline(x=R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
            plt.plot(r, T_vessel(r), label='Radial (r) T Profile')
            plt.axvline(x=r_T_vessel_max, color='red', linestyle='dashed', linewidth='0.5')
            plt.axhline(y=T_vessel_max, color='red', linestyle='dashed', linewidth='0.5')
            plt.plot(r_T_vessel_max, T_vessel_max,'--or',label='Max T')
            plt.axhline(y=T_vessel_avg, color='green', label='Wall Average T')
            plt.xlabel('Radius (m)')
            plt.ylabel('T (K)')
            plt.title('Wall Temperature Profile, Average and Maximum ')
            plt.legend()
            plt.grid()
            plt.show()

    elif adiab_flag == 1:
        print("\nVessel inner surface Temperature: %.3f °C" %T_vessel(r[0]))
        print("Vessel outer surface Temperature: %.3f °C" %T_vessel(r[-1]))
        print("Average Vessel Temperature under Adiabatic Outer Wall approximation (numerical integration): %.3f °C" %T_vessel_avg)
        #print("Average Vessel Temperature under Adiabatic Outer Wall approximation (analytical integration): %.3f °C" %T_vessel_avg_2)
        print("Maximum Vessel Temperature under Adiabatic Outer Wall approximation: %.3f °C at r = %.3f m" %(T_vessel_max, r_T_vessel_max))
        
        if T_pl_flag == 1:
            # ======================================
            # Under Adiabatic Outer Wall Approximation
            # ======================================
            plt.figure(figsize=(10,10))
            plt.axvline(x=R_int, color='black', linewidth='3', label='Vessel Inner Surface')
            plt.axvline(x=R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
            plt.plot(r, T_vessel(r), label='Radial (r) T Profile')
            plt.axvline(x=r_T_vessel_max, color='red', linestyle='dashed', linewidth='0.5')
            plt.axhline(y=T_vessel_max, color='red', linestyle='dashed', linewidth='0.5')
            plt.axvline(x=R_int, color='black', linewidth='3', label='Vessel Inner Surface')
            plt.axvline(x=R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
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
    f = lambda r: T_vessel(r)*r

    sigma_r_th = np.zeros(dr)
    sigma_t_th = np.zeros(dr)
    for i in range(len(r)):
        sigma_r_th[i] = (E*alpha_l/(1-nu))*(1/(r[i]**2)) * (( ((r[i]**2)-(R_int**2))/((R_ext**2)-(R_int**2)) ) * simpcomp(f, R_int, R_ext, dr) - simpcomp(f, R_int, r[i], dr))
        sigma_t_th[i] = (E*alpha_l/(1-nu))*(1/(r[i]**2)) * (( (((r[i]**2)+(R_int**2))/((R_ext**2)-(R_int**2)) ) * simpcomp(f, R_int, R_ext, dr)) + simpcomp(f, R_int, r[i], dr) - T_vessel(r[i])*(r[i]**2))
    sigma_t_th_SIMP = lambda r: (E*alpha_l/(1-nu))*(T_vessel_avg - T_vessel(r))   #Simplified formula assuming average T
    sigma_z_th = sigma_r_th + sigma_t_th                                          #Superposition principle under the hypothesis of long, hollow cylinder with load-free ends

    sigma_th_max = max(sigma_t_th)
    r_sigma_th_max = r[np.argmax(sigma_t_th)]
    print("\nMaximum Thermal Hoop Stress: %.3f MPa at r = %.3f m" %(sigma_th_max, r_sigma_th_max))
    #sigma_th_max_SIMP = max(sigma_t_th_SIMP(r))
    #r_sigma_th_max_SIMP = r[np.argmax(sigma_t_th_SIMP(r))]
    #print("Maximum Thermal Hoop Stress (Simplified formula): %.3f Mpa at r = %.3f m" %(sigma_th_max_SIMP, r_sigma_th_max_SIMP))

    # ======================================
    # Maximum Hoop Thermal Stress via design curves - NB: Requires double interpolation for the tentative thickness used - Manual only: data taken from the given graph
    # ======================================
    mu20ba114 = 0.61
    mu20ba116 = 0.67
    mu30ba114 = 0.76
    mu30ba116 = 0.8
    
    iso_mu20 = lambda x: mu20ba114 + ((mu20ba116-mu20ba114)/(1.16-1.14))*(x - 1.14)
    sigmaT_1st = iso_mu20(R_ext/R_int)                                                     #Interpolated sigmaT coefficient on the ISO-mu = 20 between b/a = 1.14 and b/a = 1.16
    iso_mu30 = lambda x: mu30ba114 + ((mu30ba116-mu30ba114)/(1.16-1.14))*(x - 1.14)
    sigmaT_2nd = iso_mu30(R_ext/R_int)                                                     #Interpolated sigmaT coefficient on the ISO-mu = 30 between b/a = 1.14 and b/a = 1.16
    sigmaT_eq = lambda x: sigmaT_1st + ((sigmaT_2nd-sigmaT_1st)/(30-20))*(x - 20)
    sigmaT = sigmaT_eq(mu_st)                                                              #Double-interpolated (linear) sigmaT coefficient for mu = 24 
    sigma_t_th_max_DES = sigmaT*(alpha_l*E*q_0)/(k_st*(1-nu)*(mu_st**2))
    
    print("Maximum Thermal Hoop Stress via Design Curves: %.3f MPa" %sigma_t_th_max_DES)
    
    # ======================================
    # Plotting the thermal stress profiles
    # ======================================
    sigma_th_pl_flag = int(input("\nDo you want to visualize a plot of the thermal stress profiles? (1: Yes, 0: No): "))
    if sigma_th_pl_flag == 1:
        plt.figure(figsize=(10,10))
        plt.axvline(x=R_int, color='black', linewidth='3', label='Vessel Inner Surface')
        plt.axvline(x=R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
        plt.plot(r, sigma_r_th, linewidth='0.75', label='Radial (r) Thermal Stress Profile')
        plt.plot(r, sigma_t_th, linewidth='0.75', label='Hoop (θ) Thermal Stress Profile')
        plt.plot(r, sigma_t_th_SIMP(r), label='Simplified Hoop (θ) Thermal Stress Profile')
        plt.plot(r, sigma_z_th, color='green', linewidth='0.5', label='Axial (z) Thermal Stress Profile')
        plt.axhline(y=0, color='black', linewidth='1', label='y=0')
        plt.axvline(x=r_sigma_th_max, color='red', linestyle='dashed', linewidth='0.5')
        plt.axhline(y=sigma_th_max, color='red', linestyle='dashed', linewidth='0.5')
        plt.plot(r_sigma_th_max, sigma_th_max,'--or', label='Max Hoop Stress')
        #plt.axvline(x=r_sigma_th_max_SIMP, color='cyan', linestyle='dashed', linewidth='0.5')
        #plt.axhline(y=sigma_th_max_SIMP, color='cyan', linestyle='dashed', linewidth='0.5')
        #plt.plot(r_sigma_th_max_SIMP, sigma_th_max_SIMP,'--oc', label='Simplified Max Hoop Stress')
        plt.xlabel('Radius (m)')
        plt.ylabel('Thermal Stress (MPa)')
        plt.title('Wall Thermal Stress Profiles and Maximum Hoop Stress')
        plt.legend()
        plt.grid()
        plt.show()
    
    # ======================================
    # Principal stresses sum and elastic regime verification
    # ======================================
    sigma_r_totM = sigma_rM_cyl + sigma_r_th
    sigma_t_totM = sigma_tM_cyl + sigma_t_th
    sigma_z_totM = sigma_zM_cyl + sigma_z_th
    
    sigma_r_totL = sigma_rL + sigma_r_th
    sigma_t_totL = sigma_tL + sigma_t_th
    sigma_z_totL = sigma_zL + sigma_z_th
    
    # ============================ 
    # Comparison stress - Guest-Tresca Theory - Mariotte/Lamé + Thermal stresses
    # ============================
    sigma_cTR_M = np.max([abs(sigma_t_totM - sigma_r_totM), abs(sigma_z_totM - sigma_r_totM), abs(sigma_t_totM - sigma_z_totM)])
    sigma_cTR_L = np.max([abs(sigma_t_totL - sigma_r_totL), abs(sigma_z_totL - sigma_r_totL), abs(sigma_t_totL - sigma_z_totL)])
    print("\nGuest-Tresca Equivalent Stress - Mariotte solution: %.3f Mpa" %sigma_cTR_M)
    print("Guest-Tresca Equivalent Stress - Lamé solution: %.3f Mpa" %sigma_cTR_L)

    # ============================ 
    # Comparison stress - Von Mises Theory - Mariotte/Lamé + Thermal stresses
    # ============================
    sigma_cVM_M = max(np.sqrt(0.5*((sigma_r_totM - sigma_t_totM)**2 + (sigma_t_totM - sigma_z_totM)**2 + (sigma_z_totM - sigma_r_totM)**2)))
    sigma_cVM_L = max(np.sqrt(0.5*((sigma_r_totL - sigma_t_totL)**2 + (sigma_t_totL - sigma_z_totL)**2 + (sigma_z_totL - sigma_r_totL)**2))) #The max should be the worst case, in theory
    print("\nVon Mises Equivalent Stress - Mariotte solution: %.3f Mpa" %sigma_cVM_M)
    print("Von Mises Equivalent Stress - Lamé solution: %.3f Mpa" %sigma_cVM_L)
    
    # =============================================================================================================================================================
    # CHECK FOR EXPECTED PROFILES
    # =============================================================================================================================================================
        
elif Disc_flag == 1:
    
    # ======================================
    # T discretization along z
    # ======================================
    dz = int(input("\nEnter the no. of points for the T_z vector (eg: 1000): "))
    T_z = np.linspace(T_in, T_out_avg, dz)
    adiab_flag = int(input("\nApply Adiabatic Outer Wall approximation? (1: Yes, 0: No): "))

    if adiab_flag == 0:
        C1 = ((q_0/(k_st*mu_st**2))*(np.exp(-mu_st*t)-1)-(q_0/mu_st)*((1/h_1)+(np.exp(-mu_st*t)/u_2))-(T_z - T_cpp))/(t+(k_st/h_1)+(k_st/u_2))      # Make C1 a 1D array aligned with T_z (avoid wrapping in another dimension)
    elif adiab_flag == 1:
        C1 = np.full_like(T_z, -((q_0/(k_st*mu_st)) * np.exp(-mu_st*t)))                                                                            # Constant C1 for all z points -> replicate to match T_z shape
    C2 = T_z + (q_0/(h_1*mu_st)) + C1 * (k_st/h_1) + (q_0/(k_st*mu_st**2))                                                                          # C2 should also be a 1D array matching T_z
    
    # ======================================
    # T profiles across the vessel wall, average Ts, maxima and their positions
    # ======================================
    def T_vessel_func_all(r, C1, C2):
        T_vessel_r = np.zeros((dz, dr))
        T_vessel_avg_arr = np.zeros(dz)
        T_vessel_max_arr = np.zeros(dz)
        r_T_vessel_max_arr = np.zeros(dz)
        sigma_r_th = np.zeros((dz, dr))
        sigma_t_th = np.zeros((dz, dr))
        sigma_t_th_SIMP = np.zeros((dz, dr))

        for i in range(dz):
            # ======================================
            # T Profiles computation
            # ======================================
            T_vessel_r[i, :] = -((q_0/(k_st*mu_st**2)) * np.exp(-mu_st * (r - R_int))) + C1[i] * (r - R_int) + C2[i]                               # vectorized radial temperature for this z-index               
            T_vessel_r_lamb = lambda rr: -((q_0/(k_st*mu_st**2)) * np.exp(-mu_st * (rr - R_int))) + C1[i] * (rr - R_int) + C2[i]                   # use a lambda that accepts a scalar rr (avoid shadowing array `r`)
            T_vessel_avg_arr[i] = (1 / t) * integrate.quad(T_vessel_r_lamb, R_int, R_ext)[0]                                                       # integrate the scalar function over radius (returns scalar)
            T_vessel_max_arr[i] = np.max(T_vessel_r[i, :])
            r_T_vessel_max_arr[i] = r[np.argmax(T_vessel_r[i, :])]

            # ======================================
            # Thermal stresses computation
            # ======================================
            ff = lambda rr: (-((q_0/(k_st*mu_st**2)) * np.exp(-mu_st * (rr - R_int))) + C1[i] * (rr - R_int) + C2[i])*rr
            for j in range(dr):
                sigma_r_th[i, j] = (E*alpha_l/(1-nu))*(1/(r[j]**2)) * (( ((r[j]**2)-(R_int**2))/((R_ext**2)-(R_int**2)) ) * simpcomp(ff, R_int, R_ext, dr) - simpcomp(ff, R_int, r[j], dr))
                sigma_t_th[i, j] = (E*alpha_l/(1-nu))*(1/(r[j]**2)) * (( (((r[j]**2)+(R_int**2))/((R_ext**2)-(R_int**2)) ) * simpcomp(ff, R_int, R_ext, dr)) + simpcomp(ff, R_int, r[j], dr) - T_vessel_r_lamb(r[j])*(r[j]**2))
                sigma_t_th_SIMP[i, j] = (E*alpha_l/(1-nu))*(T_vessel_avg_arr[i] - T_vessel_r_lamb(r[j]))   
        sigma_z_th = sigma_r_th + sigma_t_th                                          

        return (T_vessel_r, T_vessel_avg_arr, T_vessel_max_arr, r_T_vessel_max_arr, sigma_r_th, sigma_t_th, sigma_t_th_SIMP, sigma_z_th)
    
    Output = T_vessel_func_all(r, C1, C2)
    T_vessel_Mat = Output[0]
    T_vessel_avg_arr = Output[1]
    T_vessel_max_arr = Output[2]
    r_T_vessel_max_arr = Output[3]
    sigma_r_th = Output[4]
    sigma_t_th = Output[5]
    sigma_t_th_SIMP = Output[6]
    sigma_z_th = Output[7]
    
    # ======================================
    # Thermal power fluxes (kW/m²) on the inner and outer vessel surface
    # ======================================
    DeltaT_LM1 = np.zeros(len(T_z))
    q_s1 = np.zeros(len(T_z))
    q_s2 = np.zeros(len(T_z))
    for i in range(len(T_z)):
        DeltaT_LM1[i] = ((T_in-T_vessel_Mat[i, 0])-(T_out_avg-T_vessel_Mat[i, 0]))/(np.log((T_in-T_vessel_Mat[i, 0])/(T_out_avg-T_vessel_Mat[i, 0])))       #Log Mean Temperature Difference to account for T change along z, instead of just using Tin-T_wall
        q_s1[i] = h_1*DeltaT_LM1[i]/1000                                                                                                                    #kW/m²
        q_s2[i] = u_2*(T_vessel_Mat[i, -1]-T_cpp)/1000                                                                                                      #kW/m²
    
    print("\nThermal power flux on the inner vessel surface: \nMin: %.3f kW/m² \nMax: %.3f kW/m²" %(np.min(q_s1),np.max(q_s1)))
    print("\nThermal power flux on the outer vessel surface: \nMin: %.3f kW/m² \nMax: %.3f kW/m²" %(np.min(q_s2),np.max(q_s2)))

    # ======================================
    # Plotting the wall T profiles
    # ======================================
    if adiab_flag == 0:
        
        # ======================================
        # Wall T(T_z, r) map
        # ======================================
        R_mesh, T_z_mesh = np.meshgrid(r, T_z)   # shapes (Nz, Nr)
        plt.figure(figsize=(15,10))
        plt.subplot(1,2,1)
        pcm = plt.pcolormesh(R_mesh, T_z_mesh, T_vessel_Mat, shading='auto', cmap='hot')   # or 'hot','plasma','viridis'
        plt.colorbar(pcm, label='T (°C)')
        plt.xlabel('Radius (m)')
        plt.ylabel('T$_z$ (°C)')
        plt.title('Wall Temperature Map (r vs T$_z$)')
        plt.tight_layout()
        
        # ======================================
        # T_avg and T_max profiles as T_z grows
        # ======================================
        plt.subplot(1,2,2)
        plt.plot(T_z, T_vessel_max_arr, 'r', label='Max T Axial (z) Profile')                   #The r position of T_max is always the same in this approach!
        plt.plot(T_z, T_vessel_avg_arr, 'k', label='Average T Axial (z) Profile')
        plt.xlabel('T$_z$ (°C)')
        plt.ylabel('T (°C)')
        plt.title('Maximum and Average Wall Temperature Profiles as T$_z$ grows')
        plt.legend()
        plt.grid()
        plt.show()
        
    elif adiab_flag == 1:
        
        # ======================================
        # Wall T(T_z, r) map
        # ======================================
        R_mesh, T_z_mesh = np.meshgrid(r, T_z)
        plt.figure(figsize=(15,10))
        plt.subplot(1,2,1)
        pcm = plt.pcolormesh(R_mesh, T_z_mesh, T_vessel_Mat, shading='auto', cmap='hot')
        plt.colorbar(pcm, label='T (°C)')
        plt.xlabel('Radius (m)')
        plt.ylabel('T$_z$ (°C)')
        plt.title('Wall Temperature Map under AOW Approximation (r vs T$_z$)')
        plt.tight_layout()
        
        # ======================================
        # T_avg and T_max profiles as T_z grows
        # ======================================
        plt.subplot(1,2,2)
        plt.plot(T_z, T_vessel_max_arr, 'r', label='Max T Axial (z) Profile')
        plt.plot(T_z, T_vessel_avg_arr, 'k', label='Average T Axial (z) Profile')
        plt.xlabel('T$_z$ (°C)')
        plt.ylabel('T (°C)')
        plt.title('Maximum and Average Wall Temperature Profiles as T$_z$ grows under AOW Approximation')
        plt.legend()
        plt.grid()
        plt.show()
    
    # ======================================
    # Plotting the thermal stress profiles
    # ======================================
    R_mesh, T_z_mesh = np.meshgrid(r, T_z)   # shapes (Nz, Nr)
    plt.figure(figsize=(20,20))
    plt.subplot(1,4,1)
    pcm = plt.pcolormesh(R_mesh, T_z_mesh, sigma_r_th, shading='auto', cmap='plasma')
    plt.colorbar(pcm, label=r'$\sigma$ (MPa)')
    plt.xlabel('Radius (m)')
    plt.ylabel('T (°C)')
    plt.title('Radial Stress Map (r vs T$_z$)')
    plt.tight_layout()

    plt.subplot(1,4,2)
    pcm = plt.pcolormesh(R_mesh, T_z_mesh, sigma_t_th, shading='auto', cmap='plasma')
    plt.colorbar(pcm, label=r'$\sigma$ (MPa)')
    plt.xlabel('Radius (m)')
    plt.ylabel('T (°C)')
    plt.title('Hoop Stress Map (r vs T$_z$)')
    plt.tight_layout()

    plt.subplot(1,4,3)
    pcm = plt.pcolormesh(R_mesh, T_z_mesh, sigma_t_th_SIMP, shading='auto', cmap='plasma')
    plt.colorbar(pcm, label=r'$\sigma$ (MPa)')
    plt.xlabel('Radius (m)')
    plt.ylabel('T (°C)')
    plt.title('Simplified Hoop Stress Map (r vs T$_z$)')
    plt.tight_layout()

    plt.subplot(1,4,4)
    pcm = plt.pcolormesh(R_mesh, T_z_mesh, sigma_z_th, shading='auto', cmap='plasma')
    plt.colorbar(pcm, label=r'$\sigma$ (MPa)')
    plt.xlabel('Radius (m)')
    plt.ylabel('T (°C)')
    plt.title('Axial Stress Map (r vs T$_z$)')
    plt.tight_layout()
    plt.show()