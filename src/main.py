import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lng 
from scipy import integrate, interpolate

# ============================
# Geometrical data
# ============================
D_barr_ext = 2.5       #m
D_vess_int = 3.0       #m
t_th_ins = 0.05        #m 
k_th_ins = 1.4         #W/mK
L = 7                  #m
W = 0.01               #ASSUMED: To be changed using ASME III NB4000, as this is only valid for IRIS SG Tubes

# ============================
# Primary fluid
# ============================
T_in = 214 + 273.15        #K
T_out_avg = 254 + 273.15   #K
T_out_max = 270 + 273.15   #K
P_int = 75                 #bar
m_flr = 3227               #kg/s
Cp = 4534                  #J/(kg·K)
rho = 852.5                #kg/m³
mu = 1.259e-4              #Pa·s
k = 0.658                  #W/(m·K)

# ============================
# Containment (CPP) water
# ============================
T_cpp = 70 + 273.15             #K
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
sigma_y = np.array([240,232.5,222,216,210,204,199.5,195,190.5,186,181.5,177,171,165,157.5,147])         #MPa
sigma_in = np.array([160,155,148,144,140,136,133,130,127,124,121,118,114,110,105,98])                   #MPa
T_thr = np.array([40,65,100,125,150,175,200,225,250,275,300,325,350,375,400,425])                       #°C
T_creep = 370                                                                                           #°C
creep_flag_V = 0
creep_flag_S = 0

# ============================
# Radiation source
# ============================
Phi_0 = 1.5e13                 #photons/(cm²·s)
E_y = 6.0                      #MeV
E_y_J = E_y * 1.60218e-13      #Joules
B = 1.4                        #Build-up factor

# ============================
# Design Curves
# ============================
loaded_data = np.load('multiple_arrays.npz')        #This contains all the arrays pertaining the iso-mu design curves

data_mu2 = loaded_data['data_mu2']
data_mu5 = loaded_data['data_mu5']
data_mu8 = loaded_data['data_mu8']
data_mu10 = loaded_data['data_mu10']
data_mu15 = loaded_data['data_mu15']
data_mu20 = loaded_data['data_mu20']
data_mu30 = loaded_data['data_mu30']
data_mu40 = loaded_data['data_mu40']
data_mu50 = loaded_data['data_mu50']
data_mu75 = loaded_data['data_mu75']
data_mu100 = loaded_data['data_mu100']

mu_curves = {                  # Dictionary to access the curves
    'mu2': (1, 2, data_mu2),
    'mu5': (2, 5, data_mu5),
    'mu8': (3, 8, data_mu8),
    'mu10': (4, 10, data_mu10),
    'mu15': (5, 15, data_mu15),
    'mu20': (6, 20, data_mu20),
    'mu30': (7, 30, data_mu30),
    'mu40': (8, 40, data_mu40),
    'mu50': (9, 50, data_mu50),
    'mu75': (10, 75, data_mu75),
    'mu100': (11, 100, data_mu100)
}

ba_ratio_plot = np.linspace(1.0, 1.20, 1000)
indexes = np.array([entry[0] for entry in mu_curves.values()])
mu_values = np.array([entry[1] for entry in mu_curves.values()])
keys_list = list(mu_curves.keys())

# ============================
# Computed additional data
# ============================
t = 0.05                                    #m #First guess
R_int = D_vess_int/2                        #m
R_ext = R_int + t                           #m
R_barr_ext = D_barr_ext/2                   #m
v_flr = m_flr/rho                           #m³/s
G = E/(2*(1+nu))                            #MPa
rho_ii = (R_ext**2)/(R_ext**2 - R_int**2)
rho_i = (R_int**2)/(R_ext**2 - R_int**2)
P_int_MPa = P_int/10                        #MPa
P_cpp_MPa = P_cpp/10                        #MPa
Phi_0 = Phi_0 * 1e4                         #photons/(m²·s)
Mar_criterion = R_int/t

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
    h = (b-a)/N                                     # Intervals width
    x = np.linspace(a, b, N+1)                      # Space grid
    xL, xR = x[:-1], x[1:]                          # Left and right nodes list
    xM = 0.5*(xL + xR)                              # Middle points
    I = (h/6.0)*(f(xL)+4*f(xM)+f(xR)).sum()         # Approximate integral
    return I

# =============================================================================================================================================================
# PURELY MECHANICAL PROBLEM
# =============================================================================================================================================================
dr = 1000
r = np.linspace(R_int, R_ext, dr)

while True:
    try:
        Def_P_flag = int(input("\nAssume default pressures (75 bar = 7.5 MPa)? (1: Yes, 0: No): "))
        if Def_P_flag not in (0, 1):
            raise RuntimeError("Invalid input! Please enter either 0 or 1.")
        break  
    except ValueError:
        print("Please enter a valid integer.")
    except RuntimeError as e:
        print(e)

if Def_P_flag == 0:
    P_int = int(input("\nSet the internal pressure (bar): "))
    P_int_MPa = P_int/10
    P_cpp = int(input("Set the external pressure (bar): "))
    P_cpp_MPa = P_cpp/10
    
    while P_int != P_cpp: #Asks for this here, because asking for it in the sigmaL function would mean having to input the value for every iteration
        try:
            flag_eps = int(input("\nEnter the stress/strain condition (1: Plane Stress, 0: Plane Strain): "))
            if flag_eps not in (0, 1):
                raise RuntimeError("Invalid input! Please enter either 0 or 1.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)

# ============================
# Mariotte Solution for a thin-walled cylinder and sphere (R_int = R_ext = R)
# ============================
if Mar_criterion > 5:
    while True:
        try:
            Mariotte_flag = int(input("\nWith an initial thickness value of %.3f m, the vessel can be considered thin. Are you interested in visualizing the Mariotte solution for stress? (1: Yes, 0: No): " %t))
            if Mariotte_flag not in (0, 1):
                raise RuntimeError("Invalid input! Please enter either 0 or 1.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)

    sigma_rM_cyl = -P_int_MPa/2                        #Compressive
    sigma_tM_cyl = R_int*P_int_MPa/t                   
    sigma_zM_cyl = R_int*P_int_MPa/(2*t)
    sigma_tM_sph = R_int*P_int_MPa/(2*t)

    if Mariotte_flag == 1:

        # ======================================
        # Plotting the stress profiles: Mariotte
        # ======================================
        plt.figure(figsize=(15,10))
        plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
        plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
        plt.axhline(y = sigma_rM_cyl, color='red', label='Radial (r) Stress Mariotte')
        plt.axhline(y = sigma_tM_cyl, color='blue', label=r'Hoop ($\theta$) Stress Mariotte')
        plt.axhline(y = sigma_zM_cyl, color='green', label='Axial (z) Stress Mariotte')
        plt.plot(r, np.zeros(len(r)), color='black', linewidth='1', label='y=0')
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
def sigmaL_func(r, P_int_MPa, P_cpp_MPa, verbose): #the "verbose" variable is used to avoid printing the hydrostatic stress condition information for every iteration of the thermal shield loop
    
    A = ((P_int_MPa*(R_int**2))-(P_cpp_MPa*(R_ext**2)))/((R_ext**2)-(R_int**2))
    B = (((R_int**2)*(R_ext**2))/((R_ext**2)-(R_int**2)))*(P_int_MPa-P_cpp_MPa)
    sigma_rL = lambda r: A - B/(r**2)
    sigma_tL = lambda r: A + B/(r**2)

    if P_int == P_cpp:
        if verbose:
            print("\nInternal and external pressures are equal: hydrostatic stress condition is verified. Skipping.")    #Hydrostatic Stress Condition
        eps_z_a = (2*nu-1)*rho_ii*P_cpp_MPa/E
        eps_z_b = (1-2*nu)*rho_i*P_int_MPa/E

    elif P_int != P_cpp:
        if flag_eps == 1:                                                                                           #Plane Stress
            eps_z_a = 2*nu*rho_ii*P_cpp_MPa/E
            eps_z_b = -2*nu*rho_i*P_int_MPa/E
        elif flag_eps == 0:                                                                                         #Plane Strain
            eps_z_a = 0
            eps_z_b = 0 

    sigma_zL_a = E*eps_z_a - 2*nu*rho_ii*P_cpp_MPa  #a) P_int = 0
    sigma_zL_b = E*eps_z_b + 2*nu*rho_i*P_int_MPa   #b) P_cpp = 0
    return (sigma_rL(r), sigma_tL(r), sigma_zL_a + sigma_zL_b)              #Superposition Principle

sigma_L = sigmaL_func(r, P_int_MPa, P_cpp_MPa, 1)
sigma_rL = sigma_L[0]  
sigma_tL = sigma_L[1]
sigma_zL = sigma_L[2]

if Mariotte_flag == 1:
    while True:
        try:
            Lame_flag = int(input("\nThe Mariotte solution for a thin cylinder has been visualized. Are you interested in visualizing the more general Lamé solution? (1: Yes, 0: No): "))
            if Lame_flag not in (0, 1):
                raise RuntimeError("Invalid input! Please enter either 0 or 1.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)

elif Mariotte_flag == 0:
    print("Visualizing general Lamé solution.")
    Lame_flag = 1

if Lame_flag == 1:
    # ======================================
    # Plotting the stress profiles: Lamé
    # ======================================
    plt.figure(figsize=(15,10))
    plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
    plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
    plt.plot(r, sigma_rL, label='Radial (r) Stress Lamé')
    plt.plot(r, sigma_tL, label=r'Hoop ($\theta$) Stress Lamé')
    plt.axhline(y = sigma_zL, color='green', label='Axial (z) Stress Lamé')
    plt.axhline(y = 0, color='black', linewidth='1', label='y=0')
    plt.xlabel('Radius (m)')
    plt.ylabel('Stress (MPa)')
    plt.title('Stress Distribution in the cylinder wall - Lamé Solution')
    plt.legend()
    plt.grid()
    plt.show()

elif Lame_flag == 0:
    print("Skipping Lamé solution.")
    
# ======================================
# Thermal Shield Check
# ======================================    
while True:
    try:
        TS_flag = int(input("\nDo you want to consider the presence of a thermal shield between the barrel and the vessel? (1: Yes, 0: No): "))
        if TS_flag not in (0, 1):
            raise RuntimeError("Invalid input! Please enter either 0 or 1.")
        break  
    except ValueError:
        print("Please enter a valid integer.")
    except RuntimeError as e:
        print(e)

# =============================================================================================================================================================
# PURELY THERMAL PROBLEM - POWER IMPOSED - NO THERMAL SHIELD
# =============================================================================================================================================================
if TS_flag == 0:
    # ============================
    # Computed additional data without the thermal shield
    # ============================
    v = m_flr/(rho*np.pi*((D_vess_int**2)-(D_barr_ext**2))/4)     #m/s
    Phi_0V = Phi_0                                                #All gamma rays reach the vessel

    # ======================================
    # Radiation-induced heating in the vessel
    # ======================================
    Phi = lambda r: Phi_0V*np.exp(-mu_st*(r-R_int))    #1/(m²·s)
    I = lambda r: E_y_J*Phi(r)*B                       #W/(m²)
    q_0 = B*Phi_0V*E_y_J*mu_st                         #W/(m³)
    q_iii = lambda r: q_0*np.exp(-mu_st*(r-R_int))     #W/(m³)

    # ======================================
    # Plotting the volumetric heat source profiles 
    # ======================================
    while True:
        try:
            hs_flag = int(input("\nDo you want to visualize the volumetric heat source q0 inside the vessel's wall? (1: Yes, 0: No): "))
            if hs_flag not in (0, 1):
                raise RuntimeError("Invalid input! Please enter either 0 or 1.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)

    if hs_flag == 1:
        plt.figure(figsize=(10,10))
        plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
        plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
        plt.plot(r, q_iii(r), 'g', label='Radial (r) Volumetric heat source profile')
        plt.plot(r[0], q_iii(r[0]), 'or', label='Vessel Inner Surface Value')
        plt.plot(r[-1], q_iii(r[-1]), 'or', label='Vessel-Insulation Interface Value')
        plt.axhline(y = 0, color='black', linewidth='1', label='y=0')
        plt.xlabel('Radius (m)')
        plt.ylabel(r'$q_0$ (W/m$^3$)')
        plt.title('Volumetric heat source profile across the vessel wall')
        plt.legend()
        plt.grid()
        plt.show()

    # ======================================
    # Dimensionless numbers and heat transfer coefficients
    # ======================================
    Pr = (Cp*mu)/k                                                                              #Prandtl number
    Pr_cpp = (Cp_cpp*mu_cpp)/k_cpp                                                              #Prandtl number of the containment water  

                                       
    Re = (rho*v*(D_vess_int-D_barr_ext))/mu                                                     #Reynolds number
    Nu_1 = 0.023*(Re**0.8)*(Pr**0.4)                                                             #Dittus-Boelter equation for forced convection
    h_1 = (Nu_1*k)/(D_vess_int-D_barr_ext)                                                      #W/(m²·K)

    Gr = (rho_cpp**2)*9.81*beta_cpp*DeltaT*(L**3)/(mu_cpp**2)                                   #Grashof number (Uses the external diameter as characteristic length, might wanna use L though?)
    Nu_2 = 0.13*((Gr*Pr_cpp)**(1/3))                                                            #McAdams correlation for natural convection
    h_2 = (Nu_2*k_cpp)/L                                                                        #W/(m²·K)
    R_th_2_tot = (1/(2*np.pi*(R_ext + t_th_ins)*L)) * ((((R_ext + t_th_ins)/k_th_ins)*np.log((R_ext + t_th_ins)/R_ext)) + (1/h_2))                          #Thermal Resistance of the insulation layer + natural convection outside the vessel
    u_2 = 1/(2*np.pi*(R_ext + t_th_ins)*L*R_th_2_tot)                                           #W/(m²·K)   -   Overall heat transfer coefficient outside the vessel

    while True:
        try:
            q_0_flag = int(input("\nDo you want to account for the presence of the volumetric heat source q0 inside the vessel's wall? (1: Yes, 0: No): "))
            if q_0_flag not in (0, 1):
                raise RuntimeError("Invalid input! Please enter either 0 or 1.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)
    q_0 = q_0*q_0_flag

    # ======================================
    # Discretization Check
    # ======================================
    while True:
        try:
            Disc_flag = int(input("Do you want to use a discretization approach along z? (1: Yes, 0: No): "))
            if Disc_flag not in (0, 1):
                raise RuntimeError("Invalid input! Please enter either 0 or 1.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)

    # ======================================
    # 1D Approach: no discretization along z
    # ======================================
    if Disc_flag == 0:
        print("No discretization along z. Assuming constant temperature of the primary fluid T1.")
        while True:
            try:
                T1_flag = int(input("\nWhat temperature do you want to use as T1 to compute C1 and C2? (0: T_in, 1: T_in + 10%, 2: T_in + 20%, 3: T_avg, 4: T_out_avg): "))
                if T1_flag not in (0, 1, 2, 3, 4):
                    raise RuntimeError("Invalid input! Please enter one of the allowed values: 1, 2, 3, 4.")
                break  
            except ValueError:
                print("Please enter a valid integer.")
            except RuntimeError as e:
                print(e)
        while True:
            try:
                adiab_flag = int(input("Apply Adiabatic Outer Wall approximation? (1: Yes, 0: No): "))
                if adiab_flag not in (0, 1):
                    raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                break  
            except ValueError:
                print("Please enter a valid integer.")
            except RuntimeError as e:
                print(e)

        if T1_flag == 0:                #All these temperatures are expressed in K. T_out_max and T_avg_log have been discarded in favor of margins on T_in, to account for transients due to the system's geometry
            T1 = T_in
        elif T1_flag == 1:
            T1 = T_in * 1.1
        elif T1_flag == 2:
            T1 = T_in * 1.2
        elif T1_flag == 3:
            T1 = ((T_in + T_out_avg)/2)
        elif T1_flag == 4:
            T1 = T_out_avg
        
        # ======================================
        # T profile constants for the vessel: general and under adiabatic outer wall approximation (dT/dx = 0 at r = R_ext)
        # ======================================
        if adiab_flag == 0:
            C1 = ((q_0/(k_st*mu_st**2))*(np.exp(-mu_st*t)-1)-(q_0/mu_st)*((1/h_1)+(np.exp(-mu_st*t)/u_2))-(T1-T_cpp))/(t+(k_st/h_1)+(k_st/u_2))
        elif adiab_flag == 1:
            C1 = -((q_0/(k_st*mu_st))*np.exp(-mu_st*t))
        C2 = T1 + (q_0/(h_1*mu_st)) + C1*(k_st/h_1) + (q_0/(k_st*mu_st**2))

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
        DeltaT_1 = T1 - T_vessel(r[0])

        str1 = "\nT1 = T_in has been assumed: a logarithmic mean DeltaT could thus be useful to account for the T profile along z in an approximate way, even though the vessel wall temperature is not constant."
        str2 = "The heat flux computed with the regular DeltaT will still be displayed."
        str3 = "Do you want to adopt such an approach? (1: Yes, 0: No): "
        prompt = "\n".join([str1, str2, str3]) + " "
        if T1_flag == 0:
            while True:
                try:
                    LogDelta_flag = int(input(prompt))
                    if LogDelta_flag not in (0, 1):
                        raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                    break  
                except ValueError:
                    print("Please enter a valid integer.")
                except RuntimeError as e:
                    print(e)
            if LogDelta_flag == 1:
                DeltaT_LM1 = ((T1-T_vessel(r[0]))-(T_out_avg-T_vessel(r[0])))/(np.log((T1-T_vessel(r[0]))/(T_out_avg-T_vessel(r[0]))))    #Log Mean Temperature Difference to account for T change along z, instead of just using T1-T_wall
                q_s1_log = h_1*DeltaT_LM1/1000                                                                                            #kW/m²
        q_s1 = h_1*DeltaT_1/1000                                                                                                          #kW/m²
        q_s2 = u_2*(T_vessel(r[-1])-T_cpp)/1000                                                                                           #kW/m²

        # ======================================
        # Plotting the T profiles
        # ======================================
        while True:
            try:
                T_pl_flag = int(input("\nDo you want to visualize the T profile across the vessel's wall? (1: Yes, 0: No): "))
                if T_pl_flag not in (0, 1):
                    raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                break  
            except ValueError:
                print("Please enter a valid integer.")
            except RuntimeError as e:
                print(e)
        
        if adiab_flag == 0:
            if T_pl_flag == 1:
                plt.figure(figsize=(10,10))
                plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
                plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
                plt.plot(r, T_vessel(r) - 273.15, label='Radial (r) T Profile')
                plt.plot(r_T_vessel_max, T_vessel_max - 273.15,'or',label='Max T')
                plt.axhline(y = T_vessel_avg - 273.15, color='green', label='Wall Average T')
                plt.xlabel('Radius (m)')
                plt.ylabel('T (°C)')
                plt.title('Wall Temperature Profile, Average and Maximum ')
                plt.legend()
                plt.grid()
                plt.show()
            
        elif adiab_flag == 1:
            if T_pl_flag == 1:
                
                # ======================================
                # Under Adiabatic Outer Wall Approximation
                # ======================================
                plt.figure(figsize=(10,10))
                plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
                plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
                plt.plot(r, T_vessel(r) - 273.15, label='Radial (r) T Profile')
                plt.plot(r_T_vessel_max, T_vessel_max - 273.15,'or', label='Max T')
                plt.axhline(y = T_vessel_avg - 273.15, color='green', label='Wall Average T')
                plt.xlabel('Radius (m)')
                plt.ylabel('T (°C)')
                plt.title('Wall Temperature Profile, Average and Maximum under AOW Approximation ')
                plt.legend()
                plt.grid()
                plt.show()
    
        # ======================================
        # Vessel's Wall Thermal stresses computation
        # ======================================
        f_V = lambda r: T_vessel(r)*r

        sigma_r_th_V = np.zeros(dr)
        sigma_t_th_V = np.zeros(dr)
        for i in range(len(r)):
            sigma_r_th_V[i] = (E*alpha_l/(1-nu))*(1/(r[i]**2)) * (( ((r[i]**2)-(R_int**2))/((R_ext**2)-(R_int**2)) ) * simpcomp(f_V, R_int, R_ext, dr) - simpcomp(f_V, R_int, r[i], dr))
            sigma_t_th_V[i] = (E*alpha_l/(1-nu))*(1/(r[i]**2)) * (( (((r[i]**2)+(R_int**2))/((R_ext**2)-(R_int**2)) ) * simpcomp(f_V, R_int, R_ext, dr)) + simpcomp(f_V, R_int, r[i], dr) - T_vessel(r[i])*(r[i]**2))
        sigma_t_th_V_SIMP = lambda r: (E*alpha_l/(1-nu))*(T_vessel_avg - T_vessel(r))                  #Simplified formula assuming average T
        sigma_z_th_V = sigma_r_th_V + sigma_t_th_V                                                     #Superposition principle under the hypothesis of long, hollow cylinder with load-free ends

        sigma_t_th_V_max = max(sigma_t_th_V)
        r_sigma_t_th_V_max = r[np.argmax(sigma_t_th_V)]
        #sigma_t_th_V_max_SIMP = max(sigma_t_th_V_SIMP(r))
        #r_sigma_t_th_V_max_SIMP = r[np.argmax(sigma_t_th_V_SIMP(r))]

        # ======================================
        # Maximum Hoop Thermal Stress in the vessel via design curves
        # ======================================
        for i in range(len(indexes)):
            if mu_st > mu_values[i] and mu_st < mu_values[i+1]:
                mu_L = mu_values[i]
                mu_R = mu_values[i+1]
                #print("Current mu values: ", mu_values[i], mu_st, mu_values[i+1])
                current_L_key, current_R_key = keys_list[i], keys_list[i+1]
                x_points_L, x_points_R = mu_curves[current_L_key][2][:,0], mu_curves[current_R_key][2][:,0]
                y_points_L, y_points_R = mu_curves[current_L_key][2][:,1], mu_curves[current_R_key][2][:,1]

                p_L = np.polyfit(x_points_L, y_points_L, deg = 3)                   #len(y_points_L)-1
                p_R = np.polyfit(x_points_R, y_points_R, deg = 3)

                L_Interpolator = lambda x: np.polyval(p_L, x)
                R_Interpolator = lambda x: np.polyval(p_R, x)

                sigmaT_L = L_Interpolator(R_ext/R_int)                                                     #Interpolated sigmaT coefficient on the left ISO-mu 
                sigmaT_R = R_Interpolator(R_ext/R_int)                                                     #Interpolated sigmaT coefficient on the right ISO-mu 
                sigmaT_eq = lambda x: sigmaT_L + ((sigmaT_R-sigmaT_L)/(mu_R-mu_L))*(x - mu_L)
                sigmaT = sigmaT_eq(mu_st)                                                                  #Double-interpolated (linear) sigmaT coefficient
        
        sigma_t_th_max_DES = sigmaT*(alpha_l*E*q_0)/(k_st*(1-nu)*(mu_st**2))

        # ======================================
        # Plotting the thermal stress profiles
        # ======================================
        while True:
            try:
                sigma_th_pl_flag = int(input("\nDo you want to visualize a plot of the thermal stress profiles in the vessel? (1: Yes, 0: No): "))
                if sigma_th_pl_flag not in (0, 1):
                    raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                break  
            except ValueError:
                print("Please enter a valid integer.")
            except RuntimeError as e:
                print(e)

        if sigma_th_pl_flag == 1:
            plt.figure(figsize=(10,10))
            plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
            plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
            plt.plot(r, sigma_r_th_V, linewidth='0.75', label='Radial (r) Thermal Stress Profile')
            plt.plot(r, sigma_t_th_V, linewidth='0.75', label='Hoop (θ) Thermal Stress Profile')
            #plt.plot(r, sigma_t_th_SIMP(r), label='Simplified Hoop (θ) Thermal Stress Profile')
            plt.plot(r, sigma_z_th_V, color='green', linewidth='0.5', label='Axial (z) Thermal Stress Profile')
            plt.axhline(y = 0, color='black', linewidth='1', label='y=0')
            plt.plot(r_sigma_t_th_V_max, sigma_t_th_V_max,'or', label='Max Hoop Stress')
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
        # Principal stresses sum and elastic regime verification in the vessel
        # ======================================
        sigma_r_totM = sigma_rM_cyl + sigma_r_th_V
        sigma_t_totM = sigma_tM_cyl + sigma_t_th_V
        sigma_z_totM = sigma_zM_cyl + sigma_z_th_V
        
        sigma_r_totL = sigma_rL + sigma_r_th_V
        sigma_t_totL = sigma_tL + sigma_t_th_V
        sigma_z_totL = sigma_zL + sigma_z_th_V
        
        # ============================ 
        # Vessel Comparison stress - Guest-Tresca Theory - Mariotte/Lamé + Thermal stresses
        # ============================
        sigma_cTR_M = np.max([abs(sigma_t_totM - sigma_r_totM), abs(sigma_z_totM - sigma_r_totM), abs(sigma_t_totM - sigma_z_totM)])
        sigma_cTR_L = np.max([abs(sigma_t_totL - sigma_r_totL), abs(sigma_z_totL - sigma_r_totL), abs(sigma_t_totL - sigma_z_totL)])

        # ============================ 
        # Vessel Comparison stress - Von Mises Theory - Mariotte/Lamé + Thermal stresses
        # ============================
        sigma_cVM_M = max(np.sqrt(0.5*((sigma_r_totM - sigma_t_totM)**2 + (sigma_t_totM - sigma_z_totM)**2 + (sigma_z_totM - sigma_r_totM)**2)))
        sigma_cVM_L = max(np.sqrt(0.5*((sigma_r_totL - sigma_t_totL)**2 + (sigma_t_totL - sigma_z_totL)**2 + (sigma_z_totL - sigma_r_totL)**2)))

        # ======================================
        # Plotting the maximum thermal stress via the design curves
        # ======================================
        while True:
            try:
                des_pl_flag = int(input("\nDo you want to visualize a plot of the design curves and the maximum thermal stress in the vessel? (1: Yes, 0: No): "))
                if des_pl_flag not in (0, 1):
                    raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                break  
            except ValueError:
                print("Please enter a valid integer.")
            except RuntimeError as e:
                print(e)

        if des_pl_flag == 1:
            plt.figure(figsize=(10,10))
            plt.plot(ba_ratio_plot, L_Interpolator(ba_ratio_plot), 'k', label=f'Iso-mu = {mu_L} 1/m')
            plt.plot(ba_ratio_plot, R_Interpolator(ba_ratio_plot), 'k', label=f'Iso-mu = {mu_R} 1/m')
            plt.plot(R_ext/R_int, sigmaT,'or', label=r'Current $\sigma$$_T$')
            plt.xlabel('R$_{ext}$/R$_{int}$')
            plt.ylabel(r'$\sigma$$_T$')
            plt.title('Design curves')
            plt.legend()
            plt.grid()
            plt.show()
        
        # ============================ 
        # Yield Stress and Stress Intensity Data Interpolation
        # ============================
        T_des_vessel = T_vessel_avg                                                     #K  -   Check in the HARVEY/Thermomechanics Chapter how to choose the design T
        T_des_vessel_C = T_des_vessel - 273.15                                          #°C
        p_yield = np.polyfit(T_thr, sigma_y, deg = len(T_thr)-1)
        p_intensity = np.polyfit(T_thr, sigma_in, deg = len(T_thr)-1)
        
        Yield_Interpolator = lambda x: np.polyval(p_yield, x)                           #Yield Stress Interpolation Polynomial (n-1)
        Yield_CubicSpline = interpolate.CubicSpline(T_thr, sigma_y)                     #Yield Stress Cubic Spline Interpolation
        Yield_stress = Yield_CubicSpline(T_des_vessel_C)
        
        Intensity_Interpolator = lambda x: np.polyval(p_intensity, x)                   #Stress Intensity Interpolation Polynomial (n-1)
        Intensity_CubicSpline = interpolate.CubicSpline(T_thr, sigma_in)                #Stress Intenisty Cubic Spline Interpolation
        Stress_Intensity = Intensity_CubicSpline(T_des_vessel_C)
        sigma_allowable = 1.5 * Stress_Intensity  #MPa

        # ======================================
        # Without thermal shield
        # ======================================
        while True:
            try:
                Interp_pl_flag = int(input("\nDo you want to visualize a plot of the Yield Stress and Stress Intensity as given by ASME for the vessel? (1: Yes, 0: No): "))
                if Interp_pl_flag not in (0, 1):
                    raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                break  
            except ValueError:
                print("Please enter a valid integer.")
            except RuntimeError as e:
                print(e)
        
        if max(T_thr) > T_des_vessel_C:
            Tplot = np.linspace(min(T_thr), max(T_thr), 1000)
        else:
            Tplot = np.linspace(min(T_thr), T_des_vessel_C, 1000)
        
        if Interp_pl_flag == 1:
            
            # ============================ 
            # Yield Stress and Stress Intensity Data Plots  -   Vessel
            # ============================
            plt.figure(figsize = (12,10))
            plt.subplot(1,2,1)
            plt.plot(T_thr, sigma_y, 'sk', label = 'Yield Stress Data')
            plt.plot(Tplot, Yield_Interpolator(Tplot), '--', color = 'orange', label = 'Yield Stress n-1 Interpolation')
            plt.plot(Tplot, Yield_CubicSpline(Tplot), 'green', label = 'Yield Stress Cubic Spline Interpolation')
            plt.plot(T_des_vessel_C, Yield_stress, '--or', label = r'Current Vessel Yield Stress $\sigma$$_y$')
            plt.xlabel("Temperature (°C)")
            plt.ylabel(r"Yield Stress $\sigma$$_y$")
            plt.title("Yield Stress Data and Interpolation VS Temperature", fontsize = 10)
            plt.legend()
            plt.grid()
            plt.tight_layout()
            
            plt.subplot(1,2,2)
            plt.plot(T_thr, sigma_in, 'sk', label = 'Stress Intensity Data')
            plt.plot(Tplot, Intensity_Interpolator(Tplot), '--', color = 'orange', label = 'Stress Intensity n-1 Interpolation')
            plt.plot(Tplot, Intensity_CubicSpline(Tplot), 'green', label = 'Stress Intensity Cubic Spline Interpolation')
            plt.plot(T_des_vessel_C, Stress_Intensity, '--or', label = r'Current Vessel Stress Intensity $\sigma$$_m$')
            plt.xlabel("Temperature (°C)")
            plt.ylabel(r"Stress Intensity $\sigma$$_m$")
            plt.title("Stress Intensity Data and Interpolation VS Temperature", fontsize = 10)
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.show()
    
        # ============================ 
        # Sizing of a thick cylinder under external pressure
        # ============================
        if Mar_criterion > 5:
            while True:
                try:
                    ThinTubes_flag = int(input("\nThe vessel's wall can be considered thin. Are you interested in the thin tube limits for Elastic Instability and Plastic Collapse? (1: Yes, 0: No): "))
                    if ThinTubes_flag not in (0, 1):
                        raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                    break  
                except ValueError:
                    print("Please enter a valid integer.")
                except RuntimeError as e:
                    print(e)

            if ThinTubes_flag == 1:
                p_E_fun = lambda Dt: 2 * (E/(1-(nu**2))) * (1/(Dt**3))              #Elastic Instability Limit for Thin Tubes
                p_0_fun = lambda Dt: 2 * Yield_stress * 1/Dt                        #Plastic Collapse Limit for Thin Tubes

            elif ThinTubes_flag == 0:
                print("Skipping thin tube limits.")
        else:
            print("\nThe cylinder can't be considered thin. Skipping thin tube limits.")
            ThinTubes_flag = 0

        # ============================ 
        # Corradi Design Procedure
        # ============================
        if ThinTubes_flag == 1:
            while True:
                try:
                    Corradi_flag = int(input("\nThe thin tube limits were adopted. Are you interested in the more general Corradi Design Procedure? (1: Yes, 0: No): "))
                    if Corradi_flag not in (0, 1):
                        raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                    break  
                except ValueError:
                    print("Please enter a valid integer.")
                except RuntimeError as e:
                    print(e)
            
        elif ThinTubes_flag == 0:
            print("Adopting Corradi Design Procedure.")
            Corradi_flag = 1

        q_E_fun = lambda Dt: 2 * (E/(1-(nu**2))) * (1/(Dt*((Dt-1)**2)))     #Elastic Instability Limit for Thick Tubes
        q_0_fun = lambda Dt: 2 * Yield_stress * 1/Dt * (1+(1/(2*Dt)))       #Plastic Collapse Limit for Thick Tubes
        Dt_Crit_Ratio = np.sqrt(E/(Yield_stress*(1-(nu**2))))
        Current_Slenderness = (D_vess_int+2*t)/t
        Dt_ratio_plot = np.linspace(2,50,1000)

        if Corradi_flag == 1:
            # ============================ 
            # Corradi Design Procedure
            # ============================
            while True:
                try:
                    s = float(input("Please enter a safety factor between 1.5 and 2 for the Corradi design procedure: "))
                    if s < 1.5 or s > 2:
                        raise RuntimeError("Invalid input! Please enter a safety factor between 1.5 and 2.")
                    break  
                except ValueError:
                    print("Please enter a valid float.")
                except RuntimeError as e:
                    print(e)
            
            def Corradi(Slenderness):
                if isinstance(Slenderness, np.ndarray):
                    mu = np.zeros(len(Slenderness))
                    Z = lambda Dt: (np.sqrt(3)/4) * (2*Dt + 1) * W                  #Accounts for ovality
                    q_U = lambda Dt: q_0_fun(Dt)/np.sqrt(1+(Z(Dt)**2))
                    q_L = lambda Dt: (1/2) * (q_0_fun(Dt) + q_E_fun(Dt)*(1 + Z(Dt)) - np.sqrt(((q_0_fun(Dt) + q_E_fun(Dt)*(1 + Z(Dt)))**2)-(4 * q_0_fun(Dt) * q_E_fun(Dt))))
                    
                    for i in range(len(mu)):
                        if q_0_fun(Slenderness[i])/q_E_fun(Slenderness[i]) < 0.04:
                            mu[i] = 1
                        elif 0.04 <= q_0_fun(Slenderness[i])/q_E_fun(Slenderness[i]) <= 0.7:
                            mu[i] = (0.35 * np.log(q_E_fun(Slenderness[i])/q_0_fun(Slenderness[i]))) - 0.125
                        elif q_0_fun(Slenderness[i])/q_E_fun(Slenderness[i]) > 0.7:
                            mu[i] = 0
                        
                    q_C = mu*q_U(Slenderness) + (1-mu)*q_L(Slenderness)
                    q_a = q_C/s
                else:
                    raise TypeError("The 1st input must be a numpy array.")
                if len(q_C) == 1:
                    q_C = q_C.item()
                if len(q_a) == 1:
                    q_a = q_a.item()
                if len(mu) == 1:
                    mu = mu.item()
                return (q_C, q_a, s, mu)
            
            # ============================ 
            # Corradi Design Procedure Results
            # ============================
            Corradi_vessel = Corradi(np.array([Current_Slenderness]))
        
        elif Corradi_flag == 0:
            print("Skipping Corradi Design Procedure.")
        
        # ============================ 
        # Elastic instability and plastic collapse curves
        # ============================
        if ThinTubes_flag == 1 and Corradi_flag == 0:
            while True:
                try:
                    Collapse_pl_flag = int(input("\nDo you want to visualize the buckling and plastic collapse curves for thin and thick tubes? (1: Yes, 0: No): "))
                    if Collapse_pl_flag not in (0, 1):
                        raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                    break  
                except ValueError:
                    print("Please enter a valid integer.")
                except RuntimeError as e:
                    print(e)
            
            if Collapse_pl_flag == 1:
                
                # ============================ 
                # Plastic collapse and buckling Plots
                # ============================
                plt.figure(figsize = (8, 8))
                plt.semilogy(Dt_ratio_plot, p_E_fun(Dt_ratio_plot), 'blue', label='p$_E$')
                plt.semilogy(Dt_ratio_plot, q_E_fun(Dt_ratio_plot), '--b', label='q$_E$')
                plt.semilogy(Dt_ratio_plot, p_0_fun(Dt_ratio_plot), 'red', label='p$_0$')
                plt.semilogy(Dt_ratio_plot, q_0_fun(Dt_ratio_plot), '--r', label='q$_0$')
                plt.axvline(x = Dt_Crit_Ratio, color = 'black', linewidth = '3', label = 'Critical Slenderness')
                plt.axvline(x = Current_Slenderness, color = 'green', linewidth = '3', label = 'Current Vessel Slenderness')
                plt.xlabel("Geometrical Slenderness D/t")
                plt.ylabel("Theoretical Limit Values (MPa)")
                plt.title("Plastic Collapse and Buckling Curves")
                plt.legend()
                plt.grid()
                plt.show()

        elif ThinTubes_flag == 1 and Corradi_flag == 1:
            while True:
                try:
                    Collapse_pl_flag = int(input("\nDo you want to visualize the buckling and plastic collapse curves for thin and thick tubes and the Corradi curve? (1: Yes, 0: No): "))
                    if Collapse_pl_flag not in (0, 1):
                        raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                    break  
                except ValueError:
                    print("Please enter a valid integer.")
                except RuntimeError as e:
                    print(e)
            
            if Collapse_pl_flag == 1:
                
                # ============================ 
                # Plastic collapse and buckling Plots
                # ============================
                plt.figure(figsize = (8, 8))
                plt.subplot(1,2,1)
                plt.semilogy(Dt_ratio_plot, p_E_fun(Dt_ratio_plot), 'blue', label='p$_E$')
                plt.semilogy(Dt_ratio_plot, q_E_fun(Dt_ratio_plot), '--b', label='q$_E$')
                plt.semilogy(Dt_ratio_plot, p_0_fun(Dt_ratio_plot), 'red', label='p$_0$')
                plt.semilogy(Dt_ratio_plot, q_0_fun(Dt_ratio_plot), '--r', label='q$_0$')
                plt.semilogy(Dt_ratio_plot, Corradi(Dt_ratio_plot)[0], 'orange', label='Corradi q$_c$')
                plt.axvline(x = Dt_Crit_Ratio, color = 'black', linewidth = '3', label = 'Critical Slenderness')
                plt.axvline(x = Current_Slenderness, color = 'green', linewidth = '3', label = 'Current Vessel Slenderness')
                plt.xlabel("Geometrical Slenderness D/t")
                plt.ylabel("Theoretical Limit Values (MPa)")
                plt.title("Plastic Collapse and Buckling Curves")
                plt.legend()
                plt.grid()
                plt.tight_layout()

                plt.subplot(1,2,2)
                plt.plot(Dt_ratio_plot, Corradi(Dt_ratio_plot)[3], 'k', label=r'Corradi $\mu$')
                plt.xlabel("Geometrical Slenderness D/t")
                plt.ylabel(r"Corradi $\mu$")
                plt.title(r"$\mu$ coefficient - Corradi Procedure")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.show()
        
        # ============================ 
        # Final Results
        # ============================
        print("\n\n\n###################################################### Final  Results ######################################################")
        print("\nCurrent vessel wall thickness: %.3f m" %t)
        
        # ============================ 
        # Heat Transfer Results
        # ============================
        print("\n################################################## Heat transfer results ###################################################")
        print("\nVolumetric heat source at the vessel inner surface: %.3f W/m³" %q_iii(r[0]))
        print("Volumetric heat source at the vessel-insulation interface: %.3f W/m³" %q_iii(r[-1]))
        print("\nHeat transfer coefficient h1 = %.3f W/(m²·K)" %h_1)
        print("Heat transfer coefficient h2 = %.3f W/(m²·K)" %h_2)
        print("Overall heat transfer coefficient outside the vessel u2 = %.3f W/(m²·K)" %u_2)
        if LogDelta_flag == 1:
            print("\nThermal power flux on the inner vessel surface - Logarithmic Mean DeltaT Approach: %.3f kW/m²" %q_s1_log)
        print("\nThermal power flux on the inner vessel surface: %.3f kW/m²" %q_s1)
        print("Thermal power flux on the outer vessel surface: %.3f kW/m²" %q_s2)
        
        # ============================ 
        # Temperature Results
        # ============================
        print("\n####################################################### Temperatures #######################################################")
        if adiab_flag == 0:
            print("\nAverage Vessel Temperature (numerical integration): %.3f °C" %(T_vessel_avg - 273.15))
            #print("Average Vessel Temperature (analytical integration): %.3f °C" %T_vessel_avg_2)
            print("Maximum Vessel Temperature: %.3f °C at r = %.3f m" %(T_vessel_max - 273.15, r_T_vessel_max))
            print("Vessel Temperature at the inner surface: %-3f °C at r = %.3f m" %(T_vessel(r)[0] - 273.15, r[0]))
            print("Vessel Temperature at the outer surface: %-3f °C at r = %.3f m" %(T_vessel(r)[-1] - 273.15, r[-1]))
        elif adiab_flag == 1:
            print("\nAverage Vessel Temperature under Adiabatic Outer Wall approximation (numerical integration): %.3f °C" %(T_vessel_avg - 273.15))
            #print("Average Vessel Temperature under Adiabatic Outer Wall approximation (analytical integration): %.3f °C" %T_vessel_avg_2)
            print("Maximum Vessel Temperature under Adiabatic Outer Wall approximation: %.3f °C at r = %.3f m" %(T_vessel_max - 273.15, r_T_vessel_max))
            print("Vessel Temperature at the inner surface under Adiabatic Outer Wall approximation: %-3f °C at r = %.3f m" %(T_vessel(r)[0] - 273.15, r[0]))
            print("Vessel Temperature at the outer surface under Adiabatic Outer Wall approximation: %-3f °C at r = %.3f m" %(T_vessel(r)[-1] - 273.15, r[-1]))
        if (T_vessel_max - 273.15) > T_creep:
            print("\nWARNING: The maximum vessel temperature T = %.3f °C exceeds the creep threshold temperature of %d °C!" %(T_vessel_max - 273.15, T_creep))
        
        # ============================ 
        # Stress Results
        # ============================
        print("\n######################################################### Stresses #########################################################")
        print("\nMaximum Thermal Hoop Stress in the vessel: %.3f Mpa at r = %.3f m" %(sigma_t_th_V_max, r_sigma_t_th_V_max))
        #print("Maximum Thermal Hoop Stress (Simplified formula): %.3f Mpa at r = %.3f m" %(sigma_t_th_V_max_SIMP, r_sigma_t_th_V_max_SIMP))
        print("Maximum thermal hoop stress via design curves: %.3f MPa" %sigma_t_th_max_DES)
        
        print("\nGuest-Tresca Equivalent Stress in the vessel - Mariotte solution: %.3f Mpa" %sigma_cTR_M)
        print("Guest-Tresca Equivalent Stress in the vessel - Lamé solution: %.3f Mpa" %sigma_cTR_L)
        
        print("\nVon Mises Equivalent Stress in the vessel - Mariotte solution: %.3f Mpa" %sigma_cVM_M)
        print("Von Mises Equivalent Stress in the vessel - Lamé solution: %.3f Mpa" %sigma_cVM_L)
        
        print("\nFor a design vessel temperature of %.3f °C: " %T_des_vessel_C)
        print('Yield Stress: Sy'," = %.3f MPa" %Yield_stress)
        print('Stress Intensity: Sm'," = %.3f MPa" %Stress_Intensity)
        print("Allowable Stress: %.3f MPa" %sigma_allowable)
        
        # ============================ 
        # Buckling Results
        # ============================
        print("\n######################################################### Buckling #########################################################")
        print("\nAccording to the Corradi Design Procedure:")
        print("The theoretical limit for collapse pressure, accounting for ovality, is: q_c = %.3f MPa = %.3f bar" %(Corradi_vessel[0], 10*Corradi_vessel[0]))
        print("A safety factor s = %.3f was assumed. \nThe allowable external pressure is thus: q_a = %.3f MPa = %.3f bar" %(Corradi_vessel[2], Corradi_vessel[1], 10*Corradi_vessel[1]))
        print("\n############################################################################################################################")
        
        if (P_cpp < 10*Corradi_vessel[1] and sigma_cTR_M < sigma_allowable and sigma_cTR_L < sigma_allowable):
            print("\nThe given external pressure of %.3f bar is lower than the allowable pressure of %.3f bar" %(P_cpp, 10*Corradi_vessel[1]))
            print("The comparison stress according to Tresca-Lamé Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa\nThe comparison stress according to Tresca-Mariotte Sc = %.3f MPa is also lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_L, sigma_allowable, sigma_cTR_M, sigma_allowable))
            if creep_flag_V == 1:
                print("\nCreep might occur in the vessel due to high temperatures. Either an additional thermal shield, an increased thickness or both are required.")
            elif creep_flag_V == 0:
                print("There is no risk of thermal creep occurring in the vessel.")
                print("The vessel's integrity is ensured: the design is correct!")
            print("\n############################################################################################################################")
            
        elif (P_cpp > 10*Corradi_vessel[1]):
            print("\nThe given external pressure of %.3f bar is higher than the allowable pressure of %.3f bar: a change in thickness is required!" %(P_cpp, 10*Corradi_vessel[1]))
            print("\n############################################################################################################################")
            
        elif (sigma_cTR_M > sigma_allowable or sigma_cTR_L > sigma_allowable):
            print("\nEither the Tresca-Mariotte comparison stress Sc = %.3f MPa or the Tresca-Lamè comparison stress Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_M,sigma_cTR_L,sigma_allowable))
            print("\n############################################################################################################################")
            
    # ======================================
    # Discretization along z
    # ======================================
    elif Disc_flag == 1:
        
        # ======================================
        # T discretization along z
        # ======================================
        dz = 1000
        T_z = np.linspace(T_in, T_out_avg, dz)
        while True:
            try:
                adiab_flag = int(input("\nApply Adiabatic Outer Wall approximation? (1: Yes, 0: No): "))
                if adiab_flag not in (0, 1):
                    raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                break  
            except ValueError:
                print("Please enter a valid integer.")
            except RuntimeError as e:
                print(e)

        if adiab_flag == 0:
            C1 = ((q_0/(k_st*mu_st**2))*(np.exp(-mu_st*t)-1)-(q_0/mu_st)*((1/h_1)+(np.exp(-mu_st*t)/u_2))-(T_z - T_cpp))/(t+(k_st/h_1)+(k_st/u_2))      # Make C1 a 1D array aligned with T_z (avoid wrapping in another dimension)
        elif adiab_flag == 1:
            C1 = np.full_like(T_z, -((q_0/(k_st*mu_st)) * np.exp(-mu_st*t)))                                                                                       # Constant C1 for all z points -> replicate to match T_z shape
        C2 = T_z + (q_0/(h_1*mu_st)) + C1 * (k_st/h_1) + (q_0/(k_st*mu_st**2))                                                                                     # C2 should also be a 1D array matching T_z
        
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
                print("progress: %.3i/%.3i" %(i, dz))
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
        DeltaT_1 = np.zeros(len(T_z))
        q_s1 = np.zeros(len(T_z))
        q_s2 = np.zeros(len(T_z))
        for i in range(len(T_z)):
            DeltaT_1[i] = T_z[i] - T_vessel_Mat[i, 0]                               #Between the vessel's inner surface and the primary fluid
            q_s1[i] = h_1*DeltaT_1[i]/1000                                          #kW/m²
            q_s2[i] = u_2*(T_vessel_Mat[i, -1]-T_cpp)/1000               #kW/m²
        
        print("\nThermal power flux on the inner vessel surface: \nMin: %.3f kW/m² \nMax: %.3f kW/m²" %(np.min(q_s1),np.max(q_s1)))
        print("\nThermal power flux on the outer vessel surface: \nMin: %.3f kW/m² \nMax: %.3f kW/m²" %(np.min(q_s2),np.max(q_s2)))

        # ======================================
        # Plotting the wall T profiles
        # ======================================
        if adiab_flag == 0:
            
            # ======================================
            # Wall T(T_z, r) map
            # ======================================
            R_mesh, T_z_mesh = np.meshgrid(r, T_z - 273.15)                                                      #Shapes (Nz, Nr)
            plt.figure(figsize=(15,10))
            plt.subplot(1,2,1)
            pcm = plt.pcolormesh(R_mesh, T_z_mesh, T_vessel_Mat - 273.15, shading='auto', cmap='hot')   #Or 'hot','plasma','viridis'
            plt.colorbar(pcm, label='T (°C)')
            plt.xlabel('Radius (m)')
            plt.ylabel('T$_z$ (°C)')
            plt.title('Wall Temperature Map (r vs T$_z$)')
            plt.tight_layout()
            
            # ======================================
            # T_avg and T_max profiles as T_z grows
            # ======================================
            plt.subplot(1,2,2)
            plt.plot(T_z - 273.15, T_vessel_max_arr - 273.15, 'r', label='Max T Axial (z) Profile')              #The r position of T_max is always the same in this approach!
            plt.plot(T_z - 273.15, T_vessel_avg_arr - 273.15, 'k', label='Average T Axial (z) Profile')
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
            R_mesh, T_z_mesh = np.meshgrid(r, T_z - 273.15)
            plt.figure(figsize=(15,10))
            plt.subplot(1,2,1)
            pcm = plt.pcolormesh(R_mesh, T_z_mesh, T_vessel_Mat - 273.15, shading='auto', cmap='hot')
            plt.colorbar(pcm, label='T (°C)')
            plt.xlabel('Radius (m)')
            plt.ylabel('T$_z$ (°C)')
            plt.title('Wall Temperature Map under AOW Approximation (r vs T$_z$)')
            plt.tight_layout()
            
            # ======================================
            # T_avg and T_max profiles as T_z grows
            # ======================================
            plt.subplot(1,2,2)
            plt.plot(T_z - 273.15, T_vessel_max_arr - 273.15, 'r', label='Max T Axial (z) Profile')
            plt.plot(T_z - 273.15, T_vessel_avg_arr - 273.15, 'k', label='Average T Axial (z) Profile')
            plt.xlabel('T$_z$ (°C)')
            plt.ylabel('T (°C)')
            plt.title('Maximum and Average Wall Temperature Profiles as T$_z$ grows under AOW Approximation')
            plt.legend()
            plt.grid()
            plt.show()
        
        # ======================================
        # Plotting the thermal stress profiles
        # ======================================
        R_mesh, T_z_mesh = np.meshgrid(r, T_z - 273.15)   # shapes (Nz, Nr)
        plt.figure(figsize=(20,20))
        plt.subplot(1,4,1)
        pcm = plt.pcolormesh(R_mesh, T_z_mesh, sigma_r_th, shading='auto', cmap='viridis')
        plt.colorbar(pcm, label=r'$\sigma$ (MPa)')
        plt.xlabel('Radius (m)')
        plt.ylabel('T$_z$ (°C)')
        plt.title('Radial Stress Map (r vs T$_z$)')
        plt.tight_layout()

        plt.subplot(1,4,2)
        pcm = plt.pcolormesh(R_mesh, T_z_mesh, sigma_t_th, shading='auto', cmap='viridis')
        plt.colorbar(pcm, label=r'$\sigma$ (MPa)')
        plt.xlabel('Radius (m)')
        plt.ylabel('T$_z$ (°C)')
        plt.title('Hoop Stress Map (r vs T$_z$)')
        plt.tight_layout()

        plt.subplot(1,4,3)
        pcm = plt.pcolormesh(R_mesh, T_z_mesh, sigma_t_th_SIMP, shading='auto', cmap='viridis')
        plt.colorbar(pcm, label=r'$\sigma$ (MPa)')
        plt.xlabel('Radius (m)')
        plt.ylabel('T$_z$ (°C)')
        plt.title('Simplified Hoop Stress Map (r vs T$_z$)')
        plt.tight_layout()

        plt.subplot(1,4,4)
        pcm = plt.pcolormesh(R_mesh, T_z_mesh, sigma_z_th, shading='auto', cmap='viridis')
        plt.colorbar(pcm, label=r'$\sigma$ (MPa)')
        plt.xlabel('Radius (m)')
        plt.ylabel('T$_z$ (°C)')
        plt.title('Axial Stress Map (r vs T$_z$)')
        plt.tight_layout()
        plt.show()

# =============================================================================================================================================================
# PURELY THERMAL PROBLEM - POWER IMPOSED - THERMAL SHIELD
# =============================================================================================================================================================
elif TS_flag == 1:

    t_shield_user = 0.001           #Initial guess for the thermal shield thickness

    while True:
        try:
            User_D_flag = float(input("\nWhat position of the thermal shield do you want to consider? (2: Arbitrary, 1: Middle, 0: Equal areas): "))
            if User_D_flag not in (0, 1, 2):
                raise RuntimeError("Invalid input! Please enter either 0, 1 or 2.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)

    if User_D_flag == 2:            #Allows the user to assume the default, middle position for the thermal shield, position it himself or choose the position granting equal areas
        while True:
            try:
                D_shield_int = float(input("\nPlease enter the initial thermal shield inner diameter (m) to choose its position: "))
                if (D_shield_int < D_barr_ext) or (D_shield_int/2 + t_shield_user) > D_vess_int/2:
                    raise RuntimeError("The thermal shield is either starting inside the barrel or clipping inside the vessel.")
                break  
            except ValueError:
                print("Please enter a valid float.")
            except RuntimeError as e:
                print(e)
        R_shield_int = D_shield_int/2
        
    elif User_D_flag == 1:
        print("Assuming middle thermal shield position.")
        R_shield_int = D_barr_ext/2 + (D_vess_int - D_barr_ext)/4 - t_shield_user/2
         
    elif User_D_flag == 0:
        A_eq = 1
        B_eq = t_shield_user
        C_eq = (t_shield_user**2)/2 - (R_int**2)/2 - (R_barr_ext**2)/2
        Delta_eq = B_eq**2 - 4*A_eq*C_eq
        R_shield_int = (-B_eq + np.sqrt(Delta_eq))/(2*A_eq)
        D_shield_int = 2*R_shield_int

    print("No discretization along z. Assuming constant temperature of the primary fluid T1.")
    while True:
        try:
            T1_flag = int(input("\nWhat temperature do you want to use as T1 to compute C1 and C2? (0: T_in, 1: T_in + 10%, 2: T_in + 20%, 3: T_avg, 4: T_out_avg): "))
            if T1_flag not in (0, 1, 2, 3, 4):
                raise RuntimeError("Invalid input! Please enter one of the allowed values: 1, 2, 3, 4.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)
    while True:
        try:
            adiab_flag = int(input("Apply Adiabatic Outer Wall approximation? (1: Yes, 0: No): "))
            if adiab_flag not in (0, 1):
                raise RuntimeError("Invalid input! Please enter either 0 or 1.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)

    # =============================================================================================================================================================
    # Thermal Shield Thickness Iterative Computation
    # =============================================================================================================================================================
    final_flag = 0
    counter = 0
    counter_vessel = 0
    N_max = 10000
    eps = 1e-7                              #np.finfo(float).eps (excessive)

    t_shield = t_shield_user - 0.001
    t_shield_max = 0.1                      #arbitrary and perhaps excessive
    t_vessel_max = 0.3
    
    p_yield = np.polyfit(T_thr, sigma_y, deg = len(T_thr)-1)
    p_intensity = np.polyfit(T_thr, sigma_in, deg = len(T_thr)-1)
    
    Yield_Interpolator = lambda x: np.polyval(p_yield, x)                           #Yield Stress Interpolation Polynomial (n-1)
    Yield_CubicSpline = interpolate.CubicSpline(T_thr, sigma_y)                     #Yield Stress Cubic Spline Interpolation
    Intensity_Interpolator = lambda x: np.polyval(p_intensity, x)                   #Stress Intensity Interpolation Polynomial (n-1)
    Intensity_CubicSpline = interpolate.CubicSpline(T_thr, sigma_in)                #Stress Intenisty Cubic Spline Interpolation
    Dt_ratio_plot = np.linspace(2,50,1000)
    
    while True:
        try:
            s = float(input("Please enter a safety factor between 1.5 and 2 for the Corradi design procedure: "))
            if s < 1.5 or s > 2:
                raise RuntimeError("Invalid input! Please enter a safety factor between 1.5 and 2.")
            break  
        except ValueError:
            print("Please enter a valid float.")
        except RuntimeError as e:
            print(e)

    while final_flag == 0:
        t_shield += 0.001
        counter += 1
        print("Iteration no. %d" %counter)
        if counter > N_max:
            print("Exceeded maximum number of iterations: %d. Exiting the loop." %N_max)
            break
        if t > t_vessel_max:
            print("Vessel thickness exceeds feasibility margin. Exiting the loop.")
            break
        if User_D_flag == 2 or User_D_flag == 0:        #If the thermal shield is not in the middle, geometrical constraints are present: the thermal shield must not bump into the vessel
            if t_shield > t_shield_max or (D_shield_int/2 + t_shield) > D_vess_int/2:
                print("Ran into excessive thermal shield thickness or bumped into the vessel. Adding 1cm to the vessel thickness instead. Restarting...")
                t += 0.01
                t_shield = t_shield_user - 0.001
                counter_vessel += 1
                continue
        elif User_D_flag == 1:      #If the thermal shield is in the middle, only its thickness must be checked: no geometrical constraints
            if t_shield > t_shield_max:
                print("Ran into excessive thermal shield thickness. Adding 1cm to the vessel thickness instead. Restarting...")
                t += 0.01
                t_shield = t_shield_user - 0.001
                counter_vessel += 1
                continue

        if User_D_flag == 2:
            R_shield_int = D_shield_int/2
            
        elif User_D_flag == 1:
            R_shield_int = D_barr_ext/2 + (D_vess_int - D_barr_ext)/4 - t_shield/2
            D_shield_int = 2*R_shield_int
            
        elif User_D_flag == 0:
            A_eq = 1
            B_eq = t_shield
            C_eq = (t_shield**2)/2 - (R_int**2)/2 - (R_barr_ext**2)/2
            Delta_eq = B_eq**2 - 4*A_eq*C_eq
            R_shield_int = (-B_eq + np.sqrt(Delta_eq))/(2*A_eq)
            D_shield_int = 2*R_shield_int
            
        R_shield_ext = R_shield_int + t_shield
        D_shield_ext = 2*R_shield_ext
        
        r_S = np.linspace(R_shield_int, R_shield_ext, dr)
        Phi_0S = Phi_0                                                #All gamma rays reach the shield, not the vessel

        # ======================================
        # Dimensionless numbers and heat transfer coefficients
        # ======================================
        A_int_S = np.pi*((R_shield_int**2) - (R_barr_ext**2))                                       #Inner area crossed by the primary fluid
        A_ext_S = np.pi*((R_int**2) - (R_shield_ext**2))                                            #Outer area crossed by the primary fluid
        v_int = v_flr/A_int_S                                                                       #Inner coolant velocity
        v_ext = v_flr/A_ext_S
        
        Pr = (Cp*mu)/k                                                                              #Prandtl number
        Pr_cpp = (Cp_cpp*mu_cpp)/k_cpp                                                              #Prandtl number of the containment water
        
        Re_int = (rho*v_int*(D_shield_int - D_barr_ext))/mu                                         #Inner hydraulic diameter                                                     
        Nu_1_int = 0.023*(Re_int**0.8)*(Pr**0.4)                                                             
        h_1_int = (Nu_1_int*k)/(D_shield_int - D_barr_ext)
        
        Re_ext = (rho*v_ext*(D_vess_int - D_shield_ext))/mu                                         #Outer hydraulic diameter                                                     
        Nu_1_ext = 0.023*(Re_ext**0.8)*(Pr**0.4)                                                             
        h_1_ext = (Nu_1_ext*k)/(D_vess_int - D_shield_ext)
                                                                                 
        Gr = (rho_cpp**2)*9.81*beta_cpp*DeltaT*(L**3)/(mu_cpp**2)                                   #Grashof number (Uses the external diameter as characteristic length, might wanna use L though?)
        Nu_2 = 0.13*((Gr*Pr_cpp)**(1/3))                                                            #McAdams correlation for natural convection
        h_2 = (Nu_2*k_cpp)/L                                                                        #W/(m²·K)
        R_th_2_tot = (1/(2*np.pi*(R_ext + t_th_ins)*L)) * ((((R_ext + t_th_ins)/k_th_ins)*np.log((R_ext + t_th_ins)/R_ext)) + (1/h_2))                          #Thermal Resistance of the insulation layer + natural convection outside the vessel
        u_2 = 1/(2*np.pi*(R_ext + t_th_ins)*L*R_th_2_tot)                                           #W/(m²·K)   -   Overall heat transfer coefficient outside the vessel

        if T1_flag == 0:                #All these temperatures are expressed in K. T_out_max and T_avg_log have been discarded in favor of margins on T_in, to account for transients due to the system's geometry
            T1 = T_in
        elif T1_flag == 1:
            T1 = T_in * 1.1
        elif T1_flag == 2:
            T1 = T_in * 1.2
        elif T1_flag == 3:
            T1 = ((T_in + T_out_avg)/2)
        elif T1_flag == 4:
            T1 = T_out_avg

        R_ext = R_int + t
        r = np.linspace(R_int, R_ext, dr)

        # ======================================
        # Radiation-induced heating in the thermal shield
        # ======================================
        Phi_S = lambda r: Phi_0S*np.exp(-mu_st*(r-R_shield_int))       #1/(m²·s)
        I_S = lambda r: E_y_J*Phi_S(r)*B                               #W/(m²)
        q_0S = B*Phi_0S*E_y_J*mu_st                                    #W/(m³)
        q_iiiS = lambda r: q_0S*np.exp(-mu_st*(r-R_shield_int))        #W/(m³)

        # ======================================
        # Radiation-induced heating in the vessel
        # ======================================
        Phi = lambda r: (Phi_S(r_S)[-1])*np.exp(-mu_st*(r-R_int))      #1/(m²·s)
        I = lambda r: E_y_J*Phi(r)*B                                   #W/(m²)
        q_0 = B*(Phi_S(r_S)[-1])*E_y_J*mu_st                           #W/(m³)
        q_iii = lambda r: q_0*np.exp(-mu_st*(r-R_int))                 #W/(m³) 
        
        # ======================================
        # T profile constants for the vessel: general and under adiabatic outer wall approximation (dT/dx = 0 at r = R_ext)
        # ======================================
        if User_D_flag == 2 or User_D_flag == 1:
            h_1 = min(h_1_int, h_1_ext)                 #Conservative: minimum h means highest thermal stresses
        elif User_D_flag == 0:
            if h_1_int - h_1_ext <= eps:
                h_1 = h_1_int                           #Below the tolerance, they can be considered equal
            
        if adiab_flag == 0:
            C1 = ((q_0/(k_st*mu_st**2))*(np.exp(-mu_st*t)-1)-(q_0/mu_st)*((1/h_1)+(np.exp(-mu_st*t)/u_2))-(T1-T_cpp))/(t+(k_st/h_1)+(k_st/u_2))
        elif adiab_flag == 1:
            C1 = -((q_0/(k_st*mu_st))*np.exp(-mu_st*t))
        C2 = T1 + (q_0/(h_1*mu_st)) + C1*(k_st/h_1) + (q_0/(k_st*mu_st**2))

        # ======================================
        # T profile constants for the thermal shield
        # ======================================
        C1_S = ((q_0S/(k_st*mu_st**2))*(np.exp(-mu_st*t_shield)-1)-(q_0S/mu_st)*((1/h_1)+(np.exp(-mu_st*t_shield)/h_1))-(T1-T1))/(t_shield+(2*k_st/h_1))
        C2_S = T1 + (q_0S/(h_1*mu_st)) + C1*(k_st/h_1) + (q_0S/(k_st*mu_st**2))

        # ======================================
        # T profiles across the vessel wall, average Ts, maxima and their positions
        # ======================================
        T_vessel = lambda r: -((q_0/(k_st*mu_st**2))*np.exp(-mu_st*(r-R_int))) + C1*(r-R_int) + C2
        T_vessel_avg = (1/t)*integrate.quad(T_vessel, R_int, R_ext)[0]
        T_vessel_max = max(T_vessel(r))
        r_T_vessel_max = r[np.argmax(T_vessel(r))]

        # ======================================
        # T profiles across the thermal shield, average Ts, maxima and their positions
        # ======================================
        T_shield = lambda r: -((q_0S/(k_st*mu_st**2))*np.exp(-mu_st*(r-R_shield_int))) + C1_S*(r-R_shield_int) + C2_S
        T_shield_avg = (1/t_shield)*integrate.quad(T_shield, R_shield_int, R_shield_ext)[0]
        T_shield_max = max(T_shield(r_S))
        r_T_shield_max = r_S[np.argmax(T_shield(r_S))]

        # ======================================
        # Thermal power fluxes (kW/m²) on the inner and outer vessel surface
        # ======================================
        DeltaT_1 = T1 - T_vessel(r[0])
        q_s1 = h_1*DeltaT_1/1000                                               #kW/m²
        q_s2 = u_2*(T_vessel(r[-1])-T_cpp)/1000                                #kW/m²

        # ======================================
        # Vessel Thermal stresses computation
        # ======================================
        f_V = lambda r: T_vessel(r)*r

        sigma_r_th_V = np.zeros(dr)
        sigma_t_th_V = np.zeros(dr)
        for i in range(len(r)):
            sigma_r_th_V[i] = (E*alpha_l/(1-nu))*(1/(r[i]**2)) * (( ((r[i]**2)-(R_int**2))/((R_ext**2)-(R_int**2)) ) * simpcomp(f_V, R_int, R_ext, dr) - simpcomp(f_V, R_int, r[i], dr))
            sigma_t_th_V[i] = (E*alpha_l/(1-nu))*(1/(r[i]**2)) * (( (((r[i]**2)+(R_int**2))/((R_ext**2)-(R_int**2)) ) * simpcomp(f_V, R_int, R_ext, dr)) + simpcomp(f_V, R_int, r[i], dr) - T_vessel(r[i])*(r[i]**2))
        sigma_t_th_V_SIMP = lambda r: (E*alpha_l/(1-nu))*(T_vessel_avg - T_vessel(r))                  #Simplified formula assuming average T
        sigma_z_th_V = sigma_r_th_V + sigma_t_th_V                                                     #Superposition principle under the hypothesis of long, hollow cylinder with load-free ends

        sigma_t_th_V_max = max(sigma_t_th_V)
        r_sigma_t_th_V_max = r[np.argmax(sigma_t_th_V)]
        sigma_t_th_V_max_SIMP = max(sigma_t_th_V_SIMP(r))
        r_sigma_t_th_V_max_SIMP = r[np.argmax(sigma_t_th_V_SIMP(r))]
        
        # ======================================
        # Thermal Shield Thermal stresses computation
        # ======================================
        f_S = lambda r: T_shield(r)*r

        sigma_r_th_S = np.zeros(dr)
        sigma_t_th_S = np.zeros(dr)
        for i in range(len(r_S)):
            sigma_r_th_S[i] = (E*alpha_l/(1-nu))*(1/(r_S[i]**2)) * (( ((r_S[i]**2)-(R_shield_int**2))/((R_shield_ext**2)-(R_shield_int**2)) ) * simpcomp(f_S, R_shield_int, R_shield_ext, dr) - simpcomp(f_S, R_shield_int, r_S[i], dr))
            sigma_t_th_S[i] = (E*alpha_l/(1-nu))*(1/(r_S[i]**2)) * (( (((r_S[i]**2)+(R_shield_int**2))/((R_shield_ext**2)-(R_shield_int**2)) ) * simpcomp(f_S, R_shield_int, R_shield_ext, dr)) + simpcomp(f_S, R_shield_int, r_S[i], dr) - T_shield(r_S[i])*(r_S[i]**2))
        sigma_t_th_S_SIMP = lambda r: (E*alpha_l/(1-nu))*(T_shield_avg - T_shield(r))                  #Simplified formula assuming average T
        sigma_z_th_S = sigma_r_th_S + sigma_t_th_S                                                     #Superposition principle under the hypothesis of long, hollow cylinder with load-free ends

        sigma_t_th_S_max = max(sigma_t_th_S)
        r_sigma_t_th_S_max = r_S[np.argmax(sigma_t_th_S)]
        sigma_t_th_S_max_SIMP = max(sigma_t_th_S_SIMP(r))
        r_sigma_t_th_S_max_SIMP = r[np.argmax(sigma_t_th_S_SIMP(r))]

        # ======================================
        # Hydrostatic Stresses and Principal Stresses in the thermal shield 
        # ======================================
        sigma_L_S = sigmaL_func(r_S, P_int_MPa, P_int_MPa, 0)
        sigma_rL_S = sigma_L_S[0]  
        sigma_tL_S = sigma_L_S[1]
        sigma_zL_S = sigma_L_S[2]
        
        sigma_r_totL_S = sigma_rL_S + sigma_r_th_S
        sigma_t_totL_S = sigma_tL_S + sigma_t_th_S
        sigma_z_totL_S = sigma_zL_S + sigma_z_th_S

        # ============================ 
        # Thermal Shield Comparison stress - Guest-Tresca Theory - Lamé + Thermal stresses
        # ============================
        sigma_cTR_LS = np.max([abs(sigma_t_totL_S - sigma_r_totL_S), abs(sigma_z_totL_S - sigma_r_totL_S), abs(sigma_t_totL_S - sigma_z_totL_S)])

        # ============================ 
        # Thermal Shield Comparison stress - Von Mises Theory - Lamé + Thermal stresses
        # ============================
        sigma_cVM_LS = max(np.sqrt(0.5*((sigma_r_totL_S - sigma_t_totL_S)**2 + (sigma_t_totL_S - sigma_z_totL_S)**2 + (sigma_z_totL_S - sigma_r_totL_S)**2))) #The max should be the worst case, in theory

        # ======================================
        # Principal stresses in the vessel
        # ======================================
        sigma_r_totM = sigma_rM_cyl + sigma_r_th_V
        sigma_t_totM = sigma_tM_cyl + sigma_t_th_V
        sigma_z_totM = sigma_zM_cyl + sigma_z_th_V
        
        sigma_r_totL = sigma_rL + sigma_r_th_V
        sigma_t_totL = sigma_tL + sigma_t_th_V
        sigma_z_totL = sigma_zL + sigma_z_th_V

        # ======================================
        # Maximum Hoop Thermal Stress in the vessel via design curves
        # ======================================
        for i in range(len(indexes)):
            if mu_st > mu_values[i] and mu_st < mu_values[i+1]:
                mu_L = mu_values[i]
                mu_R = mu_values[i+1]
                #print("Current mu values: ", mu_values[i], mu_st, mu_values[i+1])
                current_L_key, current_R_key = keys_list[i], keys_list[i+1]
                x_points_L, x_points_R = mu_curves[current_L_key][2][:,0], mu_curves[current_R_key][2][:,0]
                y_points_L, y_points_R = mu_curves[current_L_key][2][:,1], mu_curves[current_R_key][2][:,1]

                p_L = np.polyfit(x_points_L, y_points_L, deg = 3) #len(y_points_L)-1
                p_R = np.polyfit(x_points_R, y_points_R, deg = 3)

                L_Interpolator = lambda x: np.polyval(p_L, x)
                R_Interpolator = lambda x: np.polyval(p_R, x)

                sigmaT_L_V = L_Interpolator(R_ext/R_int)                                                                                      #Interpolated sigmaT coefficient on the left ISO-mu 
                sigmaT_R_V = R_Interpolator(R_ext/R_int)                                                                                      #Interpolated sigmaT coefficient on the right ISO-mu
                sigmaT_L_S = L_Interpolator(R_shield_ext/R_shield_int)
                sigmaT_R_S = R_Interpolator(R_shield_ext/R_shield_int)

                sigmaT_eq_V = lambda x: sigmaT_L_V + ((sigmaT_R_V-sigmaT_L_V)/(mu_R-mu_L))*(x - mu_L)
                sigmaT_eq_S = lambda x: sigmaT_L_S + ((sigmaT_R_S-sigmaT_L_S)/(mu_R-mu_L))*(x - mu_L)
                sigmaT_V = sigmaT_eq_V(mu_st)                                                                                         #Double-interpolated (linear) sigmaT coefficient
                sigmaT_S = sigmaT_eq_S(mu_st)
        
        sigma_t_th_V_max_DES = sigmaT_V*(alpha_l*E*q_0)/(k_st*(1-nu)*(mu_st**2))
        sigma_t_th_S_max_DES = sigmaT_S*(alpha_l*E*q_0S)/(k_st*(1-nu)*(mu_st**2))
        
        # ============================ 
        # Vessel Comparison stress - Guest-Tresca Theory - Mariotte/Lamé + Thermal stresses
        # ============================
        sigma_cTR_M = np.max([abs(sigma_t_totM - sigma_r_totM), abs(sigma_z_totM - sigma_r_totM), abs(sigma_t_totM - sigma_z_totM)])
        sigma_cTR_L = np.max([abs(sigma_t_totL - sigma_r_totL), abs(sigma_z_totL - sigma_r_totL), abs(sigma_t_totL - sigma_z_totL)])

        # ============================ 
        # Vessel Comparison stress - Von Mises Theory - Mariotte/Lamé + Thermal stresses
        # ============================
        sigma_cVM_M = max(np.sqrt(0.5*((sigma_r_totM - sigma_t_totM)**2 + (sigma_t_totM - sigma_z_totM)**2 + (sigma_z_totM - sigma_r_totM)**2)))
        sigma_cVM_L = max(np.sqrt(0.5*((sigma_r_totL - sigma_t_totL)**2 + (sigma_t_totL - sigma_z_totL)**2 + (sigma_z_totL - sigma_r_totL)**2))) #The max should be the worst case, in theory
        
        # ============================ 
        # Yield Stress and Stress Intensity Data Interpolation
        # ============================
        T_des_vessel = T_vessel_avg                                                     #K  -   Check in the HARVEY/Thermomechanics Chapter how to choose the design T
        T_des_vessel_C = T_des_vessel - 273.15                                          #°C
        T_des_shield = T_shield_avg                                                     #K
        T_des_shield_C = T_des_shield - 273.15                                          #°C

        Yield_stress = Yield_CubicSpline(T_des_vessel_C)
        Stress_Intensity = Intensity_CubicSpline(T_des_vessel_C)
        sigma_allowable = 1.5 * Stress_Intensity  #MPa
    
        Yield_stress_S = Yield_CubicSpline(T_des_shield_C)
        Stress_Intensity_S = Intensity_CubicSpline(T_des_shield_C)
        sigma_allowable_S = 1.5 * Stress_Intensity_S  #MPa

        # ======================================
        # Thermal Shield Thermomechanical Integrity Verification
        # ======================================
        if max(abs(sigma_r_totL_S)) > 3*Stress_Intensity_S or max(abs(sigma_t_totL_S)) > 3*Stress_Intensity_S or max(abs(sigma_z_totL_S)) > 3*Stress_Intensity_S:
            flag_primsec = 1
        else:
            flag_primsec = 0

        if max(abs(sigma_rL_S)) > Stress_Intensity_S or max(abs(sigma_tL_S)) > Stress_Intensity_S or sigma_zL_S > Stress_Intensity_S:
            flag_prim = 1
        else:
            flag_prim = 0
        
        if flag_primsec == 1 or flag_prim == 1:
            print("\nThe current stress state in the thermal shield is not acceptable. \nPrimary + Secondary Stresses flag: %d \nPrimary Stresses flag: %d" %(flag_primsec, flag_prim))
            print("Absolute value of the maximum radial thermal stress: %.3f MPa\nAbsolute value of the maximum hoop thermal stress: %.3f MPa\nAbsolute value of the maximum axial thermal stress: %.3f MPa" %(abs(max(sigma_r_th_S)),abs(max(sigma_t_th_S)),abs(max(sigma_z_th_S))))
            continue
        elif flag_primsec == 0 and flag_prim == 0:
            Corradi_flag = 1                                                #Only enters the Corradi procedure if the thermal shield is ok

        # ============================ 
        # Corradi Design Procedure
        # ============================
        q_E_fun = lambda Dt: 2 * (E/(1-(nu**2))) * (1/(Dt*((Dt-1)**2)))     #Elastic Instability Limit for Thick Tubes
        q_0_fun = lambda Dt: 2 * Yield_stress * 1/Dt * (1+(1/(2*Dt)))       #Plastic Collapse Limit for Thick Tubes
        Dt_Crit_Ratio = np.sqrt(E/(Yield_stress*(1-(nu**2))))
        Current_Slenderness = (D_vess_int+2*t)/t

        if Corradi_flag == 1:
            def Corradi(Slenderness):
                if isinstance(Slenderness, np.ndarray):
                    mu = np.zeros(len(Slenderness))
                    Z = lambda Dt: (np.sqrt(3)/4) * (2*Dt + 1) * W                  #Accounts for ovality
                    q_U = lambda Dt: q_0_fun(Dt)/np.sqrt(1+(Z(Dt)**2))
                    q_L = lambda Dt: (1/2) * (q_0_fun(Dt) + q_E_fun(Dt)*(1 + Z(Dt)) - np.sqrt(((q_0_fun(Dt) + q_E_fun(Dt)*(1 + Z(Dt)))**2)-(4 * q_0_fun(Dt) * q_E_fun(Dt))))
                    
                    for i in range(len(mu)):
                        if q_0_fun(Slenderness[i])/q_E_fun(Slenderness[i]) < 0.04:
                            mu[i] = 1
                        elif 0.04 <= q_0_fun(Slenderness[i])/q_E_fun(Slenderness[i]) <= 0.7:
                            mu[i] = (0.35 * np.log(q_E_fun(Slenderness[i])/q_0_fun(Slenderness[i]))) - 0.125
                        elif q_0_fun(Slenderness[i])/q_E_fun(Slenderness[i]) > 0.7:
                            mu[i] = 0
                        
                    q_C = mu*q_U(Slenderness) + (1-mu)*q_L(Slenderness)
                    q_a = q_C/s
                else:
                    raise TypeError("The 1st input must be a numpy array.")
                if len(q_C) == 1:
                    q_C = q_C.item()
                if len(q_a) == 1:
                    q_a = q_a.item()
                if len(mu) == 1:
                    mu = mu.item()
                return (q_C, q_a, s, mu)
            
            # ============================ 
            # Corradi Design Procedure Results
            # ============================
            Corradi_vessel = Corradi(np.array([Current_Slenderness]))
            if (P_cpp < 10*Corradi_vessel[1] and sigma_cTR_L < sigma_allowable):        #sigma_cTR_M < sigma_allowable has been removed because it never happens due to the conservativeness of the Mariotte formula
                final_flag = 1
            else:
                final_flag = 0
    
    # ======================================
    # Plotting the volumetric heat source profiles 
    # ======================================
    while True:
        try:
            hs_flag = int(input("\nDo you want to visualize the volumetric heat source q0 inside the vessel's wall and in the thermal shield? (1: Yes, 0: No): "))
            if hs_flag not in (0, 1):
                raise RuntimeError("Invalid input! Please enter either 0 or 1.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)

    if hs_flag == 1:
        # ======================================
        # Thermal Shield
        # ======================================
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)
        if R_shield_ext - R_shield_int > 0.1:
            plt.xlim(D_barr_ext/2, R_int)
            plt.axvline(x = D_barr_ext/2, color='black', linewidth='3', label='Barrel Outer Surface')
            plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
        else:
            plt.xlim(R_shield_int - 0.05, R_shield_ext + 0.05)
        plt.axvline(x = R_shield_int, color='black', linewidth='3', label='Thermal Shield Inner Surface')
        plt.axvline(x = R_shield_ext, color='black', linewidth='3', label='Thermal Shield Outer Surface')
        plt.plot(r_S, q_iiiS(r_S), 'g', label='Radial (r) Volumetric heat source profile')
        plt.plot(r_S[0], q_iiiS(r_S[0]), 'or', label='Thermal Shield Inner Surface Value')
        plt.plot(r_S[-1], q_iiiS(r_S[-1]), 'or', label='Thermal Shield Outer Surface Value')
        plt.axhline(y = 0, color='black', linewidth='1', label='y=0')
        plt.xlabel('Radius (m)')
        plt.ylabel(r'$q_0$ (W/m$^3$)')
        plt.title('Volumetric heat source profile across the thermal shield')
        plt.legend()
        plt.grid()

        # ======================================
        # Vessel
        # ======================================
        plt.subplot(1,2,2)
        plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Surface')
        plt.axvline(x = R_ext, color='black', linewidth='3')
        plt.plot(r, q_iii(r), 'g', label='Radial (r) Volumetric heat source profile')
        plt.plot(r[0], q_iii(r[0]), 'or', label='Vessel Inner Surface Value')
        plt.plot(r[-1], q_iii(r[-1]), 'or', label='Vessel-Insulation Interface Value')
        plt.axhline(y = 0, color='black', linewidth='1', label='y=0')
        plt.xlabel('Radius (m)')
        plt.ylabel(r'$q_0$ (W/m$^3$)')
        plt.title('Volumetric heat source profile across the vessel wall')
        plt.legend()
        plt.grid()
        plt.show()

    # ======================================
    # Plotting the T profiles
    # ======================================
    while True:
        try:
            T_pl_flag = int(input("\nDo you want to visualize the T profile across the vessel's wall and the thermal shield? (1: Yes, 0: No): "))
            if T_pl_flag not in (0, 1):
                raise RuntimeError("Invalid input! Please enter either 0 or 1.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)
    
    if adiab_flag == 0:
        if (T_vessel_max - 273.15) > T_creep:
            print("\nWARNING: The maximum vessel temperature T = %.3f °C exceeds the creep threshold temperature of %d °C!" %(T_vessel_max - 273.15, T_creep))
            creep_flag_V = 1
        if (T_shield_max - 273.15) > T_creep:
            print("\nWARNING: The maximum thermal shield temperature T = %.3f °C exceeds the creep threshold temperature of %d °C!" %(T_shield_max - 273.15, T_creep))
            creep_flag_S = 1
        if T_pl_flag == 1:
            # ======================================
            # Thermal Shield T Profile
            # ======================================
            plt.figure(figsize=(15,15))
            plt.subplot(1,2,1)
            if R_shield_ext - R_shield_int > 0.1:
                plt.xlim(D_barr_ext/2, R_int)
                plt.axvline(x = D_barr_ext/2, color='black', linewidth='3', label='Barrel Outer Surface')
                plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
            else:
                plt.xlim(R_shield_int - 0.05, R_shield_ext + 0.05)
            plt.axvline(x = R_shield_int, color='black', linewidth='3', label='Thermal Shield Inner Surface')
            plt.axvline(x = R_shield_ext, color='black', linewidth='3', label='Thermal Shield Outer Surface')
            plt.plot(r_S, T_shield(r_S) - 273.15, label='Radial (r) T Profile')
            plt.plot(r_T_shield_max, T_shield_max - 273.15,'or',label='Max T')
            plt.axhline(y = T_shield_avg - 273.15, color='green', label='Thermal Shield Average T')
            plt.xlabel('Radius (m)')
            plt.ylabel('T (°C)')
            plt.title('Thermal Shield Temperature Profile, Average and Maximum ')
            plt.legend()
            plt.grid()

            # ======================================
            # Vessel T Profile
            # ======================================
            plt.subplot(1,2,2)
            plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
            plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
            plt.plot(r, T_vessel(r) - 273.15, label='Radial (r) T Profile')
            plt.plot(r_T_vessel_max, T_vessel_max - 273.15,'or',label='Max T')
            plt.axhline(y = T_vessel_avg - 273.15, color='green', label='Vessel Wall Average T')
            plt.xlabel('Radius (m)')
            plt.ylabel('T (°C)')
            plt.title('Vessel Wall Temperature Profile, Average and Maximum ')
            plt.legend()
            plt.grid()
            plt.show()
     
    elif adiab_flag == 1:
        if (T_vessel_max - 273.15) > T_creep:
            print("\nWARNING: The maximum vessel temperature T = %.3f °C exceeds the creep threshold temperature of %d °C!" %(T_vessel_max - 273.15, T_creep))
            creep_flag_V = 1
        if (T_shield_max - 273.15) > T_creep:
            print("\nWARNING: The maximum thermal shield temperature T = %.3f °C exceeds the creep threshold temperature of %d °C!" %(T_shield_max - 273.15, T_creep))
            creep_flag_S = 1
        if T_pl_flag == 1:
            # ======================================
            # Thermal Shield T Profile
            # ======================================
            plt.figure(figsize=(15,15))
            plt.subplot(1,2,1)
            if R_shield_ext - R_shield_int > 0.1:
                plt.xlim(D_barr_ext/2, R_int)
                plt.axvline(x = D_barr_ext/2, color='black', linewidth='3', label='Barrel Outer Surface')
                plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
            else:
                plt.xlim(R_shield_int - 0.05, R_shield_ext + 0.05)
            plt.axvline(x = R_shield_int, color='black', linewidth='3', label='Thermal Shield Inner Surface')
            plt.axvline(x = R_shield_ext, color='black', linewidth='3', label='Thermal Shield Outer Surface')
            plt.plot(r_S, T_shield(r_S) - 273.15, label='Radial (r) T Profile')
            plt.plot(r_T_shield_max, T_shield_max - 273.15,'or',label='Max T')
            plt.axhline(y = T_shield_avg - 273.15, color='green', label='Thermal Shield Average T')
            plt.xlabel('Radius (m)')
            plt.ylabel('T (°C)')
            plt.title('Thermal Shield Temperature Profile, Average and Maximum ')
            plt.legend()
            plt.grid()
            
            # ======================================
            # Vessel Under Adiabatic Outer Wall Approximation
            # ======================================
            plt.subplot(1,2,2)
            plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
            plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
            plt.plot(r, T_vessel(r) - 273.15, label='Radial (r) T Profile')
            plt.plot(r_T_vessel_max, T_vessel_max - 273.15,'or', label='Max T')
            plt.axhline(y = T_vessel_avg - 273.15, color='green', label='Wall Average T')
            plt.xlabel('Radius (m)')
            plt.ylabel('T (°C)')
            plt.title('Wall Temperature Profile, Average and Maximum under AOW Approximation ')
            plt.legend()
            plt.grid()
            plt.show()

    # ======================================
    # Plotting the thermal stress profiles
    # ======================================
    while True:
        try:
            sigma_th_pl_flag = int(input("\nDo you want to visualize a plot of the thermal stress profiles in the vessel and in the thermal shield? (1: Yes, 0: No): "))
            if sigma_th_pl_flag not in (0, 1):
                raise RuntimeError("Invalid input! Please enter either 0 or 1.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)

    if sigma_th_pl_flag == 1:

        # ======================================
        # Thermal shield thermal stress profiles
        # ======================================
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)
        if R_shield_ext - R_shield_int > 0.1:
            plt.xlim(D_barr_ext/2, R_int)
            plt.axvline(x = D_barr_ext/2, color='black', linewidth='3', label='Barrel Outer Surface')
            plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
        else:
            plt.xlim(R_shield_int - 0.05, R_shield_ext + 0.05)
        plt.axvline(x = R_shield_int, color='black', linewidth='3', label='Thermal Shield Inner Surface')
        plt.axvline(x = R_shield_ext, color='black', linewidth='3', label='Thermal Shield Outer Surface')
        plt.plot(r_S, sigma_r_th_S, linewidth='0.75', label='Radial (r) Thermal Stress Profile')
        plt.plot(r_S, sigma_t_th_S, linewidth='0.75', label='Hoop (θ) Thermal Stress Profile')
        plt.plot(r_S, sigma_z_th_S, color='green', linewidth='0.5', label='Axial (z) Thermal Stress Profile')
        plt.axhline(y = 0, color='black', linewidth='1', label='y=0')
        plt.plot(r_sigma_t_th_S_max, sigma_t_th_S_max,'or', label='Max Hoop Stress')
        plt.xlabel('Radius (m)')
        plt.ylabel('Thermal Stress (MPa)')
        plt.title('Thermal Shield Thermal Stress Profiles and Maximum Hoop Stress')
        plt.legend()
        plt.grid()
        
        # ======================================
        # Vessel thermal stress profiles
        # ======================================
        plt.subplot(1,2,2)
        plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
        plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
        plt.plot(r, sigma_r_th_V, linewidth='0.75', label='Radial (r) Thermal Stress Profile')
        plt.plot(r, sigma_t_th_V, linewidth='0.75', label='Hoop (θ) Thermal Stress Profile')
        plt.plot(r, sigma_z_th_V, color='green', linewidth='0.5', label='Axial (z) Thermal Stress Profile')
        plt.axhline(y = 0, color='black', linewidth='1', label='y=0')
        plt.plot(r_sigma_t_th_V_max, sigma_t_th_V_max,'or', label='Max Hoop Stress')
        plt.xlabel('Radius (m)')
        plt.ylabel('Thermal Stress (MPa)')
        plt.title('Vessel Wall Thermal Stress Profiles and Maximum Hoop Stress')
        plt.legend()
        plt.grid()
        plt.show()

    # ======================================
    # Plotting the maximum thermal stress via the design curves
    # ======================================
    while True:
        try:
            des_pl_flag = int(input("\nDo you want to visualize a plot of the design curves and the maximum thermal stress in the vessel and in the thermal shield? (1: Yes, 0: No): "))
            if des_pl_flag not in (0, 1):
                raise RuntimeError("Invalid input! Please enter either 0 or 1.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)

    if des_pl_flag == 1:
        plt.figure(figsize=(10,10))
        plt.plot(ba_ratio_plot, L_Interpolator(ba_ratio_plot), 'k', label=f'Iso-mu = {mu_L} 1/m')
        plt.plot(ba_ratio_plot, R_Interpolator(ba_ratio_plot), 'k', label=f'Iso-mu = {mu_R} 1/m')
        plt.plot(R_ext/R_int, sigmaT_V,'or', label=r'Current $\sigma$$_T$ in the vessel')
        plt.plot(R_shield_ext/R_shield_int, sigmaT_S,'ob', label=r'Current $\sigma$$_T$ in the thermal shield')
        plt.xlabel('R$_{ext}$/R$_{int}$')
        plt.ylabel(r'$\sigma$$_T$')
        plt.title('Design curves')
        plt.legend()
        plt.grid()
        plt.show()

    # ======================================
    # Plotting the yield stress and stress intensity curves
    # ======================================
    while True:
        try:
            Interp_pl_flag = int(input("\nDo you want to visualize a plot of the Yield Stress and Stress Intensity as given by ASME for both the vessel and the thermal shield? (1: Yes, 0: No): "))
            if Interp_pl_flag not in (0, 1):
                raise RuntimeError("Invalid input! Please enter either 0 or 1.")
            break  
        except ValueError:
            print("Please enter a valid integer.")
        except RuntimeError as e:
            print(e)
    
    if max(T_thr) > T_des_vessel_C:
        Tplot = np.linspace(min(T_thr), max(T_thr), 1000)
    else:
        Tplot = np.linspace(min(T_thr), T_des_vessel_C, 1000)
    
    if Interp_pl_flag == 1:
        
        # ============================ 
        # Yield Stress
        # ============================
        plt.figure(figsize = (12,10))
        plt.subplot(1,2,1)
        plt.plot(T_thr, sigma_y, 'sk', label = 'Yield Stress Data')
        plt.plot(Tplot, Yield_Interpolator(Tplot), '--', color = 'orange', label = 'Yield Stress n-1 Interpolation')
        plt.plot(Tplot, Yield_CubicSpline(Tplot), 'green', label = 'Yield Stress Cubic Spline Interpolation')
        plt.plot(T_des_vessel_C, Yield_stress, '--or', label = r'Current Vessel Yield Stress $\sigma$$_y$')
        plt.plot(T_des_shield_C, Yield_stress_S, '--ob', label = r'Current Thermal Shield Yield Stress $\sigma$$_y$')
        plt.xlabel("Temperature (°C)")
        plt.ylabel(r"Yield Stress $\sigma$$_y$")
        plt.title("Yield Stress Data and Interpolation VS Temperature", fontsize = 10)
        plt.legend()
        plt.grid()
        plt.tight_layout()

        # ============================ 
        # Stress intensity
        # ============================
        plt.subplot(1,2,2)
        plt.plot(T_thr, sigma_in, 'sk', label = 'Stress Intensity Data')
        plt.plot(Tplot, Intensity_Interpolator(Tplot), '--', color = 'orange', label = 'Stress Intensity n-1 Interpolation')
        plt.plot(Tplot, Intensity_CubicSpline(Tplot), 'green', label = 'Stress Intensity Cubic Spline Interpolation')
        plt.plot(T_des_vessel_C, Stress_Intensity, '--or', label = r'Current Vessel Stress Intensity $\sigma$$_m$')
        plt.plot(T_des_shield_C, Stress_Intensity_S, '--ob', label = r'Current Thermal Shield Stress Intensity $\sigma$$_m$')
        plt.xlabel("Temperature (°C)")
        plt.ylabel(r"Stress Intensity $\sigma$$_m$")
        plt.title("Stress Intensity Data and Interpolation VS Temperature", fontsize = 10)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()
        
    # ============================ 
    # Sizing of a thick cylinder under external pressure
    # ============================
    if R_int/t > 5:
        while True:
            try:
                ThinTubes_flag = int(input("\nWith a thickness value of %.3f m, the vessel can be considered thin. Are you interested in the thin tube limits for Elastic Instability and Plastic Collapse? (1: Yes, 0: No): " %t))
                if ThinTubes_flag not in (0, 1):
                    raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                break  
            except ValueError:
                print("Please enter a valid integer.")
            except RuntimeError as e:
                print(e)

        if ThinTubes_flag == 1:
            p_E_fun = lambda Dt: 2 * (E/(1-(nu**2))) * (1/(Dt**3))              #Elastic Instability Limit for Thin Tubes
            p_0_fun = lambda Dt: 2 * Yield_stress * 1/Dt                        #Plastic Collapse Limit for Thin Tubes  -   Vessel
            p_0_fun_S = lambda Dt: 2 * Yield_stress_S * 1/Dt                    #Plastic Collapse Limit for Thin Tubes  -   Thermal Shield

        elif ThinTubes_flag == 0:
            print("Skipping thin tube limits.")
    else:
        print("\nThe cylinder can't be considered thin. Skipping thin tube limits.")
        ThinTubes_flag = 0

    if ThinTubes_flag == 1:
        while True:
            try:
                Corradi_flag = int(input("\nThe thin tube limits were adopted. Are you interested in the more general Corradi Design Procedure? (1: Yes, 0: No): "))
                if Corradi_flag not in (0, 1):
                    raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                break  
            except ValueError:
                print("Please enter a valid integer.")
            except RuntimeError as e:
                print(e)
        if Corradi_flag == 0:
            # ============================ 
            # Elastic instability and plastic collapse curves
            # ============================
            while True:
                try:
                    Collapse_pl_flag = int(input("\nDo you want to visualize the buckling and plastic collapse curves for thin and thick tubes? (1: Yes, 0: No): "))
                    if Collapse_pl_flag not in (0, 1):
                        raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                    break  
                except ValueError:
                    print("Please enter a valid integer.")
                except RuntimeError as e:
                    print(e)
            
            if Collapse_pl_flag == 1:
                # ============================ 
                # Plastic collapse and buckling plots
                # ============================
                plt.figure(figsize = (8, 8))
                plt.semilogy(Dt_ratio_plot, p_E_fun(Dt_ratio_plot), 'blue', label='p$_E$')
                plt.semilogy(Dt_ratio_plot, q_E_fun(Dt_ratio_plot), '--b', label='q$_E$')
                plt.semilogy(Dt_ratio_plot, p_0_fun(Dt_ratio_plot), 'red', label='p$_0$')
                plt.semilogy(Dt_ratio_plot, q_0_fun(Dt_ratio_plot), '--r', label='q$_0$')
                plt.axvline(x = Dt_Crit_Ratio, color = 'black', linewidth = '3', label = 'Critical Slenderness')
                plt.axvline(x = Current_Slenderness, color = 'green', linewidth = '3', label = 'Current Vessel Slenderness')
                plt.xlabel("Geometrical Slenderness D/t")
                plt.ylabel("Theoretical Limit Values (MPa)")
                plt.title("Plastic Collapse and Buckling Curves")
                plt.legend()
                plt.grid()
                plt.show()

        elif Corradi_flag == 1:
            while True:
                try:
                    Collapse_pl_flag = int(input("\nDo you want to visualize the buckling and plastic collapse curves for thin and thick tubes and the Corradi curve? (1: Yes, 0: No): "))
                    if Collapse_pl_flag not in (0, 1):
                        raise RuntimeError("Invalid input! Please enter either 0 or 1.")
                    break  
                except ValueError:
                    print("Please enter a valid integer.")
                except RuntimeError as e:
                    print(e)
            
            if Collapse_pl_flag == 1:
                # ============================ 
                # Plastic collapse and buckling plots
                # ============================
                plt.figure(figsize = (8, 8))
                plt.subplot(1,2,1)
                plt.semilogy(Dt_ratio_plot, p_E_fun(Dt_ratio_plot), 'blue', label='p$_E$')
                plt.semilogy(Dt_ratio_plot, q_E_fun(Dt_ratio_plot), '--b', label='q$_E$')
                plt.semilogy(Dt_ratio_plot, p_0_fun(Dt_ratio_plot), 'red', label='p$_0$')
                plt.semilogy(Dt_ratio_plot, q_0_fun(Dt_ratio_plot), '--r', label='q$_0$')
                plt.semilogy(Dt_ratio_plot, Corradi(Dt_ratio_plot)[0], 'orange', label='Corradi q$_c$')
                plt.axvline(x = Dt_Crit_Ratio, color = 'black', linewidth = '3', label = 'Critical Slenderness')
                plt.axvline(x = Current_Slenderness, color = 'green', linewidth = '3', label = 'Current Vessel Slenderness')
                plt.xlabel("Geometrical Slenderness D/t")
                plt.ylabel("Theoretical Limit Values (MPa)")
                plt.title("Plastic Collapse and Buckling Curves")
                plt.legend()
                plt.grid()
                plt.tight_layout()

                plt.subplot(1,2,2)
                plt.plot(Dt_ratio_plot, Corradi(Dt_ratio_plot)[3], 'k', label=r'Corradi $\mu$')
                plt.xlabel("Geometrical Slenderness D/t")
                plt.ylabel(r"Corradi $\mu$")
                plt.title(r"$\mu$ coefficient - Corradi Procedure")
                plt.legend()
                plt.grid()
                plt.tight_layout()
                plt.show()
        
    elif ThinTubes_flag == 0:
        print("Adopting Corradi Design Procedure.")
        Corradi_flag = 1

    # ============================ 
    # Final Results Printing
    # ============================
    print("\n\n\n###################################################### Final  Results ######################################################")
        
    print("\nThe vessel thickness has been increased %d times by 1cm. Computed vessel thickness: %.3f m" %(counter_vessel, t))
    print("Computed thermal shield thickness: %.3f m" %t_shield)
    
    # ============================ 
    # Heat Transfer Results
    # ============================
    print("\n################################################## Heat transfer results ###################################################")
    print("\nVolumetric heat source at the vessel inner surface: %.3f W/m³" %q_iii(r[0]))
    print("Volumetric heat source at the vessel-insulation interface: %.3f W/m³" %q_iii(r[-1]))

    print("\nHeat transfer coefficient h1 = %.3f W/(m²·K)" %h_1)
    print("Heat transfer coefficient h2 = %.3f W/(m²·K)" %h_2)
    print("Overall heat transfer coefficient outside the vessel u2 = %.3f W/(m²·K)" %u_2)
    
    print("\nThermal power flux on the inner vessel surface: %.3f kW/m²" %q_s1)
    print("Thermal power flux on the outer vessel surface: %.3f kW/m²" %q_s2)
    
    # ============================ 
    # Temperature Results
    # ============================
    print("\n####################################################### Temperatures #######################################################")
    if adiab_flag == 0:
        print("\nAverage Vessel Temperature (numerical integration): %.3f °C" %(T_vessel_avg - 273.15))
        print("Maximum Vessel Temperature: %.3f °C at r = %.3f m" %(T_vessel_max - 273.15, r_T_vessel_max))
        print("Vessel Temperature at the inner surface: %-3f °C at r = %.3f m" %(T_vessel(r)[0] - 273.15, r[0]))
        print("Vessel Temperature at the outer surface: %-3f °C at r = %.3f m" %(T_vessel(r)[-1] - 273.15, r[-1]))

        print("\nAverage Thermal Shield Temperature (numerical integration): %.3f °C" %(T_shield_avg - 273.15))
        print("Maximum Thermal Shield Temperature: %.3f °C at r = %.3f m" %(T_shield_max - 273.15, r_T_vessel_max))
        print("Thermal Shield Temperature at the inner surface: %-3f °C at r = %.3f m" %(T_shield(r_S)[0] - 273.15, r_S[0]))
        print("Thermal Shield Temperature at the outer surface: %-3f °C at r = %.3f m" %(T_shield(r_S)[-1] - 273.15, r_S[-1]))
        
    elif adiab_flag == 1:
        print("\nAverage Vessel Temperature under Adiabatic Outer Wall approximation (numerical integration): %.3f °C" %(T_vessel_avg - 273.15))
        print("Maximum Vessel Temperature under Adiabatic Outer Wall approximation: %.3f °C at r = %.3f m" %(T_vessel_max - 273.15, r_T_vessel_max))
        print("Vessel Temperature at the inner surface under Adiabatic Outer Wall approximation: %-3f °C at r = %.3f m" %(T_vessel(r)[0] - 273.15, r[0]))
        print("Vessel Temperature at the outer surface under Adiabatic Outer Wall approximation: %-3f °C at r = %.3f m" %(T_vessel(r)[-1] - 273.15, r[-1]))
            
        print("\nAverage Thermal Shield Temperature (numerical integration): %.3f °C" %(T_shield_avg - 273.15))
        print("Maximum Thermal Shield Temperature: %.3f °C at r = %.3f m" %(T_shield_max - 273.15, r_T_vessel_max))
        print("Thermal Shield Temperature at the inner surface: %-3f °C at r = %.3f m" %(T_shield(r_S)[0] - 273.15, r_S[0]))
        print("Thermal Shield Temperature at the outer surface: %-3f °C at r = %.3f m" %(T_shield(r_S)[-1] - 273.15, r_S[-1]))
    
    # ============================ 
    # Stress Results
    # ============================
    print("\n######################################################### Stresses #########################################################")
    print("\nMaximum Thermal Hoop Stress in the vessel: %.3f Mpa at r = %.3f m" %(sigma_t_th_V_max, r_sigma_t_th_V_max))
    #print("Maximum Thermal Hoop Stress in the vessel (Simplified formula): %.3f Mpa at r = %.3f m" %(sigma_t_th_V_max_SIMP, r_sigma_t_th_V_max_SIMP))
    print("Maximum thermal hoop stress in the vessel via design curves: %.3f MPa" %sigma_t_th_V_max_DES)

    print("Maximum Thermal Hoop Stress in the thermal shield: %.3f Mpa at r = %.3f m" %(sigma_t_th_S_max, r_sigma_t_th_S_max))
    #print("Maximum Thermal Hoop Stress in the thermal shield (Simplified formula): %.3f Mpa at r = %.3f m" %(sigma_t_th_S_max_SIMP, r_sigma_t_th_S_max_SIMP))
    print("Maximum thermal hoop stress in the thermal shield via design curves: %.3f MPa" %sigma_t_th_S_max_DES)

    print("\nGuest-Tresca Equivalent Stress in the vessel - Mariotte solution: %.3f Mpa" %sigma_cTR_M)
    print("Guest-Tresca Equivalent Stress in the vessel - Lamé solution: %.3f Mpa" %sigma_cTR_L)

    print("\nFor a design vessel temperature of %.3f °C: " %T_des_vessel_C)
    print('Yield Stress: Sy'," = %.3f MPa" %Yield_stress)
    print('Stress Intensity: Sm'," = %.3f MPa" %Stress_Intensity)
    print("Allowable Stress: %.3f MPa" %sigma_allowable)

    print("\nFor a design thermal shield temperature of %.3f °C: " %T_des_shield_C)
    print('Yield Stress: Sy'," = %.3f MPa" %Yield_stress_S)
    print('Stress Intensity: Sm'," = %.3f MPa" %Stress_Intensity_S)
    print("Allowable Stress: %.3f MPa" %sigma_allowable_S)

    # ============================ 
    # Thermal Shield
    # ============================
    print("\n###################################################### Thermal Shield ######################################################")
    if flag_primsec == 1 or flag_prim == 1:
        print("\nThe current stress state in the thermal shield is not acceptable. \nPrimary + Secondary Stresses flag: %d \nPrimary Stresses flag: %d" %(flag_primsec, flag_prim))
        print("Maximum absolute value of the radial thermal stress: %.3f MPa\nMaximum absolute value of the hoop thermal stress: %.3f MPa\nMaximum absolute value of the axial thermal stress: %.3f MPa" %(max(abs(sigma_r_th_S)),max(abs(sigma_t_th_S)),max(abs(sigma_z_th_S))))
    
    elif flag_primsec == 0 and flag_prim == 0:
        print("\nThe current stress state in the thermal shield is acceptable:")
        print("\nMaximum absolute value of the total radial stress: %.3f MPa\nMaximum absolute value of the total hoop stress: %.3f MPa\nMaximum absolute value of the total axial stress: %.3f MPa" %(max(abs(sigma_r_totL_S)),max(abs(sigma_t_totL_S)),max(abs(sigma_z_totL_S))))
        print("\nAll are lower than 3Sm = %.3f MPa" %(3*Stress_Intensity_S))         
        print("\nMaximum value of the primary radial stress: %.3f MPa\nMaximum value of the primary hoop stress: %.3f MPa\nPrimary axial stress: %.3f MPa" %(max(sigma_rL_S),max(sigma_tL_S),sigma_zL_S))
        print("\nAll are lower than Sm = %.3f MPa" %Stress_Intensity_S)

    if (sigma_cTR_LS < sigma_allowable_S):
        print("\nThe comparison stress according to Tresca-Lamé Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_LS, sigma_allowable_S))
    
    if creep_flag_S == 1:
        print("\nCreep might occur in the thermal shield due to high temperatures. Either an additional thermal shield, a reduced thickness or both are required.")
    elif creep_flag_S == 0:
        print("There is no risk of thermal creep occurring in the thermal shield.")
        print("The thermal shield's integrity is ensured.")
        
    # ============================ 
    # Vessel
    # ============================
    print("\n########################################################## Vessel ##########################################################")
    Corradi_vessel = Corradi(np.array([Current_Slenderness]))
    print("\nAccording to the Corradi Design Procedure:")
    print("Current slenderness: %.3f    -   Critical slenderness: %.3f" %(Current_Slenderness, Dt_Crit_Ratio))
    print("\nThe theoretical limit for collapse pressure, accounting for ovality, is: q_c = %.3f MPa = %.3f bar" %(Corradi_vessel[0], 10*Corradi_vessel[0]))
    print("A safety factor s = %.3f was assumed. \nThe allowable external pressure is thus: q_a = %.3f MPa = %.3f bar" %(Corradi_vessel[2], Corradi_vessel[1], 10*Corradi_vessel[1]))
    print("\n############################################################################################################################")

    if (P_cpp < 10*Corradi_vessel[1] and sigma_cTR_L < sigma_allowable):       #sigma_cTR_M < sigma_allowable has been removed to avoid overly conservative results
        print("\nThe given external pressure of %.3f bar is lower than the allowable pressure of %.3f bar" %(P_cpp, 10*Corradi_vessel[1]))
        print("The comparison stress according to Tresca-Lamé Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_L, sigma_allowable))
        if creep_flag_V == 1:
            print("Creep might occur in the vessel due to high temperatures. Either an additional thermal shield, a reduced thickness or both are required.")
        elif creep_flag_V == 0:
            print("There is no risk of thermal creep occurring in the vessel.")
            print("The vessel's integrity is ensured.")
            print("\n############################################################################################################################")
            print("\nThe design is correct!")
        print("\n############################################################################################################################")
        
    elif (P_cpp > 10*Corradi_vessel[1]):
        print("\nThe given external pressure of %.3f bar is higher than the allowable pressure of %.3f bar: a change in thickness is required!" %(P_cpp, 10*Corradi_vessel[1]))
        print("\n############################################################################################################################")

    elif (sigma_cTR_L > sigma_allowable):                                       #sigma_cTR_M > sigma_allowable has been removed
        print("\nThe Tresca-Lamè comparison stress Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_L,sigma_allowable))
        print("\n############################################################################################################################")