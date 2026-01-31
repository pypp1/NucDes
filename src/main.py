import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as lng 
from scipy import integrate, interpolate
import os
import shutil

# ============================
# Geometrical data
# ============================
D_barr_ext = 2.5       #m
D_vess_int = 3.0       #m
t_th_ins = 0.05        #m 
k_th_ins = 1.4         #W/mK
L = 7                  #m

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
creep_flag_V = bool(0)
creep_flag_S = bool(0)

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
R_int = D_vess_int/2                        #m
R_barr_ext = D_barr_ext/2                   #m
v_flr = m_flr/rho                           #m³/s
G = E/(2*(1+nu))                            #MPa
P_int_MPa = P_int/10                        #MPa
P_cpp_MPa = P_cpp/10                        #MPa
Phi_0 = Phi_0 * 1e4                         #photons/(m²·s)
DeltaD_max = min(((D_vess_int*1000)+1270)/200, (D_vess_int*1000)/100)   #mm instead of in. - ASME gives 1250, but 1 inch = 25.4 mm

# ======================================
# Other necessary variables for file saving
# =====================================
current_directory = os.getcwd()                                                     # Get the current working directory
NTS_parent_directory_name = "Results_No_Thermal_Shield"                             # Parent folders
TS_parent_directory_name = "Results_Thermal_Shield"

NTS_directory_path = os.path.join(current_directory, NTS_parent_directory_name)     # Create the full path for the new directory
TS_directory_path = os.path.join(current_directory, TS_parent_directory_name)
NTS_plots_directory_path = os.path.join(NTS_directory_path, "Plots")
TS_plots_directory_path = os.path.join(TS_directory_path, "Plots")

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

print("\n\033[34m############################################################################################################################\033[0m")
print("\033[34mINITIAL SYSTEM-LEVEL ASSUMPTIONS\033[0m")
print("\033[34m############################################################################################################################\033[0m")
# ======================================
# Default Pressures Check
# ======================================
while True:
    try:
        Def_P_flag = int(input("\nAssume default pressures (75 bar = 7.5 MPa)? (1: Yes, 0: No): "))
        if Def_P_flag not in (0, 1):
            raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
        Def_P_flag = bool(Def_P_flag)
        break
    except ValueError:
        print("\033[31mPlease enter a valid integer.\033[0m")
    except RuntimeError as e:
        print(e)
if Def_P_flag:
    print("\033[34mThe default pressures have been assumed.\033[0m")
if not Def_P_flag:
    print("\033[34mThe default pressures have been discarded.\033[0m")
    P_int = float(input("\nSet the internal pressure (bar): "))
    P_int_MPa = P_int/10
    P_cpp = float(input("\nSet the external pressure (bar): "))
    P_cpp_MPa = P_cpp/10
    
    # ======================================
    # Stress/Strain Condition Input
    # ======================================
    while P_int != P_cpp: #Asks for this here, because asking for it in the sigmaL function would mean having to input the value for every iteration
        try:
            eps_choice = int(input("\nEnter the stress/strain condition (1: Plane Stress, 0: Plane Strain): "))
            if eps_choice not in (0, 1):
                raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
            break  
        except ValueError:
            print("\033[31mPlease enter a valid integer.\033[0m")
        except RuntimeError as e:
            print(e)

    if P_int != P_cpp:
        if eps_choice == 1:                                                                                           #Plane Stress
            print("\033[34mPlane stress has been assumed.\033[0m")
        elif eps_choice == 0:
            print("\033[34mPlane strain has been assumed.\033[0m")

# ======================================
# Heat Source Check
# ======================================
while True:
    try:
        q_0_flag = int(input("\nDo you want to account for the presence of the volumetric heat source q0 inside the vessel's wall? (1: Yes, 0: No): "))
        if q_0_flag not in (0, 1):
            raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
        q_0_flag = bool(q_0_flag)
        break  
    except ValueError:
        print("\033[31mPlease enter a valid integer.\033[0m")
    except RuntimeError as e:
        print(e)
        
# ======================================
# Thermal Shield Check   -   Only if volumetric heat source is considered
# ====================================== 
if q_0_flag:
    print("\033[34mThe presence of the volumetric heat source q0 has been considered.\033[0m")
    while True:
        try:
            TS_flag = int(input("\nDo you want to consider the presence of a thermal shield between the barrel and the vessel? (1: Yes, 0: No): "))
            if TS_flag not in (0, 1):
                raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
            TS_flag = bool(TS_flag)
            break  
        except ValueError:
            print("\033[31mPlease enter a valid integer.\033[0m")
        except RuntimeError as e:
            print(e)
elif not q_0_flag:
    print("\033[34mThe volumetric heat source q0 has been set to 0.\033[0m")
    TS_flag = bool(0)

# =============================================================================================================================================================
# THERMOMECHANICAL PROBLEM - POWER IMPOSED - NO THERMAL SHIELD
# =============================================================================================================================================================
if not TS_flag:
    print("\n\033[33m############################################################################################################################\033[0m")
    print("\033[33mTHERMOMECHANICAL PROBLEM - POWER IMPOSED - NO THERMAL SHIELD\033[0m")
    print("\033[33m############################################################################################################################\033[0m")
    # ============================
    # Computed additional data without the thermal shield
    # ============================
    v = m_flr/(rho*np.pi*((D_vess_int**2)-(D_barr_ext**2))/4)     #m/s
    Phi_0V = Phi_0                                                #All gamma rays reach the vessel
    
    # =============================================================================================================================================================
    # PURELY MECHANICAL PROBLEM
    # =============================================================================================================================================================
    print("\n\033[34m############################################################################################################################\033[0m")
    print("\033[34mPURELY MECHANICAL PROBLEM\033[0m")
    print("\033[34m############################################################################################################################\033[0m")
    while True:
        try:
            t = float(input("\nPlease enter the thickness of the vessel wall (m): "))
            if t <= 0 or t > 0.3:
                if t <= 0:
                    raise RuntimeError("\033[31mNegative or null thickness! Please enter a positive value.\033[0m")
                elif t > 0.3:
                    raise RuntimeError("\033[31mUnfeasible thickness! Cylinders thicker than 30cm are currently not possible.\033[0m")
            break
        except ValueError:
            print("\033[31mPlease enter a valid float.\033[0m")
        except RuntimeError as e:
            print(e)

    R_ext = R_int + t                           #m
    D_vess_ext = 2*R_ext                        #m
    rho_ii = (R_ext**2)/(R_ext**2 - R_int**2)
    rho_i = (R_int**2)/(R_ext**2 - R_int**2)
    Mar_criterion = R_int/t
    W = (DeltaD_max/1000)/((D_vess_int+D_vess_ext)/2)
    
    dr = 100
    r = np.linspace(R_int, R_ext, dr)

    # ============================
    # Mariotte Solution for a thin-walled cylinder (R_int = R_ext = R)
    # ============================
    def sigmaM_func (R_int, P_int_MPa, t): 
        sigma_rM_cyl_in = -P_int_MPa/2                        #Compressive
        sigma_tM_cyl_in = R_int*P_int_MPa/t
        sigma_zM_cyl_in = R_int*P_int_MPa/(2*t)

        sigma_rM_cyl_out = -P_cpp_MPa/2
        sigma_tM_cyl_out = -R_int*P_cpp_MPa/t                 #sigma_tM_sph = R_int*P_int_MPa/(2*t)
        sigma_zM_cyl_out = -R_int*P_cpp_MPa/(2*t)
        return (sigma_rM_cyl_in+sigma_rM_cyl_out, sigma_tM_cyl_in+sigma_tM_cyl_out, sigma_zM_cyl_in+sigma_zM_cyl_out)

    if Mar_criterion > 5:
        while True:
            try:
                Mariotte_flag = int(input("\nWith an initial thickness value of %.3f m, the vessel can be considered thin. Do you want to visualize the Mariotte solution for stress? (1: Yes, 0: No): " %t))
                if Mariotte_flag not in (0, 1):
                    raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                Mariotte_flag = bool(Mariotte_flag)
                break  
            except ValueError:
                print("\033[31mPlease enter a valid integer.\033[0m")
            except RuntimeError as e:
                print(e)
        sigma_M = sigmaM_func(R_int, P_int_MPa, t)
        sigma_rM = sigma_M[0]
        sigma_tM = sigma_M[1]
        sigma_zM = sigma_M[2]

        # ======================================
        # Plotting the stress profiles: Mariotte
        # ======================================
        os.makedirs(NTS_plots_directory_path, exist_ok=True)
        plot_file_path = os.path.join(NTS_plots_directory_path, "Stress Distribution in a thin-walled cylinder - Mariotte Solution.png")
        plt.figure(figsize=(15,10))
        plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
        plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
        plt.axhline(y = sigma_rM, color='red', label='Radial (r) Stress Mariotte')
        plt.axhline(y = sigma_tM, color='blue', label=r'Hoop ($\theta$) Stress Mariotte')
        plt.axhline(y = sigma_zM, color='green', label='Axial (z) Stress Mariotte')
        plt.plot(r, np.zeros(len(r)), color='black', linewidth='1', label='y=0')
        plt.xlabel('Radius (m)')
        plt.ylabel('Stress (MPa)')
        plt.title('Stress Distribution in a thin-walled cylinder - Mariotte Solution')
        plt.legend()
        plt.grid()
        plt.savefig(plot_file_path)
        if Mariotte_flag:
            plt.show()
            plt.close()
        elif not Mariotte_flag:
            plt.close()

    else:
        print("\n\033[34mThe cylinder can't be considered thin. Skipping Mariotte solution.\033[0m")
        Mariotte_flag = bool(0)

    # ============================ 
    # General Lamé Solution 
    # ============================
    def sigmaL_func(r, P_int_MPa, P_cpp_MPa):
        
        A = ((P_int_MPa*(R_int**2))-(P_cpp_MPa*(R_ext**2)))/((R_ext**2)-(R_int**2))
        B = (((R_int**2)*(R_ext**2))/((R_ext**2)-(R_int**2)))*(P_int_MPa-P_cpp_MPa)
        sigma_rL = lambda r: A - B/(r**2)
        sigma_tL = lambda r: A + B/(r**2)

        if P_int == P_cpp:                                                                                              #Hydrostatic Stress
            eps_z_a = (2*nu-1)*rho_ii*P_cpp_MPa/E
            eps_z_b = (1-2*nu)*rho_i*P_int_MPa/E

        elif P_int != P_cpp:
            if eps_choice == 1:                                                                                           #Plane Stress
                eps_z_a = 2*nu*rho_ii*P_cpp_MPa/E
                eps_z_b = -2*nu*rho_i*P_int_MPa/E
            elif eps_choice == 0:                                                                                         #Plane Strain
                eps_z_a = 0
                eps_z_b = 0 

        sigma_zL_a = E*eps_z_a - 2*nu*rho_ii*P_cpp_MPa  #a) P_int = 0
        sigma_zL_b = E*eps_z_b + 2*nu*rho_i*P_int_MPa   #b) P_cpp = 0
        return (sigma_rL(r), sigma_tL(r), sigma_zL_a + sigma_zL_b)              #Superposition Principle

    sigma_L = sigmaL_func(r, P_int_MPa, P_cpp_MPa)
    sigma_rL = sigma_L[0]  
    sigma_tL = sigma_L[1]
    sigma_zL = sigma_L[2]

    while True:
        try:
            Lame_flag = int(input("\nDo you want to visualize the Lamé solution? (1: Yes, 0: No): "))
            if Lame_flag not in (0, 1):
                raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
            Lame_flag = bool(Lame_flag)
            break  
        except ValueError:
            print("\033[31mPlease enter a valid integer.\033[0m")
        except RuntimeError as e:
            print(e)

    # ======================================
    # Plotting the stress profiles: Lamé
    # ======================================
    os.makedirs(NTS_plots_directory_path, exist_ok=True)
    plot_file_path = os.path.join(NTS_plots_directory_path, "Stress Distribution in the cylinder wall - Lamé Solution.png")
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
    plt.savefig(plot_file_path)
    if Lame_flag:
        plt.show()
        plt.close()
    elif not Lame_flag:
        plt.close()
    
    # =============================================================================================================================================================
    # PURELY THERMAL PROBLEM
    # =============================================================================================================================================================
    print("\n\033[34m############################################################################################################################\033[0m")
    print("\033[34mPURELY THERMAL PROBLEM\033[0m")
    print("\033[34m############################################################################################################################\033[0m")
    # ======================================
    # Radiation-induced heating in the vessel
    # ======================================
    Phi = lambda r: Phi_0V*np.exp(-mu_st*(r-R_int))    #1/(m²·s)
    I = lambda r: E_y_J*Phi(r)*B                       #W/(m²)
    q_0 = B*Phi_0V*E_y_J*mu_st*q_0_flag                #W/(m³)
    q_iii = lambda r: q_0*np.exp(-mu_st*(r-R_int))     #W/(m³)
    
    if q_0_flag == 1:
        # ======================================
        # Plotting the volumetric heat source profiles 
        # ======================================
        while True:
            try:
                hs_flag = int(input("\nDo you want to visualize the volumetric heat source q0 inside the vessel's wall? (1: Yes, 0: No): "))
                if hs_flag not in (0, 1):
                    raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                hs_flag = bool(hs_flag)
                break  
            except ValueError:
                print("\033[31mPlease enter a valid integer.\033[0m")
            except RuntimeError as e:
                print(e)

        plot_file_path = os.path.join(NTS_plots_directory_path, "Volumetric heat source profile across the vessel wall.png")
        plt.figure(figsize=(10,10))
        plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
        plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
        plt.plot(r, q_iii(r)/1e6, 'g', label='Radial (r) Volumetric heat source profile')
        plt.plot(r[0], q_iii(r[0])/1e6, 'or', label='Vessel Inner Surface Value')
        plt.plot(r[-1], q_iii(r[-1])/1e6, 'or', label='Vessel-Insulation Interface Value')
        plt.axhline(y = 0, color='black', linewidth='1', label='y=0')
        plt.xlabel('Radius (m)')
        plt.ylabel(r'$q_0$ (MW/m$^3$)')
        plt.title('Volumetric heat source profile across the vessel wall')
        plt.legend()
        plt.grid()
        plt.savefig(plot_file_path)
        if hs_flag:
            plt.show()
            plt.close()
        elif not hs_flag:
            plt.close()
    
    # ======================================
    # Dimensionless numbers and heat transfer coefficients
    # ======================================
    Pr = (Cp*mu)/k                                                                              #Prandtl number
    Pr_cpp = (Cp_cpp*mu_cpp)/k_cpp                                                              #Prandtl number of the containment water  
                                       
    Re = (rho*v*(D_vess_int-D_barr_ext))/mu                                                     #Reynolds number
    Nu_1 = 0.023*(Re**0.8)*(Pr**0.4)                                                            #Dittus-Boelter equation for forced convection
    h_1 = (Nu_1*k)/(D_vess_int-D_barr_ext)                                                      #W/(m²·K)

    Gr = (rho_cpp**2)*9.81*beta_cpp*DeltaT*(L**3)/(mu_cpp**2)                                   #Grashof number (Uses the external diameter as characteristic length, might wanna use L though?)
    Nu_2 = 0.13*((Gr*Pr_cpp)**(1/3))                                                            #McAdams correlation for natural convection
    h_2 = (Nu_2*k_cpp)/L                                                                        #W/(m²·K)
    R_th_2_tot = (1/(2*np.pi*(R_ext + t_th_ins)*L)) * ((((R_ext + t_th_ins)/k_th_ins)*np.log((R_ext + t_th_ins)/R_ext)) + (1/h_2))                          #Thermal Resistance of the insulation layer + natural convection outside the vessel
    u_2 = 1/(2*np.pi*(R_ext + t_th_ins)*L*R_th_2_tot)                                           #W/(m²·K)   -   Overall heat transfer coefficient outside the vessel

    # ======================================
    # Discretization Check
    # ======================================
    while True:
        try:
            Disc_flag = int(input("\nDo you want to adopt a discretization approach along z? (1: Yes, 0: No): "))
            if Disc_flag not in (0, 1):
                raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
            Disc_flag = bool(Disc_flag)
            break  
        except ValueError:
            print("\033[31mPlease enter a valid integer.\033[0m")
        except RuntimeError as e:
            print(e)

    # ======================================
    # 1D Approach: no discretization along z
    # ======================================
    if not Disc_flag:
        print("\033[34mNo discretization along z. Assuming constant temperature of the primary fluid T1.\033[0m")
        while True:
            try:
                T1_choice = int(input("\nWhat temperature do you want to use as T1 to compute C1 and C2? (0: T_in, 1: T_in + 10%, 2: T_in + 20%, 3: T_avg, 4: T_out_avg): "))
                if T1_choice not in (0, 1, 2, 3, 4):
                    raise RuntimeError("Invalid input! Please enter one of the allowed values: 1, 2, 3, 4.")
                break  
            except ValueError:
                print("\033[31mPlease enter a valid integer.\033[0m")
            except RuntimeError as e:
                print(e)
                
        if T1_choice == 0:                #All these temperatures are expressed in K. T_out_max and T_avg_log have been discarded in favor of margins on T_in, to account for transients due to the system's geometry
            print("\033[34mT1 = T_in has been assumed as the constant temperature of the primary fluid.\033[0m")
            T1 = T_in
        elif T1_choice == 1:
            print("\033[34mT1 = T_in + 10%% has been assumed as the constant temperature of the primary fluid.\033[0m")
            T1 = T_in * 1.1
        elif T1_choice == 2:
            print("\033[34mT1 = T_in + 20%% has been assumed as the constant temperature of the primary fluid.\033[0m")
            T1 = T_in * 1.2
        elif T1_choice == 3:
            print("\033[34mT1 = T_avg has been assumed as the constant temperature of the primary fluid.\033[0m")
            T1 = ((T_in + T_out_avg)/2)
        elif T1_choice == 4:
            print("\033[34mT1 = T_out_avg has been assumed as the constant temperature of the primary fluid.\033[0m")
            T1 = T_out_avg

        while True:
            try:
                adiab_flag = int(input("\nApply Adiabatic Outer Wall approximation? (1: Yes, 0: No): "))
                if adiab_flag not in (0, 1):
                    raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                adiab_flag = bool(adiab_flag)
                break  
            except ValueError:
                print("\033[31mPlease enter a valid integer.\033[0m")
            except RuntimeError as e:
                print(e)
        
        # ======================================
        # T profile constants for the vessel: general and under adiabatic outer wall approximation (dT/dx = 0 at r = R_ext)
        # ======================================
        if not adiab_flag:
            print("\033[34mThe Adiabatic Outer Wall approximation has not been applied.\033[0m")
            C1 = ((q_0/(k_st*mu_st**2))*(np.exp(-mu_st*t)-1)-(q_0/mu_st)*((1/h_1)+(np.exp(-mu_st*t)/u_2))-(T1-T_cpp))/(t+(k_st/h_1)+(k_st/u_2))
        elif adiab_flag:
            print("\033[34mThe Adiabatic Outer Wall approximation has been applied.\033[0m")
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
        """
        str1 = "\nT1 = T_in has been assumed: a logarithmic mean DeltaT could thus be useful to account for the T profile along z in an approximate way, even though the vessel wall temperature is not constant."
        str2 = "The heat flux computed with the regular DeltaT will still be displayed."
        str3 = "Do you want to adopt such an approach? (1: Yes, 0: No): "
        prompt = "\n".join([str1, str2, str3]) + " "
        if T1_choice == 0:
            while True:
                try:
                    LogDelta_flag = int(input(prompt))
                    if LogDelta_flag not in (0, 1):
                        raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                    LogDelta_flag = bool(LogDelta_flag)
                    break  
                except ValueError:
                    print("\033[31mPlease enter a valid integer.\033[0m")
                except RuntimeError as e:
                    print(e)
            if LogDelta_flag == 1:
                DeltaT_LM1 = ((T1-T_vessel(r[0]))-(T_out_avg-T_vessel(r[0])))/(np.log((T1-T_vessel(r[0]))/(T_out_avg-T_vessel(r[0]))))    #Log Mean Temperature Difference to account for T change along z, instead of just using T1-T_wall
                q_s1_log = h_1*DeltaT_LM1/1000                                                                                            #kW/m²
        """
        q_s1 = h_1*DeltaT_1/1000                                                                                                          #kW/m²
        q_s2 = u_2*(T_vessel(r[-1])-T_cpp)/1000                                                                                           #kW/m²

        # ======================================
        # Plotting the T profiles
        # ======================================
        while True:
            try:
                T_pl_flag = int(input("\nDo you want to visualize the T profile across the vessel's wall? (1: Yes, 0: No): "))
                if T_pl_flag not in (0, 1):
                    raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                T_pl_flag = bool(T_pl_flag)
                break  
            except ValueError:
                print("\033[31mPlease enter a valid integer.\033[0m")
            except RuntimeError as e:
                print(e)
                
        if (T_vessel_max - 273.15) > T_creep:
            creep_flag_V = bool(1)
        if not adiab_flag:
            plot_file_path = os.path.join(NTS_plots_directory_path, "Wall Temperature Profile, Average and Maximum.png")
            plt.figure(figsize=(10,10))
            plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
            plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
            plt.plot(r, T_vessel(r) - 273.15, label='Radial (r) T Profile')
            plt.plot(r_T_vessel_max, T_vessel_max - 273.15,'or',label='Max T')
            plt.axhline(y = T_vessel_avg - 273.15, color='green', label='Wall Average T')
            plt.xlabel('Radius (m)')
            plt.ylabel('T (°C)')
            plt.title('Wall Temperature Profile, Average and Maximum')
            plt.legend()
            plt.grid()
            plt.savefig(plot_file_path)
            if T_pl_flag:
                plt.show()
                plt.close()
            elif not T_pl_flag:
                plt.close()
            
        elif adiab_flag:
             
            # ======================================
            # Under Adiabatic Outer Wall Approximation
            # ======================================
            plot_file_path = os.path.join(NTS_plots_directory_path, "Wall Temperature Profile, Average and Maximum under AOW Approximation.png")
            plt.figure(figsize=(10,10))
            plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
            plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
            plt.plot(r, T_vessel(r) - 273.15, label='Radial (r) T Profile')
            plt.plot(r_T_vessel_max, T_vessel_max - 273.15,'or', label='Max T')
            plt.axhline(y = T_vessel_avg - 273.15, color='green', label='Wall Average T')
            plt.xlabel('Radius (m)')
            plt.ylabel('T (°C)')
            plt.title('Wall Temperature Profile, Average and Maximum under AOW Approximation')
            plt.legend()
            plt.grid()
            plt.savefig(plot_file_path)
            if T_pl_flag:
                plt.show()
                plt.close()
            elif not T_pl_flag:
                plt.close()
    
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
                    raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                sigma_th_pl_flag = bool(sigma_th_pl_flag)
                break  
            except ValueError:
                print("\033[31mPlease enter a valid integer.\033[0m")
            except RuntimeError as e:
                print(e)

        plot_file_path = os.path.join(NTS_plots_directory_path, "Wall Thermal Stress Profiles and Maximum Hoop Stress.png")
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
        plt.savefig(plot_file_path)
        if sigma_th_pl_flag:
            plt.show()
            plt.close()
        elif not sigma_th_pl_flag:
            plt.close()

        # ======================================
        # Principal stresses sum and elastic regime verification in the vessel
        # ======================================
        sigma_r_totM = sigma_rM + sigma_r_th_V
        sigma_t_totM = sigma_tM + sigma_t_th_V
        sigma_z_totM = sigma_zM + sigma_z_th_V
        
        sigma_r_totL = sigma_rL + sigma_r_th_V
        sigma_t_totL = sigma_tL + sigma_t_th_V
        sigma_z_totL = sigma_zL + sigma_z_th_V
        
        # ============================ 
        # Vessel Comparison stress - Guest-Tresca Theory - Mariotte/Lamé only
        # ============================
        sigma_cTR_M_PO = np.max([abs(sigma_tM - sigma_rM), abs(sigma_zM - sigma_rM), abs(sigma_tM - sigma_zM)])
        sigma_cTR_L_PO = []
        for i in range(len(r)):
            sigma_cTR_L_PO.append(np.max([abs(sigma_tL - sigma_rL)[i], abs(sigma_zL - sigma_rL)[i], abs(sigma_tL - sigma_zL)[i]]))
        sigma_cTR_L_PO = max(sigma_cTR_L_PO)

        # ============================ 
        # Vessel Comparison stress - Guest-Tresca Theory - Mariotte/Lamé + Thermal stresses
        # ============================
        sigma_cTR_M = []
        for i in range(len(r)):
            sigma_cTR_M.append(np.max([abs(sigma_t_totM - sigma_r_totM)[i], abs(sigma_z_totM - sigma_r_totM)[i], abs(sigma_t_totM - sigma_z_totM)[i]]))
        sigma_cTR_M = max(sigma_cTR_M)
        sigma_cTR_L = []
        for i in range(len(r)):
            sigma_cTR_L.append(np.max([abs(sigma_t_totL - sigma_r_totL)[i], abs(sigma_z_totL - sigma_r_totL)[i], abs(sigma_t_totL - sigma_z_totL)[i]]))
        sigma_cTR_L = max(sigma_cTR_L)

        # ======================================
        # Plotting the maximum thermal stress via the design curves
        # ======================================
        while True:
            try:
                des_pl_flag = int(input("\nDo you want to visualize a plot of the design curves and the maximum thermal stress in the vessel? (1: Yes, 0: No): "))
                if des_pl_flag not in (0, 1):
                    raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                des_pl_flag = bool(des_pl_flag)
                break  
            except ValueError:
                print("\033[31mPlease enter a valid integer.\033[0m")
            except RuntimeError as e:
                print(e)

        plot_file_path = os.path.join(NTS_plots_directory_path, "Design Curves.png")
        plt.figure(figsize=(10,10))
        plt.xlim(ba_ratio_plot[0], ba_ratio_plot[-1])
        plt.plot(ba_ratio_plot, L_Interpolator(ba_ratio_plot), 'k', label=f'Iso-mu = {mu_L} 1/m')
        plt.plot(ba_ratio_plot, R_Interpolator(ba_ratio_plot), 'k', label=f'Iso-mu = {mu_R} 1/m')
        plt.text(ba_ratio_plot[-1]+0.001, L_Interpolator(ba_ratio_plot)[-1], f'Iso-mu = {mu_L}', color='black', fontsize=10)
        plt.text(ba_ratio_plot[-1]+0.001, R_Interpolator(ba_ratio_plot)[-1], f'Iso-mu = {mu_R}', color='black', fontsize=10)
        plt.plot(R_ext/R_int, sigmaT,'or', label=r'Current $\sigma$$_T$')
        plt.xlabel('R$_{ext}$/R$_{int}$')
        plt.ylabel(r'$\sigma$$_T$')
        plt.title('Design curves')
        plt.legend()
        plt.grid()
        plt.savefig(plot_file_path)
        if des_pl_flag:
            plt.show()
            plt.close()
        elif not des_pl_flag:
            plt.close()
        
        # ============================ 
        # Yield Stress and Stress Intensity Data Interpolation
        # ============================
        T_des_vessel = T_vessel_avg                                                     #K
        T_des_vessel_C = 270 #T_des_vessel - 273.15                                          #°C
        p_yield = np.polyfit(T_thr, sigma_y, deg = len(T_thr)-1)
        p_intensity = np.polyfit(T_thr, sigma_in, deg = len(T_thr)-1)
        
        Yield_Interpolator = lambda x: np.polyval(p_yield, x)                           #Yield Stress Interpolation Polynomial (n-1)
        Yield_CubicSpline = interpolate.CubicSpline(T_thr, sigma_y)                     #Yield Stress Cubic Spline Interpolation
        Yield_stress = Yield_CubicSpline(T_des_vessel_C)
        
        Intensity_Interpolator = lambda x: np.polyval(p_intensity, x)                   #Stress Intensity Interpolation Polynomial (n-1)
        Intensity_CubicSpline = interpolate.CubicSpline(T_thr, sigma_in)                #Stress Intenisty Cubic Spline Interpolation
        Stress_Intensity = Intensity_CubicSpline(T_des_vessel_C)
        sigma_allowable = Stress_Intensity                                              #MPa

        """
        # ======================================
        # Vessel Thermomechanical Integrity Verification    -   Lamé + Thermal stresses
        # ======================================
        if max(abs(sigma_r_totL)) > 3*Stress_Intensity or max(abs(sigma_t_totL)) > 3*Stress_Intensity or max(abs(sigma_z_totL)) > 3*Stress_Intensity:
            flag_primsec = bool(1)
        else:
            flag_primsec = bool(0)

        if max(abs(sigma_rL)) > Stress_Intensity or max(abs(sigma_tL)) > Stress_Intensity or sigma_zL > Stress_Intensity:
            flag_prim = bool(1)
        else:
            flag_prim = bool(0)
         
        # ======================================
        # Vessel Thermomechanical Integrity Verification    -   Mariotte + Thermal stresses
        # ======================================
        if max(abs(sigma_r_totM)) > 3*Stress_Intensity or max(abs(sigma_t_totM)) > 3*Stress_Intensity or max(abs(sigma_z_totM)) > 3*Stress_Intensity:
            flag_primsec = bool(1)
        else:
            flag_primsec = bool(0)

        if sigma_rM > Stress_Intensity or sigma_tM > Stress_Intensity or sigma_zM > Stress_Intensity:
            flag_prim = bool(1)
        else:
            flag_prim = bool(0)
        """
        # ======================================
        # Vessel Thermomechanical Integrity Verification    -   Tresca-Mariotte
        # ======================================
        if sigma_cTR_M > 3*Stress_Intensity:
            flag_primsec = bool(1)
        else:
            flag_primsec = bool(0)

        if sigma_cTR_M_PO > Stress_Intensity:
            flag_prim = bool(1)
        else:
            flag_prim = bool(0)
            
        # ======================================
        # Without thermal shield
        # ======================================
        while True:
            try:
                Interp_pl_flag = int(input("\nDo you want to visualize a plot of the Yield Stress and Stress Intensity as given by ASME for the vessel? (1: Yes, 0: No): "))
                if Interp_pl_flag not in (0, 1):
                    raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                Interp_pl_flag = bool(Interp_pl_flag)
                break  
            except ValueError:
                print("\033[31mPlease enter a valid integer.\033[0m")
            except RuntimeError as e:
                print(e)
        
        if max(T_thr) > T_des_vessel_C:
            Tplot = np.linspace(min(T_thr), max(T_thr), 1000)
        else:
            Tplot = np.linspace(min(T_thr), T_des_vessel_C, 1000)
        
        # ============================ 
        # Yield Stress and Stress Intensity Data Plots  -   Vessel
        # ============================
        plot_file_path = os.path.join(NTS_plots_directory_path, "Yield Stress and Stress Intensity Data.png")
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
        plt.savefig(plot_file_path)
        if Interp_pl_flag:
            plt.show()
            plt.close()
        elif not Interp_pl_flag:
            plt.close()
    
        # ============================ 
        # Sizing of a thick cylinder under external pressure
        # ============================
        if Mar_criterion > 5:
            while True:
                try:
                    ThinTubes_flag = int(input("\nThe vessel's wall can be considered thin. Are you interested in the thin tube limits for Elastic Instability and Plastic Collapse? (1: Yes, 0: No): "))
                    if ThinTubes_flag not in (0, 1):
                        raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                    ThinTubes_flag = bool(ThinTubes_flag)
                    break  
                except ValueError:
                    print("\033[31mPlease enter a valid integer.\033[0m")
                except RuntimeError as e:
                    print(e)

            if ThinTubes_flag:
                print("\033[34mThe thin tube limits were adopted.\033[0m")
                p_E_fun = lambda Dt: 2 * (E/(1-(nu**2))) * (1/(Dt**3))              #Elastic Instability Limit for Thin Tubes
                p_0_fun = lambda Dt: 2 * Yield_stress * 1/Dt                        #Plastic Collapse Limit for Thin Tubes

            elif not ThinTubes_flag:
                print("\033[34mSkipping thin tube limits.\033[0m")
        else:
            print("\n\033[34mThe cylinder can't be considered thin. Skipping thin tube limits.\033[0m")
            ThinTubes_flag = bool(0)

        # ============================ 
        # Corradi Design Procedure
        # ============================
        if ThinTubes_flag:
            while True:
                try:
                    Corradi_flag = int(input("\nAre you interested in the Corradi Design Procedure? (1: Yes, 0: No): "))
                    if Corradi_flag not in (0, 1):
                        raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                    Corradi_flag = bool(Corradi_flag)
                    break  
                except ValueError:
                    print("\033[31mPlease enter a valid integer.\033[0m")
                except RuntimeError as e:
                    print(e)
            
        elif not ThinTubes_flag:
            print("\033[34mAdopting Corradi Design Procedure.\033[0m")
            Corradi_flag = bool(1)

        q_E_fun = lambda Dt: 2 * (E/(1-(nu**2))) * (1/(Dt*((Dt-1)**2)))     #Elastic Instability Limit for Thick Tubes
        q_0_fun = lambda Dt: 2 * Yield_stress * 1/Dt * (1+(1/(2*Dt)))       #Plastic Collapse Limit for Thick Tubes
        Dt_Crit_Ratio = np.sqrt(E/(Yield_stress*(1-(nu**2))))
        Current_Slenderness = (D_vess_int+2*t)/t
        Dt_ratio_plot = np.linspace(2,50,1000)

        if Corradi_flag:
            # ============================ 
            # Corradi Design Procedure
            # ============================
            while True:
                try:
                    s = float(input("Please enter a safety factor between 1.5 and 2 for the Corradi design procedure: "))
                    if s < 1.5 or s > 2:
                        raise RuntimeError("\033[31mInvalid input! Please enter a safety factor between 1.5 and 2.\033[0m")
                    break  
                except ValueError:
                    print("\033[31mPlease enter a valid float.\033[0m")
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
            if (P_cpp < 10*Corradi_vessel[1]):
                buckling_flag = bool(1)
            else:
                buckling_flag = bool(0)
        
        elif not Corradi_flag:
            print("\033[34mSkipping Corradi Design Procedure.\033[0m")
        
        # ============================ 
        # Elastic instability and plastic collapse curves
        # ============================
        if ThinTubes_flag and not Corradi_flag:
            while True:
                try:
                    Collapse_pl_flag = int(input("\nDo you want to visualize the buckling and plastic collapse curves for thin and thick tubes? (1: Yes, 0: No): "))
                    if Collapse_pl_flag not in (0, 1):
                        raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                    Collapse_pl_flag = bool(Collapse_pl_flag)
                    break  
                except ValueError:
                    print("\033[31mPlease enter a valid integer.\033[0m")
                except RuntimeError as e:
                    print(e)
                
            # ============================ 
            # Plastic collapse and buckling Plots
            # ============================
            plot_file_path = os.path.join(NTS_plots_directory_path, "Plastic Collapse and Buckling Curves.png")
            plt.figure(figsize = (8, 8))
            plt.xlim(0,50)
            plt.ylim(0.1,max(q_E_fun(Dt_ratio_plot)))
            plt.semilogy(Dt_ratio_plot, p_E_fun(Dt_ratio_plot), 'blue', label='p$_E$')
            plt.semilogy(Dt_ratio_plot, q_E_fun(Dt_ratio_plot), '--b', label='q$_E$')
            plt.semilogy(Dt_ratio_plot, p_0_fun(Dt_ratio_plot), 'red', label='p$_0$')
            plt.semilogy(Dt_ratio_plot, q_0_fun(Dt_ratio_plot), '--r', label='q$_0$')
            plt.axvline(x = Dt_Crit_Ratio, color = 'black', linewidth = '3', label = 'Critical Slenderness')
            plt.axvline(x = Current_Slenderness, color = 'green', linestyle='--', linewidth = '1.5', label = 'Current Vessel Slenderness')
            plt.plot(Current_Slenderness, Corradi_vessel[1], 'og', label='Current Vessel Allowable Pressure q$_a$')
            plt.fill_betweenx((0.1,max(q_E_fun(Dt_ratio_plot))), 0, Dt_Crit_Ratio, color='lightgreen', alpha=0.40, label='Plastic collapse dominated zone')
            plt.fill_betweenx((0.1,max(q_E_fun(Dt_ratio_plot))), Dt_Crit_Ratio, 50, color='orange', alpha=0.30, label='Elastic instability dominated zone')
            plt.xlabel("Geometrical Slenderness D/t")
            plt.ylabel("Theoretical Limit Values (MPa)")
            plt.title("Plastic Collapse and Buckling Curves")
            plt.legend()
            plt.grid()
            plt.savefig(plot_file_path)
            if Collapse_pl_flag:
                plt.show()
                plt.close()
            elif not Collapse_pl_flag:
                plt.close()

        elif ThinTubes_flag and Corradi_flag:
            while True:
                try:
                    Collapse_pl_flag = int(input("\nDo you want to visualize the buckling and plastic collapse curves for thin and thick tubes and the Corradi curve? (1: Yes, 0: No): "))
                    if Collapse_pl_flag not in (0, 1):
                        raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                    Collapse_pl_flag = bool(Collapse_pl_flag)
                    break  
                except ValueError:
                    print("\033[31mPlease enter a valid integer.\033[0m")
                except RuntimeError as e:
                    print(e)

            # ============================ 
            # Plastic collapse and buckling Plots
            # ============================
            plot_file_path = os.path.join(NTS_plots_directory_path, "Plastic Collapse and Buckling Curves - With Corradi.png")
            plt.figure(figsize = (8, 8))
            plt.xlim(0,50)
            plt.ylim(0.1,max(q_E_fun(Dt_ratio_plot)))
            plt.subplot(1,2,1)
            plt.semilogy(Dt_ratio_plot, p_E_fun(Dt_ratio_plot), 'blue', label='p$_E$')
            plt.semilogy(Dt_ratio_plot, q_E_fun(Dt_ratio_plot), '--b', label='q$_E$')
            plt.semilogy(Dt_ratio_plot, p_0_fun(Dt_ratio_plot), 'red', label='p$_0$')
            plt.semilogy(Dt_ratio_plot, q_0_fun(Dt_ratio_plot), '--r', label='q$_0$')
            plt.semilogy(Dt_ratio_plot, Corradi(Dt_ratio_plot)[0], 'orange', label='Corradi q$_c$')
            plt.axvline(x = Dt_Crit_Ratio, color = 'black', linewidth = '3', label = 'Critical Slenderness')
            plt.axvline(x = Current_Slenderness, color = 'green', linestyle='--', linewidth = '1.5', label = 'Current Vessel Slenderness')
            plt.plot(Current_Slenderness, Corradi_vessel[1], 'og', label='Current Vessel Allowable Pressure q$_a$')
            plt.fill_betweenx((0.1,max(q_E_fun(Dt_ratio_plot))), 0, Dt_Crit_Ratio, color='lightgreen', alpha=0.40, label='Plastic collapse dominated zone')
            plt.fill_betweenx((0.1,max(q_E_fun(Dt_ratio_plot))), Dt_Crit_Ratio, 50, color='orange', alpha=0.30, label='Elastic instability dominated zone')
            plt.xlabel("Geometrical Slenderness D/t")
            plt.ylabel("Theoretical Limit Values (MPa)")
            plt.title("Plastic Collapse and Buckling Curves")
            plt.legend()
            plt.grid()

            plt.subplot(1,2,2)
            plt.plot(Dt_ratio_plot, Corradi(Dt_ratio_plot)[3], 'k', label=r'Corradi $\mu$')
            plt.xlabel("Geometrical Slenderness D/t")
            plt.ylabel(r"Corradi $\mu$")
            plt.title(r"$\mu$ coefficient - Corradi Procedure")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(plot_file_path)
            if Collapse_pl_flag:
                plt.show()
                plt.close()
            elif not Collapse_pl_flag:
                plt.close()
                
        # ============================ 
        # Minimum thickness under internal pressure check
        # ============================
        t_min = (P_int_MPa*R_int)/(Stress_Intensity - 0.5*P_int_MPa)
        isAbove = t >= t_min
        
        # ============================ 
        # Current case specific path creation
        # ============================
        case_directory_name = []
        case_directory_name.append(f"t_{t}m")
        case_directory_name.append(f"_T_desV_{T_des_vessel_C}C")
        if Def_P_flag:
            case_directory_name.append("_Def_P")
        else:
            case_directory_name.append("_Pint_%.1f_MPa_Pext_%.1f_MPa" %(P_int, P_cpp))
            if P_int != P_cpp:
                if eps_choice == 1:
                    case_directory_name.append("_Plane_Strain")
                elif eps_choice == 0:
                    case_directory_name.append("_Plane_Stress")
        if q_0_flag:
            case_directory_name.append("_q0")
        if Disc_flag:
            case_directory_name.append("_2D")
        if T1_choice == 0:
            case_directory_name.append("_Tin")
        elif T1_choice == 1:
            case_directory_name.append("_T_in + 10%%")
        elif T1_choice == 2:
            case_directory_name.append("_T_in + 20%%")
        elif T1_choice == 3:
            case_directory_name.append("_T_avg")
        elif T1_choice == 4:
            case_directory_name.append("_T_out_avg")
        if adiab_flag:
            case_directory_name.append("_AOW")
        if ThinTubes_flag:
            case_directory_name.append("_ThinTubes")
        if Corradi_flag:
            case_directory_name.append("_Corradi_s_%.2f" %s)
        case_directory_path = os.path.join(NTS_directory_path, "".join(case_directory_name))
        
        # ============================ 
        # Final results printing and saving
        # ============================
        if os.path.exists(case_directory_path):
            shutil.rmtree(case_directory_path)                                       # Deletes the pre-existing folder
        if not os.path.exists(case_directory_path):                                  # Create the directory if it doesn't exist
            os.makedirs(case_directory_path, exist_ok=True)                          # Exist_ok=True avoids error if directory already exists
        
        file_path = os.path.join(case_directory_path, "Final_Results.txt")           # Specify the file path inside the newly created directory
        with open(file_path, "w") as file:
            output_lines = []                                                        # Create a list to hold the printed messages
            
            # ============================
            # Hypothesis and data: not printed, saved only
            # ============================
            output_lines.append("################################################### Hypothesis and data ####################################################")
            output_lines.append("============================================================================================================================")
            output_lines.append("\nDefault pressures assumed: %s" %Def_P_flag)
            if not Def_P_flag:
                output_lines.append("Internal pressure: %.3f MPa" %P_int_MPa)
                output_lines.append("External pressure: %.3f MPa" %P_cpp_MPa)
            if P_int != P_cpp:
                output_lines.append("\nAssumed stress/strain condition (1: Plane Stress, 0: Plane Strain): %d" %eps_choice)
            output_lines.append("\n============================================================================================================================")
            output_lines.append("\nPresence of the volumetric heat source q0: %s" %q_0_flag)
            if q_0_flag:
                output_lines.append("Presence of the thermal shield: %s" %TS_flag)
            output_lines.append("\n============================================================================================================================")
            output_lines.append("\nDiscretization along z: %s" %Disc_flag)
            if T1_choice == 0:
                output_lines.append("Chosen temperature T1 to compute C1, C2: T_in = %.3s °C" %(T1-273.15))
            elif T1_choice == 1:
                output_lines.append("Chosen temperature T1 to compute C1, C2: T_in + 10% = %.3s °C" %(T1-273.15))
            elif T1_choice == 2:
                output_lines.append("Chosen temperature T1 to compute C1, C2: T_in + 20% = %.3s °C" %(T1-273.15))
            elif T1_choice == 3:
                output_lines.append("Chosen temperature T1 to compute C1, C2: T_avg = %.3s °C" %(T1-273.15))
            elif T1_choice == 4:
                output_lines.append("Chosen temperature T1 to compute C1, C2: T_out_avg = %.3s °C" %(T1-273.15))
            output_lines.append("Adiabatic Outer Wall approximation adopted: %s" %adiab_flag)
            #output_lines.append("Logarithmic Mean DeltaT approach adopted for inner heat flux computation: %d" %LogDelta_flag)
            output_lines.append("\n============================================================================================================================")
            output_lines.append("\nThin tube limits for Elastic Instability and Plastic Collapse adopted: %s" %ThinTubes_flag)
            output_lines.append("Corradi Design Procedure adopted: %s" %Corradi_flag)
            if Corradi_flag:
                output_lines.append("Safety coefficient adopted for the Corradi Design Procedure: %.3f" %s)
            output_lines.append("\n============================================================================================================================")
            
            # ============================
            # Actual Results
            # ============================
            output_lines.append("\n\n\n\n###################################################### Final  Results ######################################################")
            output_lines.append("============================================================================================================================")
            output_lines.append("\nCurrent vessel wall thickness: %.3f m" %t)
            output_lines.append("Vessel max ovality W: %.5f = %.3f%%" %(W,W*100))
            output_lines.append("Maximum permissible deviation from theoretical form for the vessel according to NB-4221.2: e = %.3f m" %(0.3*t))
            output_lines.append("Maximum difference in cross-sectional diameters: %.3f mm" %DeltaD_max)
            if isAbove:
                output_lines.append("\n============================================================================================================================")
                output_lines.append("\nThe current vessel wall thickness is equal to or greater than the minimum thickness required under internal pressure: %.3f m" %t_min)
                output_lines.append("\n============================================================================================================================")
            elif not isAbove:
                output_lines.append("\n============================================================================================================================")
                output_lines.append("WARNING: The current vessel wall thickness is below the minimum thickness required under internal pressure: %.3f m" %t_min)
                output_lines.append("============================================================================================================================")

            # ============================
            # Heat Transfer Results
            # ============================
            output_lines.append("\n\n\n\n################################################## Heat transfer results ###################################################")
            output_lines.append("============================================================================================================================")
            output_lines.append("\nVolumetric heat source at the vessel inner surface: %.3f W/m³" %q_iii(r[0]))
            output_lines.append("Volumetric heat source at the vessel-insulation interface: %.3f W/m³" %q_iii(r[-1]))
            output_lines.append("\n============================================================================================================================")
            output_lines.append("\nHeat transfer coefficient h1 = %.3f W/(m²·K)" %h_1)
            output_lines.append("Heat transfer coefficient h2 = %.3f W/(m²·K)" %h_2)
            output_lines.append("Overall heat transfer coefficient outside the vessel u2 = %.3f W/(m²·K)" %u_2)
            # if LogDelta_flag == 1:
            #     output_lines.append("\nThermal power flux on the inner vessel surface - Logarithmic Mean DeltaT Approach: %.3f kW/m²" %q_s1_log)
            output_lines.append("\n============================================================================================================================")
            output_lines.append("\nThermal power flux on the inner vessel surface: %.3f kW/m²" %q_s1)
            output_lines.append("Thermal power flux on the outer vessel surface: %.3f kW/m²" %q_s2)
            output_lines.append("\n============================================================================================================================")
            
            # ============================ 
            # Temperature Results
            # ============================
            output_lines.append("\n\n\n\n####################################################### Temperatures #######################################################")
            output_lines.append("============================================================================================================================")
            if not adiab_flag:
                output_lines.append("\nAverage Vessel Temperature (numerical integration): %.3f °C" %(T_vessel_avg - 273.15))
                #output_lines.append("Average Vessel Temperature (analytical integration): %.3f °C" %T_vessel_avg_2)
                output_lines.append("Maximum Vessel Temperature: %.3f °C at r = %.3f m" %(T_vessel_max - 273.15, r_T_vessel_max))
                output_lines.append("Vessel Temperature at the inner surface: %-3f °C at r = %.3f m" %(T_vessel(r)[0] - 273.15, r[0]))
                output_lines.append("Vessel Temperature at the outer surface: %-3f °C at r = %.3f m" %(T_vessel(r)[-1] - 273.15, r[-1]))
                if creep_flag_V:
                    output_lines.append("\n============================================================================================================================")
                    output_lines.append("WARNING: The maximum vessel temperature T = %.3f °C exceeds the creep threshold temperature of %d °C!" %(T_vessel_max - 273.15, T_creep))
                    output_lines.append("============================================================================================================================")
                elif not creep_flag_V:
                    output_lines.append("\nThere is no risk of thermal creep occurring in the vessel.")
                    output_lines.append("\n============================================================================================================================")
                    
            elif adiab_flag:
                output_lines.append("\nAverage Vessel Temperature under Adiabatic Outer Wall approximation (numerical integration): %.3f °C" %(T_vessel_avg - 273.15))
                #output_lines.append("Average Vessel Temperature under Adiabatic Outer Wall approximation (analytical integration): %.3f °C" %T_vessel_avg_2)
                output_lines.append("Maximum Vessel Temperature under Adiabatic Outer Wall approximation: %.3f °C at r = %.3f m" %(T_vessel_max - 273.15, r_T_vessel_max))
                output_lines.append("Vessel Temperature at the inner surface under Adiabatic Outer Wall approximation: %-3f °C at r = %.3f m" %(T_vessel(r)[0] - 273.15, r[0]))
                output_lines.append("Vessel Temperature at the outer surface under Adiabatic Outer Wall approximation: %-3f °C at r = %.3f m" %(T_vessel(r)[-1] - 273.15, r[-1]))
                if creep_flag_V:
                    output_lines.append("\n============================================================================================================================")
                    output_lines.append("WARNING: The maximum vessel temperature T = %.3f °C exceeds the creep threshold temperature of %d °C!" %(T_vessel_max - 273.15, T_creep))
                    output_lines.append("============================================================================================================================")
                elif not creep_flag_V:
                    output_lines.append("\nThere is no risk of thermal creep occurring in the vessel.")
                    output_lines.append("\n============================================================================================================================")               
            
            # ============================ 
            # Stress Results
            # ============================
            output_lines.append("\n\n\n\n######################################################### Stresses #########################################################")
            output_lines.append("============================================================================================================================")
            output_lines.append("\nMaximum Thermal Hoop Stress in the vessel: %.3f Mpa at r = %.3f m" %(sigma_t_th_V_max, r_sigma_t_th_V_max))
            #output_lines.append("Maximum Thermal Hoop Stress (Simplified formula): %.3f Mpa at r = %.3f m" %(sigma_t_th_V_max_SIMP, r_sigma_t_th_V_max_SIMP))
            output_lines.append("Maximum thermal hoop stress via design curves: %.3f MPa" %sigma_t_th_max_DES)
            output_lines.append("\n============================================================================================================================")
            output_lines.append("\nGuest-Tresca comparison stress of primary stresses only in the vessel - Mariotte solution: %.3f Mpa" %sigma_cTR_M_PO)
            output_lines.append("Guest-Tresca comparison stress of primary stresses only in the vessel - Lamé solution: %.3f Mpa" %sigma_cTR_L_PO)
            output_lines.append("Guest-Tresca comparison stress in the vessel - Mariotte solution: %.3f Mpa" %sigma_cTR_M)
            output_lines.append("Guest-Tresca comparison stress in the vessel - Lamé solution: %.3f Mpa" %sigma_cTR_L)
            output_lines.append("\n============================================================================================================================")

            output_lines.append("\nFor a design vessel temperature of %.3f °C: " %T_des_vessel_C)
            output_lines.append("Yield Stress: Sy = %.3f MPa" %Yield_stress)
            output_lines.append("Stress Intensity: Sm = %.3f MPa" %Stress_Intensity)
            output_lines.append("Allowable Stress: %.3f MPa" %sigma_allowable)
            output_lines.append("\n============================================================================================================================")
            
            # ============================ 
            # Vessel
            # ============================
            output_lines.append("\n\n\n\n########################################################## Vessel ##########################################################")
            output_lines.append("============================================================================================================================")
            #Corradi_vessel = Corradi(np.array([Current_Slenderness]))
            if flag_primsec or flag_prim:
                output_lines.append("\nAccording to Lamé:")
                output_lines.append("Maximum absolute value of the total radial stress: %.3f MPa\nMaximum absolute value of the total hoop stress: %.3f MPa\nMaximum absolute value of the total axial stress: %.3f MPa" %(max(abs(sigma_r_totL)),max(abs(sigma_t_totL)),max(abs(sigma_z_totL))))
                output_lines.append("\nMaximum value of the primary radial stress: %.3f MPa\nMaximum value of the primary hoop stress: %.3f MPa\nPrimary axial stress: %.3f MPa" %(max(sigma_rL),max(sigma_tL),sigma_zL))
                if max(abs(sigma_r_totL)) > 3*Stress_Intensity:
                    output_lines.append("\nThe maximum value of the total radial stress exceeds allowable stress.")
                if max(abs(sigma_t_totL)) > 3*Stress_Intensity:
                    output_lines.append("\nThe maximum value of the total hoop stress exceeds allowable stress.")
                if max(abs(sigma_z_totL)) > 3*Stress_Intensity:
                    output_lines.append("\nThe maximum value of the total axial stress exceeds allowable stress.")
                if max(sigma_rL) > Stress_Intensity:
                    output_lines.append("\nThe maximum value of the primary radial stress exceeds allowable stress.")
                if max(sigma_tL) > Stress_Intensity:
                    output_lines.append("\nThe maximum value of the primary hoop stress exceeds allowable stress.")
                if sigma_zL > Stress_Intensity:
                    output_lines.append("\nThe primary axial stress exceeds allowable stress.")
                    
                output_lines.append("\n============================================================================================================================")
                output_lines.append("\nAccording to Mariotte:")
                output_lines.append("Maximum absolute value of the total radial stress: %.3f MPa\nMaximum absolute value of the total hoop stress: %.3f MPa\nMaximum absolute value of the total axial stress: %.3f MPa" %(max(abs(sigma_r_totM)),max(abs(sigma_t_totM)),max(abs(sigma_z_totM))))
                output_lines.append("\nPrimary radial stress: %.3f MPa\nPrimary hoop stress: %.3f MPa\nPrimary axial stress: %.3f MPa" %(sigma_rM,sigma_tM,sigma_zM))
                if max(abs(sigma_r_totM)) > 3*Stress_Intensity:
                    output_lines.append("\nThe maximum value of the total radial stress exceeds allowable stress.")
                if max(abs(sigma_t_totM)) > 3*Stress_Intensity:
                    output_lines.append("\nThe maximum value of the total hoop stress exceeds allowable stress.")
                if max(abs(sigma_z_totM)) > 3*Stress_Intensity:
                    output_lines.append("\nThe maximum value of the total axial stress exceeds allowable stress.")
                if sigma_rM > Stress_Intensity:
                    output_lines.append("\nThe primary radial stress exceeds allowable stress.")
                if sigma_tM > Stress_Intensity:
                    output_lines.append("\nThe primary hoop stress exceeds allowable stress.")
                if sigma_zM > Stress_Intensity:
                    output_lines.append("\nThe primary axial stress exceeds allowable stress.")
                output_lines.append("\n============================================================================================================================")
                output_lines.append("WARNING: The current stress state in the vessel is not acceptable.")
                output_lines.append("============================================================================================================================")
            
            elif not flag_primsec and not flag_prim:
                output_lines.append("\nAccording to Lamé:") 
                output_lines.append("Maximum absolute value of the total radial stress: %.3f MPa\nMaximum absolute value of the total hoop stress: %.3f MPa\nMaximum absolute value of the total axial stress: %.3f MPa" %(max(abs(sigma_r_totL)),max(abs(sigma_t_totL)),max(abs(sigma_z_totL))))
                output_lines.append("All are lower than 3Sm = %.3f MPa" %(3*Stress_Intensity))         
                output_lines.append("\nMaximum value of the primary radial stress: %.3f MPa\nMaximum value of the primary hoop stress: %.3f MPa\nPrimary axial stress: %.3f MPa" %(max(sigma_rL),max(sigma_tL),sigma_zL))
                output_lines.append("\nAll are lower than Sm = %.3f MPa" %Stress_Intensity)
                output_lines.append("\n============================================================================================================================")
                output_lines.append("\nAccording to Mariotte:")
                output_lines.append("Maximum absolute value of the total radial stress: %.3f MPa\nMaximum absolute value of the total hoop stress: %.3f MPa\nMaximum absolute value of the total axial stress: %.3f MPa" %(max(abs(sigma_r_totM)),max(abs(sigma_t_totM)),max(abs(sigma_z_totM))))
                output_lines.append("All are lower than 3Sm = %.3f MPa" %(3*Stress_Intensity)) 
                output_lines.append("\nPrimary radial stress: %.3f MPa\nPrimary hoop stress: %.3f MPa\nPrimary axial stress: %.3f MPa" %(sigma_rM,sigma_tM,sigma_zM))
                output_lines.append("\nAll are lower than Sm = %.3f MPa" %Stress_Intensity)

            if (sigma_cTR_L_PO < Stress_Intensity):
                output_lines.append("\n============================================================================================================================")
                output_lines.append("\nThe comparison stress of primary stresses only according to Tresca-Lamé Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_L_PO, Stress_Intensity))
            else:
                output_lines.append("\n============================================================================================================================")
                output_lines.append("WARNING: The comparison stress of primary stresses only according to Tresca-Lamé Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_L_PO, Stress_Intensity))
                output_lines.append("============================================================================================================================")
            if (sigma_cTR_M_PO < Stress_Intensity):
                output_lines.append("The comparison stress of primary stresses only according to Tresca-Mariotte Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_M_PO, Stress_Intensity))
                output_lines.append("\n============================================================================================================================")
            else:
                output_lines.append("\n============================================================================================================================")
                output_lines.append("WARNING: The comparison stress of primary stresses only according to Tresca-Mariotte Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_M_PO, Stress_Intensity))
                output_lines.append("============================================================================================================================")
                
            if (sigma_cTR_L < 3*Stress_Intensity):
                output_lines.append("\n============================================================================================================================")
                output_lines.append("\nThe comparison stress according to Tresca-Lamé Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_L, 3*Stress_Intensity))
            else:
                output_lines.append("\n============================================================================================================================")
                output_lines.append("WARNING: The comparison stress according to Tresca-Lamé Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_L, 3*Stress_Intensity))
                output_lines.append("============================================================================================================================")
            if (sigma_cTR_M < 3*Stress_Intensity):
                output_lines.append("The comparison stress according to Tresca-Mariotte Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_M, 3*Stress_Intensity))
                output_lines.append("\n============================================================================================================================")
            else:
                output_lines.append("\n============================================================================================================================")
                output_lines.append("WARNING: The comparison stress according to Tresca-Mariotte Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_M, 3*Stress_Intensity))
                output_lines.append("============================================================================================================================")
                
            output_lines.append("\n\n\n\n######################################################### Buckling #########################################################")
            output_lines.append("============================================================================================================================")
            output_lines.append("\nAccording to the Corradi Design Procedure:")
            output_lines.append("The theoretical limit for collapse pressure, accounting for ovality, is: q_c = %.3f MPa = %.3f bar" %(Corradi_vessel[0], 10*Corradi_vessel[0]))
            output_lines.append("A safety factor s = %.3f was assumed. \nThe allowable external pressure is thus: q_a = %.3f MPa = %.3f bar" %(Corradi_vessel[2], Corradi_vessel[1], 10*Corradi_vessel[1]))
            
            if buckling_flag:
                output_lines.append("\nThe given external pressure of %.3f bar is lower than the allowable pressure of %.3f bar" %(P_cpp, 10*Corradi_vessel[1]))
                output_lines.append("\n============================================================================================================================")
            elif not buckling_flag:
                output_lines.append("\n============================================================================================================================")
                output_lines.append("WARNING: The given external pressure of %.3f bar is higher than the allowable pressure of %.3f bar: a change in thickness is required!" %(P_cpp, 10*Corradi_vessel[1]))
                output_lines.append("============================================================================================================================")

            if buckling_flag and sigma_cTR_L_PO < Stress_Intensity and sigma_cTR_M_PO < Stress_Intensity and sigma_cTR_L < 3*Stress_Intensity and sigma_cTR_M < 3*Stress_Intensity and not creep_flag_V:
                output_lines.append("\n\n\n\n############################################################################################################################")
                output_lines.append("The vessel's integrity is ensured: the design is correct!")
                output_lines.append("############################################################################################################################")

            for line in output_lines:                                   # Print messages to the console and write to the file
                #print(line)
                file.write(line + '\n')                                 # Add a newline for formatting in the text file
            shutil.move(NTS_plots_directory_path, case_directory_path)  # Move the plots directory into the case directory
            
            print("\n\n\033[32m############################################################################################################################\033[0m")
        print("\033[32mResults have been saved at: %s\033[0m" %case_directory_path)
        print("\033[32m############################################################################################################################\033[0m\n\n")
            
    # ======================================
    # Discretization along z
    # ======================================
    elif Disc_flag:
        
        # ======================================
        # T discretization along z
        # ======================================
        dz = 100
        T_z = np.linspace(T_in, T_out_avg, dz)
        while True:
            try:
                adiab_flag = int(input("\nApply Adiabatic Outer Wall approximation? (1: Yes, 0: No): "))
                if adiab_flag not in (0, 1):
                    raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                adiab_flag = bool(adiab_flag)
                break  
            except ValueError:
                print("\033[31mPlease enter a valid integer.\033[0m")
            except RuntimeError as e:
                print(e)

        if not adiab_flag:
            C1 = ((q_0/(k_st*mu_st**2))*(np.exp(-mu_st*t)-1)-(q_0/mu_st)*((1/h_1)+(np.exp(-mu_st*t)/u_2))-(T_z - T_cpp))/(t+(k_st/h_1)+(k_st/u_2))      # Make C1 a 1D array aligned with T_z (avoid wrapping in another dimension)
        elif adiab_flag:
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
        if not adiab_flag:
            
            # ======================================
            # Wall T(T_z, r) map
            # ======================================
            R_mesh, T_z_mesh = np.meshgrid(r, T_z - 273.15)                                                      #Shapes (Nz, Nr)
            os.makedirs(NTS_plots_directory_path, exist_ok=True)
            plot_file_path = os.path.join(NTS_plots_directory_path, "Temperature 2D Map and Profiles.png")
            plt.figure(figsize=(15,10))
            plt.subplot(1,2,1)
            pcm = plt.pcolormesh(R_mesh, T_z_mesh, T_vessel_Mat - 273.15, shading='auto', cmap='hot')   #Or 'hot','plasma','viridis'
            plt.colorbar(pcm, label='T (°C)')
            plt.xlabel('Radius (m)')
            plt.ylabel('T$_z$ (°C)')
            plt.title('Wall Temperature Map (r vs T$_z$)')
            
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
            plt.tight_layout()
            plt.savefig(plot_file_path)
            plt.show()
            plt.close()
            
        elif adiab_flag:
            
            # ======================================
            # Wall T(T_z, r) map
            # ======================================
            R_mesh, T_z_mesh = np.meshgrid(r, T_z - 273.15)
            os.makedirs(NTS_plots_directory_path, exist_ok=True)
            plot_file_path = os.path.join(NTS_plots_directory_path, "Temperature 2D Map and Profiles under AOW Approximation.png")
            plt.figure(figsize=(15,10))
            plt.subplot(1,2,1)
            pcm = plt.pcolormesh(R_mesh, T_z_mesh, T_vessel_Mat - 273.15, shading='auto', cmap='hot')
            plt.colorbar(pcm, label='T (°C)')
            plt.xlabel('Radius (m)')
            plt.ylabel('T$_z$ (°C)')
            plt.title('Wall Temperature Map under AOW Approximation (r vs T$_z$)')
            
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
            plt.tight_layout()
            plt.savefig(plot_file_path)
            plt.show()
            plt.close()
        
        # ======================================
        # Plotting the thermal stress profiles
        # ======================================
        R_mesh, T_z_mesh = np.meshgrid(r, T_z - 273.15)   # shapes (Nz, Nr)
        plot_file_path = os.path.join(NTS_plots_directory_path, "Thermal Stresses 2D Map.png")
        plt.figure(figsize=(20,20))
        plt.subplot(1,4,1)
        pcm = plt.pcolormesh(R_mesh, T_z_mesh, sigma_r_th, shading='auto', cmap='viridis')
        plt.colorbar(pcm, label=r'$\sigma$ (MPa)')
        plt.xlabel('Radius (m)')
        plt.ylabel('T$_z$ (°C)')
        plt.title('Radial Stress Map (r vs T$_z$)')

        plt.subplot(1,4,2)
        pcm = plt.pcolormesh(R_mesh, T_z_mesh, sigma_t_th, shading='auto', cmap='viridis')
        plt.colorbar(pcm, label=r'$\sigma$ (MPa)')
        plt.xlabel('Radius (m)')
        plt.ylabel('T$_z$ (°C)')
        plt.title('Hoop Stress Map (r vs T$_z$)')

        plt.subplot(1,4,3)
        pcm = plt.pcolormesh(R_mesh, T_z_mesh, sigma_t_th_SIMP, shading='auto', cmap='viridis')
        plt.colorbar(pcm, label=r'$\sigma$ (MPa)')
        plt.xlabel('Radius (m)')
        plt.ylabel('T$_z$ (°C)')
        plt.title('Simplified Hoop Stress Map (r vs T$_z$)')

        plt.subplot(1,4,4)
        pcm = plt.pcolormesh(R_mesh, T_z_mesh, sigma_z_th, shading='auto', cmap='viridis')
        plt.colorbar(pcm, label=r'$\sigma$ (MPa)')
        plt.xlabel('Radius (m)')
        plt.ylabel('T$_z$ (°C)')
        plt.title('Axial Stress Map (r vs T$_z$)')
        plt.tight_layout()
        plt.savefig(plot_file_path)
        plt.show()
        plt.close()

# =============================================================================================================================================================
# THERMOMECHANICAL PROBLEM - POWER IMPOSED - THERMAL SHIELD
# =============================================================================================================================================================
elif TS_flag:
    print("\n\033[33m############################################################################################################################\033[0m")
    print("\033[33mTHERMOMECHANICAL PROBLEM - POWER IMPOSED - THERMAL SHIELD\033[0m")
    print("\033[33m############################################################################################################################\033[0m")

    # =============================================================================================================================================================
    # PURELY MECHANICAL PROBLEM
    # =============================================================================================================================================================
    print("\n\033[34m############################################################################################################################\033[0m")
    print("\033[34mPURELY MECHANICAL PROBLEM\033[0m")
    print("\033[34m############################################################################################################################\033[0m")
    t = 0.05                                    #m
    R_ext = R_int + t                           #m
    D_vess_ext = 2*R_ext                        #m
    rho_ii = (R_ext**2)/(R_ext**2 - R_int**2)
    rho_i = (R_int**2)/(R_ext**2 - R_int**2)
    Mar_criterion = R_int/t
    W = (DeltaD_max/1000)/((D_vess_int+D_vess_ext)/2)
    
    dr = 100
    r = np.linspace(R_int, R_ext, dr)

    # ============================
    # Mariotte Solution for a thin-walled cylinder (R_int = R_ext = R)
    # ============================
    def sigmaM_func (R_int, P_int_MPa, t): 
        sigma_rM_cyl_in = -P_int_MPa/2                        #Compressive
        sigma_tM_cyl_in = R_int*P_int_MPa/t
        sigma_zM_cyl_in = R_int*P_int_MPa/(2*t)

        sigma_rM_cyl_out = -P_cpp_MPa/2
        sigma_tM_cyl_out = -R_int*P_cpp_MPa/t                 #sigma_tM_sph = R_int*P_int_MPa/(2*t)
        sigma_zM_cyl_out = -R_int*P_cpp_MPa/(2*t)
        return (sigma_rM_cyl_in+sigma_rM_cyl_out, sigma_tM_cyl_in+sigma_tM_cyl_out, sigma_zM_cyl_in+sigma_zM_cyl_out)

    if Mar_criterion > 5:
        while True:
            try:
                Mariotte_flag = int(input("\nWith an initial thickness value of %.3f m, the vessel can be considered thin. Do you want to visualize the Mariotte solution for stress? (1: Yes, 0: No): " %t))
                if Mariotte_flag not in (0, 1):
                    raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                Mariotte_flag = bool(Mariotte_flag)
                break  
            except ValueError:
                print("\033[31mPlease enter a valid integer.\033[0m")
            except RuntimeError as e:
                print(e)
        sigma_M = sigmaM_func(R_int, P_int_MPa, t)
        sigma_rM = sigma_M[0]
        sigma_tM = sigma_M[1]
        sigma_zM = sigma_M[2]

        # ======================================
        # Plotting the stress profiles: Mariotte
        # ======================================
        os.makedirs(TS_plots_directory_path, exist_ok=True)
        plot_file_path = os.path.join(TS_plots_directory_path, "Stress Distribution in a thin-walled cylinder - Mariotte Solution.png")
        plt.figure(figsize=(15,10))
        plt.axvline(x = R_int, color='black', linewidth='3', label='Vessel Inner Surface')
        plt.axvline(x = R_ext, color='black', linewidth='3', label='Vessel Outer Surface')
        plt.axhline(y = sigma_rM, color='red', label='Radial (r) Stress Mariotte')
        plt.axhline(y = sigma_tM, color='blue', label=r'Hoop ($\theta$) Stress Mariotte')
        plt.axhline(y = sigma_zM, color='green', label='Axial (z) Stress Mariotte')
        plt.plot(r, np.zeros(len(r)), color='black', linewidth='1', label='y=0')
        plt.xlabel('Radius (m)')
        plt.ylabel('Stress (MPa)')
        plt.title('Stress Distribution in a thin-walled cylinder - Mariotte Solution')
        plt.legend()
        plt.grid()
        plt.savefig(plot_file_path)
        if Mariotte_flag:
            plt.show()
            plt.close()
        elif not Mariotte_flag:
            plt.close()

    else:
        print("\n\033[34mThe cylinder can't be considered thin. Skipping Mariotte solution.\033[0m")
        Mariotte_flag = bool(0)

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
                print("\n\033[34mInternal and external pressures are equal: hydrostatic stress condition is verified. Skipping.\033[0m")    #Hydrostatic Stress Condition
            eps_z_a = (2*nu-1)*rho_ii*P_cpp_MPa/E
            eps_z_b = (1-2*nu)*rho_i*P_int_MPa/E

        elif P_int != P_cpp:
            if eps_choice == 1:                                                                                           #Plane Stress
                eps_z_a = 2*nu*rho_ii*P_cpp_MPa/E
                eps_z_b = -2*nu*rho_i*P_int_MPa/E
            elif eps_choice == 0:                                                                                         #Plane Strain
                eps_z_a = 0
                eps_z_b = 0 

        sigma_zL_a = E*eps_z_a - 2*nu*rho_ii*P_cpp_MPa  #a) P_int = 0
        sigma_zL_b = E*eps_z_b + 2*nu*rho_i*P_int_MPa   #b) P_cpp = 0
        return (sigma_rL(r), sigma_tL(r), sigma_zL_a + sigma_zL_b)              #Superposition Principle

    sigma_L = sigmaL_func(r, P_int_MPa, P_cpp_MPa, 1)
    sigma_rL = sigma_L[0]  
    sigma_tL = sigma_L[1]
    sigma_zL = sigma_L[2]

    while True:
        try:
            Lame_flag = int(input("\nDo you want to visualize the Lamé solution? (1: Yes, 0: No): "))
            if Lame_flag not in (0, 1):
                raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
            Lame_flag = bool(Lame_flag)
            break  
        except ValueError:
            print("\033[31mPlease enter a valid integer.\033[0m")
        except RuntimeError as e:
            print(e)

    # ======================================
    # Plotting the stress profiles: Lamé
    # ======================================
    plot_file_path = os.path.join(TS_plots_directory_path, "Stress Distribution in the cylinder wall - Lamé Solution.png")
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
    plt.savefig(plot_file_path)
    if Lame_flag:
        plt.show()
        plt.close()
    elif not Lame_flag:
        plt.close()
    
    # =============================================================================================================================================================
    # PURELY THERMAL PROBLEM
    # =============================================================================================================================================================
    print("\n\033[34m############################################################################################################################\033[0m")
    print("\033[34mPURELY THERMAL PROBLEM\033[0m")
    print("\033[34m############################################################################################################################\033[0m")
    t_shield_user = 0.001           #Initial guess for the thermal shield thickness

    while True:
        try:
            user_D_choice = float(input("\nWhat position of the thermal shield do you want to consider? (3: Arbitrary, 2: Middle, 1: Equal areas, 0: Equal h_1): "))
            if user_D_choice not in (0, 1, 2, 3):
                raise RuntimeError("Invalid input! Please enter either 0, 1, 2 or 3.")
            break  
        except ValueError:
            print("\033[31mPlease enter a valid integer.\033[0m")
        except RuntimeError as e:
            print(e)

    if user_D_choice == 3:            #Allows the user to assume the default, middle position for the thermal shield, position it himself or choose the position granting equal areas
        while True:
            try:
                D_shield_int = float(input("\nPlease enter the initial thermal shield inner diameter (m) to choose its position: "))
                if (D_shield_int < D_barr_ext) or (D_shield_int/2 + t_shield_user) > D_vess_int/2:
                    raise RuntimeError("\033[31mThe thermal shield is either starting inside the barrel or clipping inside the vessel.\033[0m")
                break  
            except ValueError:
                print("\033[31mPlease enter a valid float.\033[0m")
            except RuntimeError as e:
                print(e)
        R_shield_int = D_shield_int/2
        
    elif user_D_choice == 2:
        print("\033[34mAssuming middle thermal shield position.\033[0m")
        R_shield_int = D_barr_ext/2 + (D_vess_int - D_barr_ext)/4 - t_shield_user/2
         
    elif user_D_choice == 1:
        A_eq = 1
        B_eq = t_shield_user
        C_eq = (t_shield_user**2)/2 - (R_int**2)/2 - (R_barr_ext**2)/2
        Delta_eq = B_eq**2 - 4*A_eq*C_eq
        R_shield_int = (-B_eq + np.sqrt(Delta_eq))/(2*A_eq)
        D_shield_int = 2*R_shield_int
    
    # =============================================================================================================================================================
    # Thermal Shield Initial Position Iterative Computation to equalize h1
    # =============================================================================================================================================================
    elif user_D_choice == 0:
        counter_h1 = 0
        N_max_h1 = 1000
        eps = 1e-8
        
        R_shield_int = R_barr_ext + 0.001 #To avoid float division by zero
        D_shield_int = 2*R_shield_int
        R_shield_ext = R_shield_int + t_shield_user
        D_shield_ext = 2*R_shield_ext
        R_ext = R_int + t
        D_vess_ext = 2*R_ext
        
        #Dummy entries to enter the loop
        h_1_int = 0 
        h_1_ext = 1
        
        #Bisection method bounds
        l_bound = R_barr_ext
        r_bound = R_int - t_shield_user
        
        while abs(h_1_int - h_1_ext) > eps:
            counter_h1 += 1
            print("Iteration no. %d" %counter_h1)
            if counter_h1 > N_max_h1:
                print("\033[31mExceeded maximum number of iterations: %d. Exiting the loop.\033[0m" %N_max_h1)
                break
            if R_shield_int > R_int - t_shield_user:
                print("\033[31mInner thermal shield radius violates geometric constraints. Exiting the loop.\033[0m")
                break
            
            D_shield_int = 2*R_shield_int
            D_shield_ext = 2*R_shield_ext
            
            A_int_S = np.pi*((R_shield_int**2) - (R_barr_ext**2))                                       #Inner area crossed by the primary fluid
            A_ext_S = np.pi*((R_int**2) - (R_shield_ext**2))                                            #Outer area crossed by the primary fluid
            v_int = v_flr/A_int_S                                                                       #Inner coolant velocity
            v_ext = v_flr/A_ext_S                                                                       #Outer coolant velocity
            
            Pr = (Cp*mu)/k                                                                              #Prandtl number
            
            Re_int = (rho*v_int*(D_shield_int - D_barr_ext))/mu                                         #Inner hydraulic diameter                                                     
            Nu_1_int = 0.023*(Re_int**0.8)*(Pr**0.4)                                                             
            h_1_int = (Nu_1_int*k)/(D_shield_int - D_barr_ext)
            
            Re_ext = (rho*v_ext*(D_vess_int - D_shield_ext))/mu                                         #Outer hydraulic diameter                                                     
            Nu_1_ext = 0.023*(Re_ext**0.8)*(Pr**0.4)                                                             
            h_1_ext = (Nu_1_ext*k)/(D_vess_int - D_shield_ext)
            
            # Bisection method
            if h_1_int < h_1_ext:
                r_bound = R_shield_int
                R_shield_int = (r_bound + l_bound)/2
                R_shield_ext = R_shield_int + t_shield_user
            else:
                l_bound = R_shield_int
                R_shield_int = (l_bound + r_bound)/2
                R_shield_ext = R_shield_int + t_shield_user
        
        print("\n\033[32m############################################################################################################################\033[0m")
        print("\033[32mInitial heat transfer coefficients equalized in %d iterations. Thermal shield initial inner radius: %.3f m\033[0m" %(counter_h1, R_shield_int))
        print("\033[32m############################################################################################################################\033[0m\n")

    print("\033[34mNo discretization along z. Assuming constant temperature of the primary fluid T1.\033[0m")
    while True:
        try:
            T1_choice = int(input("\nWhat temperature do you want to use as T1 to compute C1 and C2? (0: T_in, 1: T_in + 10%, 2: T_in + 20%, 3: T_avg, 4: T_out_avg): "))
            if T1_choice not in (0, 1, 2, 3, 4):
                raise RuntimeError("\033[31mInvalid input! Please enter one of the allowed values: 1, 2, 3, 4.\033[0m")
            break  
        except ValueError:
            print("\033[31mPlease enter a valid integer.\033[0m")
        except RuntimeError as e:
            print(e)

    if T1_choice == 0:                #All these temperatures are expressed in K. T_out_max and T_avg_log have been discarded in favor of margins on T_in, to account for transients due to the system's geometry
        print("\033[34mT1 = T_in has been assumed as the constant temperature of the primary fluid.\033[0m")
        T1 = T_in
    elif T1_choice == 1:
        print("\033[34mT1 = T_in + 10%% has been assumed as the constant temperature of the primary fluid.\033[0m")
        T1 = T_in * 1.1
    elif T1_choice == 2:
        print("\033[34mT1 = T_in + 20%% has been assumed as the constant temperature of the primary fluid.\033[0m")
        T1 = T_in * 1.2
    elif T1_choice == 3:
        print("\033[34mT1 = T_avg has been assumed as the constant temperature of the primary fluid.\033[0m")
        T1 = ((T_in + T_out_avg)/2)
    elif T1_choice == 4:
        print("\033[34mT1 = T_out_avg has been assumed as the constant temperature of the primary fluid.\033[0m")
        T1 = T_out_avg

    while True:
        try:
            adiab_flag = int(input("\nApply Adiabatic Outer Wall approximation? (1: Yes, 0: No): "))
            if adiab_flag not in (0, 1):
                raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
            adiab_flag = bool(adiab_flag)
            break  
        except ValueError:
            print("\033[31mPlease enter a valid integer.\033[0m")
        except RuntimeError as e:
            print(e)
    if not adiab_flag:
        print("\033[34mThe Adiabatic Outer Wall approximation has not been applied.\033[0m")
    elif adiab_flag:
        print("\033[34mThe Adiabatic Outer Wall approximation has been applied.\033[0m")

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
            s = float(input("\nPlease enter a safety factor between 1.5 and 2 for the Corradi design procedure: "))
            if s < 1.5 or s > 2:
                raise RuntimeError("\033[31mInvalid input! Please enter a safety factor between 1.5 and 2.\033[0m")
            break  
        except ValueError:
            print("\033[31mPlease enter a valid float.\033[0m")
        except RuntimeError as e:
            print(e)

    while not final_flag:
        t_shield += 0.001
        counter += 1
        print("Iteration no. %d" %counter)
        if counter > N_max:
            print("\033[31mExceeded maximum number of iterations: %d. Exiting the loop.\033[0m" %N_max)
            break
        if t > t_vessel_max:
            print("\033[31mVessel thickness exceeds feasibility margin. Exiting the loop.\033[0m")
            break
        if user_D_choice == 3 or user_D_choice == 1:        #If the thermal shield is not in the middle, geometrical constraints are present: the thermal shield must not bump into the vessel
            if t_shield > t_shield_max or (D_shield_int/2 + t_shield) > D_vess_int/2:
                print("\033[34mRan into excessive thermal shield thickness or bumped into the vessel. Adding 1cm to the vessel thickness instead. Restarting...\033[0m")
                t += 0.01
                t_shield = t_shield_user - 0.001
                counter_vessel += 1
                continue
        elif user_D_choice == 2 or user_D_choice == 0:      #If the thermal shield is in the middle, only its thickness must be checked: no geometrical constraints.The same goes for the equal h1 case, as the geometrical constraints are verified elsewhere.
            if t_shield > t_shield_max:
                print("\033[34mRan into excessive thermal shield thickness. Adding 1cm to the vessel thickness instead. Restarting...\033[0m")
                t += 0.01
                t_shield = t_shield_user - 0.001
                counter_vessel += 1
                continue

        if user_D_choice == 3:
            R_shield_int = D_shield_int/2
            
        elif user_D_choice == 2:
            R_shield_int = D_barr_ext/2 + (D_vess_int - D_barr_ext)/4 - t_shield/2
            D_shield_int = 2*R_shield_int
            
        elif user_D_choice == 1:
            A_eq = 1
            B_eq = t_shield
            C_eq = (t_shield**2)/2 - (R_int**2)/2 - (R_barr_ext**2)/2
            Delta_eq = B_eq**2 - 4*A_eq*C_eq
            R_shield_int = (-B_eq + np.sqrt(Delta_eq))/(2*A_eq)
            D_shield_int = 2*R_shield_int
            
    # =============================================================================================================================================================
    # Thermal Shield Position Iterative Computation
    # =============================================================================================================================================================
        elif user_D_choice == 0:
            counter_h1 = 0
            N_max_h1 = 10000
            eps = 1e-6
            
            R_shield_int = R_barr_ext + 0.001 #To avoid float division by zero
            D_shield_int = 2*R_shield_int
            R_shield_ext = R_shield_int + t_shield
            D_shield_ext = 2*R_shield_ext
            R_ext = R_int + t
            D_vess_ext = 2*R_ext
            
            #Dummy entries to enter the loop
            h_1_int = 0 
            h_1_ext = 1
            
            #Bisection method bounds
            l_bound = R_barr_ext
            r_bound = R_int - t_shield
            
            while abs(h_1_int - h_1_ext) > eps:
                counter_h1 += 1
                #print("Sub-iteration no. %d" %counter_h1)
                if counter_h1 > N_max_h1:
                    print("\033[31mExceeded maximum number of sub-iterations: %d. Exiting the loop.\033[0m" %N_max_h1)
                    break
                if R_shield_int > R_int - t_shield:
                    print("\033[31mInner thermal shield radius violates geometric constraints. Exiting the loop.\033[0m")
                    break
                
                D_shield_int = 2*R_shield_int
                D_shield_ext = 2*R_shield_ext
                
                A_int_S = np.pi*((R_shield_int**2) - (R_barr_ext**2))                                       #Inner area crossed by the primary fluid
                A_ext_S = np.pi*((R_int**2) - (R_shield_ext**2))                                            #Outer area crossed by the primary fluid
                v_int = v_flr/A_int_S                                                                       #Inner coolant velocity
                v_ext = v_flr/A_ext_S                                                                       #Outer coolant velocity
                
                Pr = (Cp*mu)/k                                                                              #Prandtl number
                
                Re_int = (rho*v_int*(D_shield_int - D_barr_ext))/mu                                         #Inner hydraulic diameter                                                     
                Nu_1_int = 0.023*(Re_int**0.8)*(Pr**0.4)                                                             
                h_1_int = (Nu_1_int*k)/(D_shield_int - D_barr_ext)
                
                Re_ext = (rho*v_ext*(D_vess_int - D_shield_ext))/mu                                         #Outer hydraulic diameter                                                     
                Nu_1_ext = 0.023*(Re_ext**0.8)*(Pr**0.4)                                                             
                h_1_ext = (Nu_1_ext*k)/(D_vess_int - D_shield_ext)
                
                # Bisection method
                if h_1_int < h_1_ext:
                    r_bound = R_shield_int
                    R_shield_int = (r_bound + l_bound)/2
                    R_shield_ext = R_shield_int + t_shield
                else:
                    l_bound = R_shield_int
                    R_shield_int = (l_bound + r_bound)/2
                    R_shield_ext = R_shield_int + t_shield
        
        R_shield_ext = R_shield_int + t_shield
        D_shield_ext = 2*R_shield_ext
        
        R_ext = R_int + t                           #m   -   Must be updated at every change of t
        D_vess_ext = 2*R_ext                        #m
        W = (DeltaD_max/1000)/((D_vess_int+D_vess_ext)/2)  #The denominator is the average diameter of the vessel wall  
                                                           #No need to compute these for the thermal shield, which is not subject to buckling
                
        r_S = np.linspace(R_shield_int, R_shield_ext, dr)
        Phi_0S = Phi_0                                                #All gamma rays reach the shield, not the vessel

        # ======================================
        # Dimensionless numbers and heat transfer coefficients
        # ======================================
        if user_D_choice != 0:
            A_int_S = np.pi*((R_shield_int**2) - (R_barr_ext**2))                                       #Inner area crossed by the primary fluid
            A_ext_S = np.pi*((R_int**2) - (R_shield_ext**2))                                            #Outer area crossed by the primary fluid
            v_int = v_flr/A_int_S                                                                       #Inner coolant velocity
            v_ext = v_flr/A_ext_S                                                                       #Outer coolant velocity
            
            Pr = (Cp*mu)/k                                                                              #Prandtl number
            
            Re_int = (rho*v_int*(D_shield_int - D_barr_ext))/mu                                         #Inner hydraulic diameter                                                     
            Nu_1_int = 0.023*(Re_int**0.8)*(Pr**0.4)                                                             
            h_1_int = (Nu_1_int*k)/(D_shield_int - D_barr_ext)
            
            Re_ext = (rho*v_ext*(D_vess_int - D_shield_ext))/mu                                         #Outer hydraulic diameter                                                     
            Nu_1_ext = 0.023*(Re_ext**0.8)*(Pr**0.4)                                                             
            h_1_ext = (Nu_1_ext*k)/(D_vess_int - D_shield_ext)
            
        Pr_cpp = (Cp_cpp*mu_cpp)/k_cpp                                                              #Prandtl number of the containment water                                 
        Gr = (rho_cpp**2)*9.81*beta_cpp*DeltaT*(L**3)/(mu_cpp**2)                                   #Grashof number (Uses the external diameter as characteristic length, might wanna use L though?)
        Nu_2 = 0.13*((Gr*Pr_cpp)**(1/3))                                                            #McAdams correlation for natural convection
        h_2 = (Nu_2*k_cpp)/L                                                                        #W/(m²·K)
        R_th_2_tot = (1/(2*np.pi*(R_ext + t_th_ins)*L)) * ((((R_ext + t_th_ins)/k_th_ins)*np.log((R_ext + t_th_ins)/R_ext)) + (1/h_2))                          #Thermal Resistance of the insulation layer + natural convection outside the vessel
        u_2 = 1/(2*np.pi*(R_ext + t_th_ins)*L*R_th_2_tot)                                           #W/(m²·K)   -   Overall heat transfer coefficient outside the vessel

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
        # Convective heat transfer coefficient choice
        # ======================================
        if user_D_choice == 3 or user_D_choice == 2:
            h_1 = min(h_1_int, h_1_ext)                 #Conservative: minimum h means highest thermal stresses
        elif user_D_choice == 1:
            h_1 = min(h_1_int, h_1_ext)
            if abs(h_1_int - h_1_ext) <= eps:
                h_1 = h_1_int                           #Below the tolerance, they can be considered equal
        elif user_D_choice == 0:
            h_1 = h_1_int                               #By construction, h_1_int = h_1_ext = h_1
        
        # ======================================
        # T profile constants for the vessel: general and under adiabatic outer wall approximation (dT/dx = 0 at r = R_ext)
        # ======================================    
        if not adiab_flag:
            C1 = ((q_0/(k_st*mu_st**2))*(np.exp(-mu_st*t)-1)-(q_0/mu_st)*((1/h_1)+(np.exp(-mu_st*t)/u_2))-(T1-T_cpp))/(t+(k_st/h_1)+(k_st/u_2))
        elif adiab_flag:
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
        # Mechanical Stresses and Principal stresses in the vessel
        # ======================================
        sigma_L = sigmaL_func(r, P_int_MPa, P_cpp_MPa, 0)
        sigma_rL = sigma_L[0]  
        sigma_tL = sigma_L[1]
        sigma_zL = sigma_L[2]
        
        sigma_M = sigmaM_func(R_int, P_int_MPa, t)
        sigma_rM = sigma_M[0]
        sigma_tM = sigma_M[1]
        sigma_zM = sigma_M[2]
        
        sigma_r_totL = sigma_rL + sigma_r_th_V
        sigma_t_totL = sigma_tL + sigma_t_th_V
        sigma_z_totL = sigma_zL + sigma_z_th_V
        
        sigma_r_totM = sigma_rM + sigma_r_th_V
        sigma_t_totM = sigma_tM + sigma_t_th_V
        sigma_z_totM = sigma_zM + sigma_z_th_V
        
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
        # Vessel Comparison stress - Guest-Tresca Theory - Mariotte/Lamé only
        # ============================
        sigma_cTR_M_PO = np.max([abs(sigma_tM - sigma_rM), abs(sigma_zM - sigma_rM), abs(sigma_tM - sigma_zM)])
        sigma_cTR_L_PO = []
        for i in range(len(r)):
            sigma_cTR_L_PO.append(np.max([abs(sigma_tL - sigma_rL)[i], abs(sigma_zL - sigma_rL)[i], abs(sigma_tL - sigma_zL)[i]]))
        sigma_cTR_L_PO = max(sigma_cTR_L_PO)

        # ============================ 
        # Vessel Comparison stress - Guest-Tresca Theory - Mariotte/Lamé + Thermal stresses
        # ============================
        sigma_cTR_M = []
        for i in range(len(r)):
            sigma_cTR_M.append(np.max([abs(sigma_t_totM - sigma_r_totM)[i], abs(sigma_z_totM - sigma_r_totM)[i], abs(sigma_t_totM - sigma_z_totM)[i]]))
        sigma_cTR_M = max(sigma_cTR_M)
        sigma_cTR_L = []
        for i in range(len(r)):
            sigma_cTR_L.append(np.max([abs(sigma_t_totL - sigma_r_totL)[i], abs(sigma_z_totL - sigma_r_totL)[i], abs(sigma_t_totL - sigma_z_totL)[i]]))
        sigma_cTR_L = max(sigma_cTR_L)

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
        # Mechanical Stresses and Principal Stresses in the thermal shield 
        # ======================================
        sigma_L_S = sigmaL_func(r_S, P_int_MPa, P_int_MPa, 0)
        sigma_rL_S = sigma_L_S[0]  
        sigma_tL_S = sigma_L_S[1]
        sigma_zL_S = sigma_L_S[2]
        
        sigma_M_S = sigmaM_func(R_shield_int, P_int_MPa, t_shield)
        sigma_rM_S = sigma_M_S[0]
        sigma_tM_S = sigma_M_S[1]
        sigma_zM_S = sigma_M_S[2]

        sigma_r_totL_S = sigma_rL_S + sigma_r_th_S
        sigma_t_totL_S = sigma_tL_S + sigma_t_th_S
        sigma_z_totL_S = sigma_zL_S + sigma_z_th_S
        
        sigma_r_totM_S = sigma_rM_S + sigma_r_th_S
        sigma_t_totM_S = sigma_tM_S + sigma_t_th_S
        sigma_z_totM_S = sigma_zM_S + sigma_z_th_S

        # ============================ 
        # Thermal Shield Comparison stress - Guest-Tresca Theory - Mariotte/Lamé only
        # ============================
        sigma_cTR_MS_PO = np.max([abs(sigma_tM_S - sigma_rM_S), abs(sigma_zM_S - sigma_rM_S), abs(sigma_tM_S - sigma_zM_S)])
        sigma_cTR_LS_PO = []
        for i in range(len(r_S)):
            sigma_cTR_LS_PO.append(np.max([abs(sigma_tL_S - sigma_rL_S)[i], abs(sigma_zL_S - sigma_rL_S)[i], abs(sigma_tL_S - sigma_zL_S)[i]]))
        sigma_cTR_LS_PO = max(sigma_cTR_LS_PO)

        # ============================ 
        # Thermal Shield Comparison stress - Guest-Tresca Theory - Mariotte/Lamé + Thermal stresses
        # ============================
        sigma_cTR_MS = []
        for i in range(len(r_S)):
            sigma_cTR_MS.append(np.max([abs(sigma_t_totM_S - sigma_r_totM_S)[i], abs(sigma_z_totM_S - sigma_r_totM_S)[i], abs(sigma_t_totM_S - sigma_z_totM_S)[i]]))
        sigma_cTR_MS = max(sigma_cTR_MS)
        sigma_cTR_LS = []
        for i in range(len(r_S)):
            sigma_cTR_LS.append(np.max([abs(sigma_t_totL_S - sigma_r_totL_S)[i], abs(sigma_z_totL_S - sigma_r_totL_S)[i], abs(sigma_t_totL_S - sigma_z_totL_S)[i]]))
        sigma_cTR_LS = max(sigma_cTR_LS)

        # ============================ 
        # Yield Stress and Stress Intensity Data Interpolation
        # ============================
        T_des_vessel = T_vessel_avg                                                     #K  -   Check in the HARVEY/Thermomechanics Chapter how to choose the design T
        T_des_vessel_C = 270 #T_des_vessel - 273.15                                          #°C
        T_des_shield = T_shield_avg                                                     #K
        T_des_shield_C = 270 #T_des_shield - 273.15                                          #°C

        Yield_stress = Yield_CubicSpline(T_des_vessel_C)
        Stress_Intensity = Intensity_CubicSpline(T_des_vessel_C)
        sigma_allowable = Stress_Intensity                                              #MPa
    
        Yield_stress_S = Yield_CubicSpline(T_des_shield_C)
        Stress_Intensity_S = Intensity_CubicSpline(T_des_shield_C)
        sigma_allowable_S = Stress_Intensity_S                                          #MPa
        """
        # ======================================
        # Thermal Shield Thermomechanical Integrity Verification    -   Lamé + Thermal stresses
        # ======================================
        if max(abs(sigma_r_totL_S)) > 3*Stress_Intensity_S or max(abs(sigma_t_totL_S)) > 3*Stress_Intensity_S or max(abs(sigma_z_totL_S)) > 3*Stress_Intensity_S:
            flag_primsec_S = 1
        else:
            flag_primsec_S = 0

        if max(abs(sigma_rL_S)) > Stress_Intensity_S or max(abs(sigma_tL_S)) > Stress_Intensity_S or sigma_zL_S > Stress_Intensity_S:
            flag_prim_S = 1
        else:
            flag_prim_S = 0

        # ======================================
        # Thermal Shield Thermomechanical Integrity Verification    -   Mariotte + Thermal stresses
        # ======================================
        if max(abs(sigma_r_totM_S)) > 3*Stress_Intensity_S or max(abs(sigma_t_totM_S)) > 3*Stress_Intensity_S or max(abs(sigma_z_totM_S)) > 3*Stress_Intensity_S:
            flag_primsec_S = bool(1)
        else:
            flag_primsec_S = bool(0)

        if sigma_rM_S > Stress_Intensity_S or sigma_tM_S > Stress_Intensity_S or sigma_zM_S > Stress_Intensity_S:
            flag_prim_S = bool(1)
        else:
            flag_prim_S = bool(0)
        """
        # ======================================
        # Thermal Shield Thermomechanical Integrity Verification    -   Tresca-Mariotte + Thermal stresses
        # ======================================
        if sigma_cTR_MS > 3*Stress_Intensity_S:
            flag_primsec_S = bool(1)
        else:
            flag_primsec_S = bool(0)

        if sigma_cTR_MS_PO > Stress_Intensity_S:
            flag_prim_S = bool(1)
        else:
            flag_prim_S = bool(0)
            
        if flag_primsec_S or flag_prim_S:
            continue
        elif not flag_primsec_S and not flag_prim_S:
            Corradi_flag = bool(1)                                                #Only enters the Corradi procedure if the thermal shield is ok

        # ============================ 
        # Corradi Design Procedure
        # ============================
        q_E_fun = lambda Dt: 2 * (E/(1-(nu**2))) * (1/(Dt*((Dt-1)**2)))     #Elastic Instability Limit for Thick Tubes
        q_0_fun = lambda Dt: 2 * Yield_stress * 1/Dt * (1+(1/(2*Dt)))       #Plastic Collapse Limit for Thick Tubes
        Dt_Crit_Ratio = np.sqrt(E/(Yield_stress*(1-(nu**2))))
        Current_Slenderness = (D_vess_int+2*t)/t

        if Corradi_flag:
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
            
            # ======================================
            # Buckling Verification
            # ======================================
            Corradi_vessel = Corradi(np.array([Current_Slenderness]))
            if (P_cpp < 10*Corradi_vessel[1]):
                buckling_flag = bool(1)
            else:
                buckling_flag = bool(0)
        """
        # ======================================
        # Vessel Thermomechanical Integrity Verification    -   Lamé + Thermal stresses
        # ======================================
        if max(abs(sigma_r_totL)) > 3*Stress_Intensity or max(abs(sigma_t_totL)) > 3*Stress_Intensity or max(abs(sigma_z_totL)) > 3*Stress_Intensity:
            flag_primsec = 1
        else:
            flag_primsec = 0

        if max(abs(sigma_rL)) > Stress_Intensity or max(abs(sigma_tL)) > Stress_Intensity or sigma_zL > Stress_Intensity:
            flag_prim = 1
        else:
            flag_prim = 0

        # ======================================
        # Vessel Thermomechanical Integrity Verification    -   Mariotte + Thermal stresses
        # ======================================
        if max(abs(sigma_r_totM)) > 3*Stress_Intensity or max(abs(sigma_t_totM)) > 3*Stress_Intensity or max(abs(sigma_z_totM)) > 3*Stress_Intensity:
            flag_primsec = bool(1)
        else:
            flag_primsec = bool(0)

        if sigma_rM > Stress_Intensity or sigma_tM > Stress_Intensity or sigma_zM > Stress_Intensity:
            flag_prim = bool(1)
        else:
            flag_prim = bool(0)
        """
        # ======================================
        # Vessel Thermomechanical Integrity Verification    -   Tresca-Mariotte + Thermal stresses
        # ======================================
        if sigma_cTR_M > 3*Stress_Intensity:
            flag_primsec = bool(1)
        else:
            flag_primsec = bool(0)

        if sigma_cTR_M_PO > Stress_Intensity:
            flag_prim = bool(1)
        else:
            flag_prim = bool(0)
        
        if flag_primsec or flag_prim:
            continue
        elif not flag_primsec and not flag_prim:
            vessel_flag = bool(1)
        
        # ======================================
        # Final Verification: buckling + inner P requirement + vessel stress state to exit the loop
        # ======================================
        t_min = (P_int_MPa*R_int)/(Stress_Intensity - 0.5*P_int_MPa)
        #t_min_S = (P_int_MPa * R_shield_int)/(Stress_Intensity_S - 0.5*P_int_MPa)
        
        if buckling_flag and vessel_flag:        
            if t >= t_min: #and t_shield >= t_min_S:    -     The thermal shield is always in hydrostatic conditions: there's no need to check for its minimum thickness required to sustain a P_int
                final_flag = bool(1)                          
            else:
                final_flag = bool(0)                          # ======================================
        else:                                                 # Tested, but the Tresca-Mariotte comparison stress is never lower than the allowable stress intensity Sm
            final_flag = bool(0)                              # ======================================
    
    # ======================================
    # Plotting the volumetric heat source profiles 
    # ======================================
    while True:
        try:
            hs_flag = int(input("\nDo you want to visualize the volumetric heat source q0 inside the vessel's wall and in the thermal shield? (1: Yes, 0: No): "))
            if hs_flag not in (0, 1):
                raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
            hs_flag = bool(hs_flag)
            break  
        except ValueError:
            print("\033[31mPlease enter a valid integer.\033[0m")
        except RuntimeError as e:
            print(e)

    # ======================================
    # Thermal Shield
    # ======================================
    plot_file_path = os.path.join(TS_plots_directory_path, "Volumetric heat source profiles.png")
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
    plt.plot(r_S, q_iiiS(r_S)/1e6, 'g', label='Radial (r) Volumetric heat source profile')
    plt.plot(r_S[0], q_iiiS(r_S[0])/1e6, 'or', label='Thermal Shield Inner Surface Value')
    plt.plot(r_S[-1], q_iiiS(r_S[-1])/1e6, 'or', label='Thermal Shield Outer Surface Value')
    plt.axhline(y = 0, color='black', linewidth='1', label='y=0')
    plt.xlabel('Radius (m)')
    plt.ylabel(r'$q_0$ (MW/m$^3$)')
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
    plt.tight_layout()
    plt.savefig(plot_file_path)
    if hs_flag:
        plt.show()
        plt.close()
    elif not hs_flag:
        plt.close()

    # ======================================
    # Plotting the T profiles
    # ======================================
    while True:
        try:
            T_pl_flag = int(input("\nDo you want to visualize the T profile across the vessel's wall and the thermal shield? (1: Yes, 0: No): "))
            if T_pl_flag not in (0, 1):
                raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
            T_pl_flag = bool(T_pl_flag)
            break  
        except ValueError:
            print("\033[31mPlease enter a valid integer.\033[0m")
        except RuntimeError as e:
            print(e)
            
    if (T_vessel_max - 273.15) > T_creep:
        creep_flag_V = bool(1)
    if (T_shield_max - 273.15) > T_creep:
        creep_flag_S = bool(1)
        
    if not adiab_flag:
        
        # ======================================
        # Thermal Shield T Profile
        # ======================================
        plot_file_path = os.path.join(TS_plots_directory_path, "Temperature profiles, averages and maxima.png")
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
        plt.tight_layout()
        plt.savefig(plot_file_path)
        if T_pl_flag:
            plt.show()
            plt.close()
        elif not T_pl_flag:
            plt.close()
     
    elif adiab_flag:

        # ======================================
        # Thermal Shield T Profile
        # ======================================
        plot_file_path = os.path.join(TS_plots_directory_path, "Temperature profiles, averages and maxima under AOW approximation.png")
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
        plt.title('Thermal Shield Temperature Profile, Average and Maximum')
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
        plt.title('Wall Temperature Profile, Average and Maximum under AOW Approximation')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(plot_file_path)
        if T_pl_flag:
            plt.show()
            plt.close()
        elif not T_pl_flag:
            plt.close()

    # ======================================
    # Plotting the thermal stress profiles
    # ======================================
    while True:
        try:
            sigma_th_pl_flag = int(input("\nDo you want to visualize a plot of the thermal stress profiles in the vessel and in the thermal shield? (1: Yes, 0: No): "))
            if sigma_th_pl_flag not in (0, 1):
                raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
            simga_th_pl_flag = bool(sigma_th_pl_flag)
            break  
        except ValueError:
            print("\033[31mPlease enter a valid integer.\033[0m")
        except RuntimeError as e:
            print(e)

    # ======================================
    # Thermal shield thermal stress profiles
    # ======================================
    plot_file_path = os.path.join(TS_plots_directory_path, "Thermal stresses profiles.png")
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
    plt.tight_layout()
    plt.savefig(plot_file_path)
    if sigma_th_pl_flag:
        plt.show()
        plt.close()
    elif not sigma_th_pl_flag:
        plt.close()

    # ======================================
    # Plotting the maximum thermal stress via the design curves
    # ======================================
    while True:
        try:
            des_pl_flag = int(input("\nDo you want to visualize a plot of the design curves and the maximum thermal stress in the vessel and in the thermal shield? (1: Yes, 0: No): "))
            if des_pl_flag not in (0, 1):
                raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
            des_pl_flag = bool(des_pl_flag)
            break  
        except ValueError:
            print("\033[31mPlease enter a valid integer.\033[0m")
        except RuntimeError as e:
            print(e)

    plot_file_path = os.path.join(TS_plots_directory_path, "Design curves.png")
    plt.figure(figsize=(10,10))
    plt.xlim(ba_ratio_plot[0], ba_ratio_plot[-1])
    plt.plot(ba_ratio_plot, L_Interpolator(ba_ratio_plot), 'k', label=f'Iso-mu = {mu_L} 1/m')
    plt.plot(ba_ratio_plot, R_Interpolator(ba_ratio_plot), 'k', label=f'Iso-mu = {mu_R} 1/m')
    plt.text(ba_ratio_plot[-1]+0.001, L_Interpolator(ba_ratio_plot)[-1], f'Iso-mu = {mu_L}', color='black', fontsize=10)
    plt.text(ba_ratio_plot[-1]+0.001, R_Interpolator(ba_ratio_plot)[-1], f'Iso-mu = {mu_R}', color='black', fontsize=10)
    plt.plot(R_ext/R_int, sigmaT_V,'or', label=r'Current $\sigma$$_T$ in the vessel')
    plt.plot(R_shield_ext/R_shield_int, sigmaT_S,'ob', label=r'Current $\sigma$$_T$ in the thermal shield')
    plt.xlabel('R$_{ext}$/R$_{int}$')
    plt.ylabel(r'$\sigma$$_T$')
    plt.title('Design curves')
    plt.legend()
    plt.grid()
    plt.savefig(plot_file_path)
    if des_pl_flag:
        plt.show()
        plt.close()
    elif not des_pl_flag:
        plt.close()

    # ======================================
    # Plotting the yield stress and stress intensity curves
    # ======================================
    while True:
        try:
            Interp_pl_flag = int(input("\nDo you want to visualize a plot of the Yield Stress and Stress Intensity as given by ASME for both the vessel and the thermal shield? (1: Yes, 0: No): "))
            if Interp_pl_flag not in (0, 1):
                raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
            Interp_pl_flag = bool(Interp_pl_flag)
            break  
        except ValueError:
            print("\033[31mPlease enter a valid integer.\033[0m")
        except RuntimeError as e:
            print(e)
    
    if max(T_thr) > T_des_vessel_C:
        Tplot = np.linspace(min(T_thr), max(T_thr), 1000)
    else:
        Tplot = np.linspace(min(T_thr), T_des_vessel_C, 1000)
    
    # ============================ 
    # Yield Stress
    # ============================
    plot_file_path = os.path.join(TS_plots_directory_path, "Yield Stress and Stress Intensity.png")
    plt.figure(figsize = (12,10))
    plt.subplot(1,2,1)
    plt.plot(T_thr, sigma_y, 'sk', label = 'Yield Stress Data')
    plt.plot(Tplot, Yield_Interpolator(Tplot), '--', color = 'orange', label = 'Yield Stress n-1 Interpolation')
    plt.plot(Tplot, Yield_CubicSpline(Tplot), 'green', label = 'Yield Stress Cubic Spline Interpolation')
    plt.plot(T_des_vessel_C, Yield_stress, '--or', label = r'Current Vessel Yield Stress $\sigma$$_y$')
    plt.plot(T_des_shield_C, Yield_stress_S, '--ob', label = r'Current Thermal Shield Yield Stress $\sigma$$_y$')
    plt.xlabel("Temperature (°C)")
    plt.ylabel(r"Yield Stress $\sigma$$_y$ (MPa)")
    plt.title("Yield Stress Data and Interpolation VS Temperature", fontsize = 10)
    plt.legend()
    plt.grid()

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
    plt.ylabel(r"Stress Intensity $\sigma$$_m$ (MPa)")
    plt.title("Stress Intensity Data and Interpolation VS Temperature", fontsize = 10)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(plot_file_path)
    if Interp_pl_flag:
        plt.show()
        plt.close()
    elif not Interp_pl_flag:
        plt.close()
        
    # ============================ 
    # Sizing of a thick cylinder under external pressure
    # ============================
    if R_int/t > 5:
        while True:
            try:
                ThinTubes_flag = int(input("\nWith a thickness value of %.3f m, the vessel can be considered thin. Are you interested in the thin tube limits for Elastic Instability and Plastic Collapse? (1: Yes, 0: No): " %t))
                if ThinTubes_flag not in (0, 1):
                    raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                ThinTubes_flag = bool(ThinTubes_flag)
                break  
            except ValueError:
                print("\033[31mPlease enter a valid integer.\033[0m")
            except RuntimeError as e:
                print(e)

        if ThinTubes_flag:
            print("\033[34mThe thin tube limits were adopted.\033[0m")
            p_E_fun = lambda Dt: 2 * (E/(1-(nu**2))) * (1/(Dt**3))              #Elastic Instability Limit for Thin Tubes
            p_0_fun = lambda Dt: 2 * Yield_stress * 1/Dt                        #Plastic Collapse Limit for Thin Tubes  -   Vessel
            p_0_fun_S = lambda Dt: 2 * Yield_stress_S * 1/Dt                    #Plastic Collapse Limit for Thin Tubes  -   Thermal Shield

        elif not ThinTubes_flag:
            print("\033[34mSkipping thin tube limits.\033[0m")
    else:
        print("\n\033[34mThe cylinder can't be considered thin. Skipping thin tube limits.\033[0m")
        ThinTubes_flag = bool(0)

    if ThinTubes_flag:
        while True:
            try:
                Corradi_flag = int(input("\nAre you interested in the Corradi Design Procedure? (1: Yes, 0: No): "))
                if Corradi_flag not in (0, 1):
                    raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                Corradi_flag = bool(Corradi_flag)
                break  
            except ValueError:
                print("\033[31mPlease enter a valid integer.\033[0m")
            except RuntimeError as e:
                print(e)
                
        if not Corradi_flag:
            # ============================ 
            # Elastic instability and plastic collapse curves
            # ============================
            while True:
                try:
                    Collapse_pl_flag = int(input("\nDo you want to visualize the buckling and plastic collapse curves for thin and thick tubes? (1: Yes, 0: No): "))
                    if Collapse_pl_flag not in (0, 1):
                        raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                    Collapse_pl_flag = bool(Collapse_pl_flag)
                    break  
                except ValueError:
                    print("\033[31mPlease enter a valid integer.\033[0m")
                except RuntimeError as e:
                    print(e)
            
            # ============================
            # Plastic collapse and buckling plots
            # ============================
            plot_file_path = os.path.join(TS_plots_directory_path, "Plastic Collapse and Buckling Curves.png")
            plt.figure(figsize = (8, 8))
            plt.xlim(0,50)
            plt.ylim(0.1,max(q_E_fun(Dt_ratio_plot)))
            plt.semilogy(Dt_ratio_plot, p_E_fun(Dt_ratio_plot), 'blue', label='p$_E$')
            plt.semilogy(Dt_ratio_plot, q_E_fun(Dt_ratio_plot), '--b', label='q$_E$')
            plt.semilogy(Dt_ratio_plot, p_0_fun(Dt_ratio_plot), 'red', label='p$_0$')
            plt.semilogy(Dt_ratio_plot, q_0_fun(Dt_ratio_plot), '--r', label='q$_0$')
            plt.axvline(x = Dt_Crit_Ratio, color = 'black', linewidth = '3', label = 'Critical Slenderness')
            plt.axvline(x = Current_Slenderness, color = 'green', linestyle='--', linewidth = '1.5', label = 'Current Vessel Slenderness')
            plt.plot(Current_Slenderness, Corradi_vessel[1], 'og', label='Current Vessel Allowable Pressure q$_a$')
            plt.fill_betweenx((0.1,max(q_E_fun(Dt_ratio_plot))), 0, Dt_Crit_Ratio, color='lightgreen', alpha=0.40, label='Plastic collapse dominated zone')
            plt.fill_betweenx((0.1,max(q_E_fun(Dt_ratio_plot))), Dt_Crit_Ratio, 50, color='orange', alpha=0.30, label='Elastic instability dominated zone')
            plt.xlabel("Geometrical Slenderness D/t")
            plt.ylabel("Theoretical Limit Values (MPa)")
            plt.title("Plastic Collapse and Buckling Curves")
            plt.legend()
            plt.grid()
            plt.savefig(plot_file_path)
            if Collapse_pl_flag:
                plt.show()
                plt.close()
            elif not Collapse_pl_flag:
                plt.close()

        elif Corradi_flag:
            while True:
                try:
                    Collapse_pl_flag = int(input("\nDo you want to visualize the buckling and plastic collapse curves for thin and thick tubes and the Corradi curve? (1: Yes, 0: No): "))
                    if Collapse_pl_flag not in (0, 1):
                        raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                    Collapse_pl_flag = bool(Collapse_pl_flag)
                    break  
                except ValueError:
                    print("\033[31mPlease enter a valid integer.\033[0m")
                except RuntimeError as e:
                    print(e)
            
            
            # ============================ 
            # Plastic collapse and buckling plots
            # ============================
            plot_file_path = os.path.join(TS_plots_directory_path, "Plastic Collapse and Buckling Curves - With Corradi.png")
            plt.figure(figsize = (8, 8))
            plt.xlim(0,50)
            plt.ylim(0.1,max(q_E_fun(Dt_ratio_plot)))
            plt.subplot(1,2,1)
            plt.semilogy(Dt_ratio_plot, p_E_fun(Dt_ratio_plot), 'blue', label='p$_E$')
            plt.semilogy(Dt_ratio_plot, q_E_fun(Dt_ratio_plot), '--b', label='q$_E$')
            plt.semilogy(Dt_ratio_plot, p_0_fun(Dt_ratio_plot), 'red', label='p$_0$')
            plt.semilogy(Dt_ratio_plot, q_0_fun(Dt_ratio_plot), '--r', label='q$_0$')
            plt.semilogy(Dt_ratio_plot, Corradi(Dt_ratio_plot)[0], 'orange', label='Corradi q$_c$')
            plt.axvline(x = Dt_Crit_Ratio, color = 'black', linewidth = '3', label = 'Critical Slenderness')
            plt.axvline(x = Current_Slenderness, color = 'green', linestyle='--', linewidth = '1.5', label = 'Current Vessel Slenderness')
            plt.plot(Current_Slenderness, Corradi_vessel[1], 'og', label='Current Vessel Allowable Pressure q$_a$')
            plt.fill_betweenx((0.1,max(q_E_fun(Dt_ratio_plot))), 0, Dt_Crit_Ratio, color='lightgreen', alpha=0.40, label='Plastic collapse dominated zone')
            plt.fill_betweenx((0.1,max(q_E_fun(Dt_ratio_plot))), Dt_Crit_Ratio, 50, color='orange', alpha=0.30, label='Elastic instability dominated zone')
            plt.xlabel("Geometrical Slenderness D/t")
            plt.ylabel("Theoretical Limit Values (MPa)")
            plt.title("Plastic Collapse and Buckling Curves")
            plt.legend()
            plt.grid()

            plt.subplot(1,2,2)
            plt.plot(Dt_ratio_plot, Corradi(Dt_ratio_plot)[3], 'k', label=r'Corradi $\mu$')
            plt.xlabel("Geometrical Slenderness D/t")
            plt.ylabel(r"Corradi $\mu$")
            plt.title(r"$\mu$ coefficient - Corradi Procedure")
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(plot_file_path)
            if Collapse_pl_flag:
                plt.show()
                plt.close()
            elif not Collapse_pl_flag:
                plt.close()
        
    elif not ThinTubes_flag:
        print("\033[34mAdopting Corradi Design Procedure.\033[0m")
        Corradi_flag = bool(1)
        while True:
            try:
                Collapse_pl_flag = int(input("\nDo you want to visualize the buckling and plastic collapse curves for thin and thick tubes and the Corradi curve? (1: Yes, 0: No): "))
                if Collapse_pl_flag not in (0, 1):
                    raise RuntimeError("\033[31mInvalid input! Please enter either 0 or 1.\033[0m")
                Collapse_pl_flag = bool(Collapse_pl_flag)
                break  
            except ValueError:
                print("\033[31mPlease enter a valid integer.\033[0m")
            except RuntimeError as e:
                print(e)
        
        # ============================ 
        # Plastic collapse and buckling plots
        # ============================
        plot_file_path = os.path.join(TS_plots_directory_path, "Plastic Collapse and Buckling Curves - With Corradi.png")
        plt.figure(figsize = (8, 8))
        plt.xlim(0,50)
        plt.ylim(0.1,max(q_E_fun(Dt_ratio_plot)))
        plt.subplot(1,2,1)
        plt.semilogy(Dt_ratio_plot, q_E_fun(Dt_ratio_plot), '--b', label='q$_E$')
        plt.semilogy(Dt_ratio_plot, q_0_fun(Dt_ratio_plot), '--r', label='q$_0$')
        plt.semilogy(Dt_ratio_plot, Corradi(Dt_ratio_plot)[0], 'orange', label='Corradi q$_c$')
        plt.axvline(x = Dt_Crit_Ratio, color = 'black', linewidth = '3', label = 'Critical Slenderness')
        plt.axvline(x = Current_Slenderness, color = 'green', linestyle='--', linewidth = '1.5', label = 'Current Vessel Slenderness')
        plt.plot(Current_Slenderness, Corradi_vessel[1], 'og', label='Current Vessel Allowable Pressure q$_a$')
        plt.fill_betweenx((0.1,max(q_E_fun(Dt_ratio_plot))), 0, Dt_Crit_Ratio, color='lightgreen', alpha=0.40, label='Plastic collapse dominated zone')
        plt.fill_betweenx((0.1,max(q_E_fun(Dt_ratio_plot))), Dt_Crit_Ratio, 50, color='orange', alpha=0.30, label='Elastic instability dominated zone')
        plt.xlabel("Geometrical Slenderness D/t")
        plt.ylabel("Theoretical Limit Values (MPa)")
        plt.title("Plastic Collapse and Buckling Curves")
        plt.legend()
        plt.grid()

        plt.subplot(1,2,2)
        plt.plot(Dt_ratio_plot, Corradi(Dt_ratio_plot)[3], 'k', label=r'Corradi $\mu$')
        plt.xlabel("Geometrical Slenderness D/t")
        plt.ylabel(r"Corradi $\mu$")
        plt.title(r"$\mu$ coefficient - Corradi Procedure")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(plot_file_path)
        if Collapse_pl_flag:
            plt.show()
            plt.close()
        elif not Collapse_pl_flag:
            plt.close()
            
    # ============================ 
    # Minimum thickness under internal pressure check
    # ============================
    t_min = (P_int_MPa * R_int)/(Stress_Intensity - 0.5*P_int_MPa)
    isAbove = t >= t_min
    t_min_S = (P_int_MPa * R_shield_int)/(Stress_Intensity_S - 0.5*P_int_MPa)
    isAbove_S = t_shield >= t_min_S
    
    # ============================ 
    # Current case specific path creation
    # ============================
    case_directory_name = []
    case_directory_name.append(f"t_{t}m")
    case_directory_name.append(f"_T_desV_{T_des_vessel_C}C")
    case_directory_name.append(f"_T_desS_{T_des_shield_C}C")
    if Def_P_flag:
        case_directory_name.append("_Def_P")
    else:
        case_directory_name.append("_Pint_%.1f_MPa_Pext_%.1f_MPa" %(P_int, P_cpp))
        if P_int != P_cpp:
            if eps_choice == 1:
                case_directory_name.append("_Plane_Strain")
            elif eps_choice == 0:
                case_directory_name.append("_Plane_Stress")
    case_directory_name.append("_q0")
    case_directory_name.append("_TS")
    if T1_choice == 0:
        case_directory_name.append("_Tin")
    elif T1_choice == 1:
        case_directory_name.append("_T_in + 10%%")
    elif T1_choice == 2:
        case_directory_name.append("_T_in + 20%%")
    elif T1_choice == 3:
        case_directory_name.append("_T_avg")
    elif T1_choice == 4:
        case_directory_name.append("_T_out_avg")
    if adiab_flag:
        case_directory_name.append("_AOW")
    if ThinTubes_flag:
        case_directory_name.append("_ThinTubes")
    if Corradi_flag:
        case_directory_name.append("_Corradi_s_%.2f" %s)
    case_directory_path = os.path.join(TS_directory_path, "".join(case_directory_name))
    
    # ============================
    # Final Results Printing and saving
    # ============================
    if os.path.exists(case_directory_path):
        shutil.rmtree(case_directory_path)                                       # Deletes the pre-existing folder
    if not os.path.exists(case_directory_path):                                  # Create the directory if it doesn't exist
        os.makedirs(case_directory_path, exist_ok=True)                          # Exist_ok=True avoids error if directory already exists
    
    file_path = os.path.join(case_directory_path, "Final_Results.txt")          # Specify the file path inside the newly created directory
    with open(file_path, "w") as file:
        output_lines = []

        # ============================
        # Hypothesis and data: not printed, saved only
        # ============================
        output_lines.append("################################################### Hypothesis and data ####################################################")
        output_lines.append("============================================================================================================================")
        output_lines.append("\nDefault pressures assumed: %s" %Def_P_flag)
        if not Def_P_flag:
            output_lines.append("Internal pressure: %.3f MPa" %P_int_MPa)
            output_lines.append("External pressure: %.3f MPa" %P_cpp_MPa)
        if P_int != P_cpp:
            output_lines.append("\nAssumed stress/strain condition (1: Plane Stress, 0: Plane Strain): %d" %eps_choice)
        output_lines.append("\n============================================================================================================================")
        output_lines.append("\nPresence of the volumetric heat source q0: %s" %q_0_flag)
        output_lines.append("Presence of the thermal shield: %s" %TS_flag)
        output_lines.append("Thermal shield chosen position (3: Arbitrary, 2: Middle, 1: Equal areas, 0: Equal h_1): %d" %user_D_choice)
        if user_D_choice == 0:
            output_lines.append("Heat transfer coefficients equalized in %d sub-iterations. Final difference: %.9e W/m²K" %(counter_h1, abs(h_1_int - h_1_ext)))
        output_lines.append("\n============================================================================================================================")
        output_lines.append("\nDiscretization along z: False")
        if T1_choice == 0:
            output_lines.append("Chosen temperature T1 to compute C1, C2: T_in = %.3s °C" %(T1-273.15))
        elif T1_choice == 1:
            output_lines.append("Chosen temperature T1 to compute C1, C2: T_in + 10% = %.3s °C" %(T1-273.15))
        elif T1_choice == 2:
            output_lines.append("Chosen temperature T1 to compute C1, C2: T_in + 20% = %.3s °C" %(T1-273.15))
        elif T1_choice == 3:
            output_lines.append("Chosen temperature T1 to compute C1, C2: T_avg = %.3s °C" %(T1-273.15))
        elif T1_choice == 4:
            output_lines.append("Chosen temperature T1 to compute C1, C2: T_out_avg = %.3s °C" %(T1-273.15))
        output_lines.append("Adiabatic Outer Wall approximation adopted: %s" %adiab_flag)
        #output_lines.append("Logarithmic Mean DeltaT approach adopted for inner heat flux computation: 0")
        output_lines.append("\n============================================================================================================================")
        output_lines.append("\nThin tube limits for Elastic Instability and Plastic Collapse adopted: %s" %ThinTubes_flag)
        output_lines.append("Corradi Design Procedure adopted: %s" %Corradi_flag)
        if Corradi_flag:
            output_lines.append("Safety coefficient adopted for the Corradi Design Procedure: %.3f" %s)
        output_lines.append("\n============================================================================================================================")
                
        # ============================
        # Actual Results
        # ============================
        output_lines.append("\n\n\n\n###################################################### Final  Results ######################################################")
        output_lines.append("============================================================================================================================")
        output_lines.append("\nThe vessel thickness has been increased %d times by 1cm. Computed vessel thickness: %.3f m" % (counter_vessel, t))
        output_lines.append("Computed thermal shield thickness: %.3f m" % t_shield)
        output_lines.append("Inner thermal shield radius: %.3f m" % R_shield_int)
        output_lines.append("\n============================================================================================================================")
        output_lines.append("\nVessel max ovality W: %.5f = %.3f%%" % (W, W * 100))
        output_lines.append("Maximum permissible deviation from theoretical form for the vessel according to NB-4221.2: e = %.3f m" % (0.3 * t))
        output_lines.append("Maximum difference in cross-sectional diameters: %.3f mm" % DeltaD_max)
        output_lines.append("\n============================================================================================================================")
        if isAbove:
            output_lines.append("\nThe current vessel wall thickness is equal to or greater than the minimum thickness required under internal pressure: %.3f m" %t_min)
        elif not isAbove:
            output_lines.append("\n============================================================================================================================")
            output_lines.append("WARNING: The current vessel wall thickness is below the minimum thickness required under internal pressure: %.3f m" %t_min)
            output_lines.append("============================================================================================================================")
        if isAbove_S:
            output_lines.append("The current thermal shield thickness is equal to or greater than the minimum thickness required under internal pressure: %.3f m" %t_min_S)
            output_lines.append("\n============================================================================================================================")
        elif not isAbove_S:
            output_lines.append("\n============================================================================================================================")
            output_lines.append("WARNING: The current thermal shield thickness is below the minimum thickness required under internal pressure: %.3f m" %t_min_S)
            output_lines.append("============================================================================================================================")

        # ============================ 
        # Heat Transfer Results
        # ============================
        output_lines.append("\n\n\n\n################################################## Heat transfer results ###################################################")
        output_lines.append("============================================================================================================================")
        output_lines.append("\nVolumetric heat source at the vessel inner surface: %.3f W/m³" %q_iii(r[0]))
        output_lines.append("Volumetric heat source at the vessel-insulation interface: %.3f W/m³" %q_iii(r[-1]))
        output_lines.append("\n============================================================================================================================")
        if user_D_choice == 3 or user_D_choice == 2:
            output_lines.append("\nInner heat transfer coefficient h1_int = %.3f W/(m²·K)" %h_1_int)
            output_lines.append("Outer heat transfer coefficient h1_ext = %.3f W/(m²·K)" %h_1_ext)
            output_lines.append("Chosen heat transfer coefficient h1 = %.3f W/(m²·K)    -    Conservative: minimum h means highest thermal stresses" %h_1)
        elif user_D_choice == 1:
            output_lines.append("\nInner heat transfer coefficient h1_int = %.3f W/(m²·K)" %h_1_int)
            output_lines.append("Outer heat transfer coefficient h1_ext = %.3f W/(m²·K)" %h_1_ext)
            if abs(h_1_int - h_1_ext) <= eps:
                output_lines.append("Chosen heat transfer coefficient h1 = %.3f W/(m²·K)    -    Essentially equal: the difference is of the order of %.3e" %(h_1, abs(h_1_int - h_1_ext)))
            else:
                output_lines.append("Chosen heat transfer coefficient h1 = %.3f W/(m²·K)    -    Conservative: minimum h means highest thermal stresses" %h_1)
        elif user_D_choice == 0:
            output_lines.append("\nInner heat transfer coefficient h1_int = %.3f W/(m²·K)" %h_1_int)
            output_lines.append("Outer heat transfer coefficient h1_ext = %.3f W/(m²·K)" %h_1_ext)
            output_lines.append("Heat transfer coefficients equalized in %d sub-iterations. Difference: %.9e W/m²K" %(counter_h1, abs(h_1_int - h_1_ext)))
        output_lines.append("\n============================================================================================================================")
        output_lines.append("\nHeat transfer coefficient h2 = %.3f W/(m²·K)" %h_2)
        output_lines.append("Overall heat transfer coefficient outside the vessel u2 = %.3f W/(m²·K)" %u_2)
        output_lines.append("\n============================================================================================================================")
        output_lines.append("\nThermal power flux on the inner vessel surface: %.3f kW/m²" %q_s1)
        output_lines.append("Thermal power flux on the outer vessel surface: %.3f kW/m²" %q_s2)
        output_lines.append("\n============================================================================================================================")
        
        # ============================ 
        # Temperature Results
        # ============================
        output_lines.append("\n\n\n\n####################################################### Temperatures #######################################################")
        output_lines.append("============================================================================================================================")
        if not adiab_flag:
            output_lines.append("\nAverage Vessel Temperature (numerical integration): %.3f °C" %(T_vessel_avg - 273.15))
            output_lines.append("Maximum Vessel Temperature: %.3f °C at r = %.3f m" %(T_vessel_max - 273.15, r_T_vessel_max))
            output_lines.append("Vessel Temperature at the inner surface: %-3f °C at r = %.3f m" %(T_vessel(r)[0] - 273.15, r[0]))
            output_lines.append("Vessel Temperature at the outer surface: %-3f °C at r = %.3f m" %(T_vessel(r)[-1] - 273.15, r[-1]))
            if creep_flag_V:
                output_lines.append("\n============================================================================================================================")
                output_lines.append("WARNING: The maximum vessel temperature T = %.3f °C exceeds the creep threshold temperature of %d °C!" %(T_vessel_max - 273.15, T_creep))
                output_lines.append("============================================================================================================================")
            elif not creep_flag_V:
                output_lines.append("\nThere is no risk of thermal creep occurring in the vessel.")
                output_lines.append("\n============================================================================================================================")
                
            output_lines.append("\nAverage Thermal Shield Temperature (numerical integration): %.3f °C" %(T_shield_avg - 273.15))
            output_lines.append("Maximum Thermal Shield Temperature: %.3f °C at r = %.3f m" %(T_shield_max - 273.15, r_T_shield_max))
            output_lines.append("Thermal Shield Temperature at the inner surface: %-3f °C at r = %.3f m" %(T_shield(r_S)[0] - 273.15, r_S[0]))
            output_lines.append("Thermal Shield Temperature at the outer surface: %-3f °C at r = %.3f m" %(T_shield(r_S)[-1] - 273.15, r_S[-1]))
            if creep_flag_S:
                output_lines.append("\n============================================================================================================================")
                output_lines.append("WARNING: The maximum thermal shield temperature T = %.3f °C exceeds the creep threshold temperature of %d °C!" %(T_shield_max - 273.15, T_creep))
                output_lines.append("============================================================================================================================")
            elif not creep_flag_S:
                output_lines.append("\nThere is no risk of thermal creep occurring in the thermal shield.")
                output_lines.append("\n============================================================================================================================")
                
        elif adiab_flag:
            output_lines.append("\nAverage Vessel Temperature under Adiabatic Outer Wall approximation (numerical integration): %.3f °C" %(T_vessel_avg - 273.15))
            output_lines.append("Maximum Vessel Temperature under Adiabatic Outer Wall approximation: %.3f °C at r = %.3f m" %(T_vessel_max - 273.15, r_T_vessel_max))
            output_lines.append("Vessel Temperature at the inner surface under Adiabatic Outer Wall approximation: %-3f °C at r = %.3f m" %(T_vessel(r)[0] - 273.15, r[0]))
            output_lines.append("Vessel Temperature at the outer surface under Adiabatic Outer Wall approximation: %-3f °C at r = %.3f m" %(T_vessel(r)[-1] - 273.15, r[-1]))
            if creep_flag_V:
                output_lines.append("\n============================================================================================================================")
                output_lines.append("WARNING: The maximum vessel temperature T = %.3f °C exceeds the creep threshold temperature of %d °C!" %(T_vessel_max - 273.15, T_creep))
                output_lines.append("============================================================================================================================")
            elif not creep_flag_V:
                output_lines.append("\nThere is no risk of thermal creep occurring in the vessel.")
                output_lines.append("\n============================================================================================================================")
            
            output_lines.append("\nAverage Thermal Shield Temperature (numerical integration): %.3f °C" %(T_shield_avg - 273.15))
            output_lines.append("Maximum Thermal Shield Temperature: %.3f °C at r = %.3f m" %(T_shield_max - 273.15, r_T_shield_max))
            output_lines.append("Thermal Shield Temperature at the inner surface: %-3f °C at r = %.3f m" %(T_shield(r_S)[0] - 273.15, r_S[0]))
            output_lines.append("Thermal Shield Temperature at the outer surface: %-3f °C at r = %.3f m" %(T_shield(r_S)[-1] - 273.15, r_S[-1]))
            if creep_flag_S:
                output_lines.append("\n============================================================================================================================")
                output_lines.append("WARNING: The maximum thermal shield temperature T = %.3f °C exceeds the creep threshold temperature of %d °C!" %(T_shield_max - 273.15, T_creep))
                output_lines.append("============================================================================================================================")
            elif not creep_flag_S:
                output_lines.append("\nThere is no risk of thermal creep occurring in the thermal shield.")
                output_lines.append("\n============================================================================================================================")

        # ============================ 
        # Stress Results
        # ============================
        output_lines.append("\n\n\n\n######################################################### Stresses #########################################################")
        output_lines.append("============================================================================================================================")
        output_lines.append("\nMaximum Thermal Hoop Stress in the vessel: %.3f Mpa at r = %.3f m" %(sigma_t_th_V_max, r_sigma_t_th_V_max))
        #output_lines.append("Maximum Thermal Hoop Stress in the vessel (Simplified formula): %.3f Mpa at r = %.3f m" %(sigma_t_th_V_max_SIMP, r_sigma_t_th_V_max_SIMP))
        output_lines.append("Maximum thermal hoop stress in the vessel via design curves: %.3f MPa" %sigma_t_th_V_max_DES)

        output_lines.append("\nMaximum Thermal Hoop Stress in the thermal shield: %.3f Mpa at r = %.3f m" %(sigma_t_th_S_max, r_sigma_t_th_S_max))
        #output_lines.append("Maximum Thermal Hoop Stress in the thermal shield (Simplified formula): %.3f Mpa at r = %.3f m" %(sigma_t_th_S_max_SIMP, r_sigma_t_th_S_max_SIMP))
        output_lines.append("Maximum thermal hoop stress in the thermal shield via design curves: %.3f MPa" %sigma_t_th_S_max_DES)
        output_lines.append("\n============================================================================================================================")
        output_lines.append("\nGuest-Tresca comparison stress of primary stresses only in the vessel - Mariotte solution: %.3f Mpa" %sigma_cTR_M_PO)
        output_lines.append("Guest-Tresca comparison stress of primary stresses only in the vessel - Lamé solution: %.3f Mpa" %sigma_cTR_L_PO)
        output_lines.append("Guest-Tresca comparison stress in the vessel - Mariotte solution: %.3f Mpa" %sigma_cTR_M)
        output_lines.append("Guest-Tresca comparison stress in the vessel - Lamé solution: %.3f Mpa" %sigma_cTR_L)

        output_lines.append("\nGuest-Tresca comparison stress of primary stresses only in the thermal shield - Mariotte solution: %.3f Mpa" %sigma_cTR_MS_PO)
        output_lines.append("Guest-Tresca comparison stress of primary stresses only in the thermal shield - Lamé solution: %.3f Mpa" %sigma_cTR_LS_PO)
        output_lines.append("Guest-Tresca comparison stress in the thermal shield - Mariotte solution: %.3f Mpa" %sigma_cTR_MS)
        output_lines.append("Guest-Tresca comparison stress in the thermal shield - Lamé solution: %.3f Mpa" %sigma_cTR_LS)
        output_lines.append("\n============================================================================================================================")
        output_lines.append("\nFor a design vessel temperature of %.3f °C: " %T_des_vessel_C)
        output_lines.append("Yield Stress: Sy = %.3f MPa" %Yield_stress)
        output_lines.append("Stress Intensity: Sm = %.3f MPa" %Stress_Intensity)
        output_lines.append("Allowable Stress: %.3f MPa" %sigma_allowable)
        output_lines.append("\nFor a design thermal shield temperature of %.3f °C: " %T_des_shield_C)
        output_lines.append("Yield Stress: Sy = %.3f MPa" %Yield_stress_S)
        output_lines.append("Stress Intensity: Sm = %.3f MPa" %Stress_Intensity_S)
        output_lines.append("Allowable Stress: %.3f MPa" %sigma_allowable_S)
        output_lines.append("\n============================================================================================================================")
        
        # ============================ 
        # Thermal Shield
        # ============================
        output_lines.append("\n\n\n\n###################################################### Thermal Shield ######################################################")
        output_lines.append("============================================================================================================================")
        if flag_primsec_S or flag_prim_S:
            output_lines.append("\nAccording to Lamé:")
            output_lines.append("Maximum absolute value of the total radial stress: %.3f MPa\nMaximum absolute value of the total hoop stress: %.3f MPa\nMaximum absolute value of the total axial stress: %.3f MPa" %(max(abs(sigma_r_totL_S)),max(abs(sigma_t_totL_S)),max(abs(sigma_z_totL_S))))
            output_lines.append("\nMaximum value of the primary radial stress: %.3f MPa\nMaximum value of the primary hoop stress: %.3f MPa\nPrimary axial stress: %.3f MPa" %(max(sigma_rL_S),max(sigma_tL_S),sigma_zL_S))
            if max(abs(sigma_r_totL_S)) > 3*Stress_Intensity_S:
                output_lines.append("\nThe maximum value of the total radial stress exceeds allowable stress.")
            if max(abs(sigma_t_totL_S)) > 3*Stress_Intensity_S:
                output_lines.append("\nThe maximum value of the total hoop stress exceeds allowable stress.")
            if max(abs(sigma_z_totL_S)) > 3*Stress_Intensity_S:
                output_lines.append("\nThe maximum value of the total axial stress exceeds allowable stress.")
            if max(sigma_rL_S) > Stress_Intensity_S:
                output_lines.append("\nThe maximum value of the primary radial stress exceeds allowable stress.")
            if max(sigma_tL_S) > Stress_Intensity_S:
                output_lines.append("\nThe maximum value of the primary hoop stress exceeds allowable stress.")
            if sigma_zL_S > Stress_Intensity_S:
                output_lines.append("\nThe primary axial stress exceeds allowable stress.")
                
            output_lines.append("\n============================================================================================================================")
            output_lines.append("\nAccording to Mariotte:")
            output_lines.append("Maximum absolute value of the total radial stress: %.3f MPa\nMaximum absolute value of the total hoop stress: %.3f MPa\nMaximum absolute value of the total axial stress: %.3f MPa" %(max(abs(sigma_r_totM_S)),max(abs(sigma_t_totM_S)),max(abs(sigma_z_totM_S))))
            output_lines.append("\nPrimary radial stress: %.3f MPa\nPrimary hoop stress: %.3f MPa\nPrimary axial stress: %.3f MPa" %(sigma_rM_S,sigma_tM_S,sigma_zM_S))
            if max(abs(sigma_r_totM_S)) > 3*Stress_Intensity_S:
                output_lines.append("\nThe maximum value of the total radial stress exceeds allowable stress.")
            if max(abs(sigma_t_totM_S)) > 3*Stress_Intensity_S:
                output_lines.append("\nThe maximum value of the total hoop stress exceeds allowable stress.")
            if max(abs(sigma_z_totM_S)) > 3*Stress_Intensity_S:
                output_lines.append("\nThe maximum value of the total axial stress exceeds allowable stress.")
            if sigma_rM_S > Stress_Intensity_S:
                output_lines.append("\nThe primary radial stress exceeds allowable stress.")
            if sigma_tM_S > Stress_Intensity_S:
                output_lines.append("\nThe primary hoop stress exceeds allowable stress.")
            if sigma_zM_S > Stress_Intensity_S:
                output_lines.append("\nThe primary axial stress exceeds allowable stress.")
            output_lines.append("\n============================================================================================================================")
            output_lines.append("WARNING: The current stress state in the thermal shield is not acceptable.")
            output_lines.append("============================================================================================================================")
        
        elif not flag_primsec_S and not flag_prim_S:
            output_lines.append("\nAccording to Lamé:")
            output_lines.append("Maximum absolute value of the total radial stress: %.3f MPa\nMaximum absolute value of the total hoop stress: %.3f MPa\nMaximum absolute value of the total axial stress: %.3f MPa" %(max(abs(sigma_r_totL_S)),max(abs(sigma_t_totL_S)),max(abs(sigma_z_totL_S))))
            output_lines.append("\nAll are lower than 3Sm = %.3f MPa" %(3*Stress_Intensity_S))         
            output_lines.append("\nMaximum value of the primary radial stress: %.3f MPa\nMaximum value of the primary hoop stress: %.3f MPa\nPrimary axial stress: %.3f MPa" %(max(sigma_rL_S),max(sigma_tL_S),sigma_zL_S))
            output_lines.append("\nAll are lower than Sm = %.3f MPa" %Stress_Intensity_S)
            output_lines.append("\n============================================================================================================================")
            output_lines.append("\nAccording to Mariotte:")
            output_lines.append("Maximum absolute value of the total radial stress: %.3f MPa\nMaximum absolute value of the total hoop stress: %.3f MPa\nMaximum absolute value of the total axial stress: %.3f MPa" %(max(abs(sigma_r_totM_S)),max(abs(sigma_t_totM_S)),max(abs(sigma_z_totM_S))))
            output_lines.append("\nAll are lower than 3Sm = %.3f MPa" %(3*Stress_Intensity_S))         
            output_lines.append("\nPrimary radial stress: %.3f MPa\nPrimary hoop stress: %.3f MPa\nPrimary axial stress: %.3f MPa" %(sigma_rM_S,sigma_tM_S,sigma_zM_S))
            output_lines.append("\nAll are lower than Sm = %.3f MPa" %Stress_Intensity_S)
            output_lines.append("\n============================================================================================================================")

        if (sigma_cTR_LS_PO < Stress_Intensity_S):
            output_lines.append("\nThe comparison stress of primary stresses only according to Tresca-Lamé Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_LS_PO, Stress_Intensity_S))
        else:
            output_lines.append("\n============================================================================================================================")
            output_lines.append("WARNING: The comparison stress of primary stresses only according to Tresca-Lamé Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_LS_PO, Stress_Intensity_S))
            output_lines.append("============================================================================================================================")
        if (sigma_cTR_MS_PO < Stress_Intensity_S):
            output_lines.append("The comparison stress of primary stresses only according to Tresca-Mariotte Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_MS_PO, Stress_Intensity_S))
            output_lines.append("\n============================================================================================================================")
        else:
            output_lines.append("\n============================================================================================================================")
            output_lines.append("WARNING: The comparison stress of primary stresses only according to Tresca-Mariotte Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_MS_PO, Stress_Intensity_S))
            output_lines.append("============================================================================================================================")
        
        if (sigma_cTR_LS < 3*Stress_Intensity_S):
            output_lines.append("\nThe comparison stress according to Tresca-Lamé Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_LS, 3*Stress_Intensity_S))
        else:
            output_lines.append("\n============================================================================================================================")
            output_lines.append("WARNING: The comparison stress according to Tresca-Lamé Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_LS, 3*Stress_Intensity_S))
            output_lines.append("============================================================================================================================")
        if (sigma_cTR_MS < 3*Stress_Intensity_S):
            output_lines.append("The comparison stress according to Tresca-Mariotte Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_MS, 3*Stress_Intensity_S))
            output_lines.append("\n============================================================================================================================")
        else:
            output_lines.append("\n============================================================================================================================")
            output_lines.append("WARNING: The comparison stress according to Tresca-Mariotte Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_MS, 3*Stress_Intensity_S))
            output_lines.append("============================================================================================================================")
            
        # if not flag_primsec_S and not flag_prim_S and sigma_cTR_LS < sigma_allowable_S and sigma_cTR_MS < sigma_allowable_S and not creep_flag_S:
        #     output_lines.append("\n\n\n\n############################################################################################################################")
        #     output_lines.append("The thermal shield's integrity is ensured.")
        #     output_lines.append("############################################################################################################################")

        # ============================
        # Vessel
        # ============================
        output_lines.append("\n\n\n\n########################################################## Vessel ##########################################################")
        output_lines.append("============================================================================================================================")
        if flag_primsec or flag_prim:
            output_lines.append("\nAccording to Lamé:")
            output_lines.append("Maximum absolute value of the total radial stress: %.3f MPa\nMaximum absolute value of the total hoop stress: %.3f MPa\nMaximum absolute value of the total axial stress: %.3f MPa" %(max(abs(sigma_r_totL)),max(abs(sigma_t_totL)),max(abs(sigma_z_totL))))
            output_lines.append("\nMaximum value of the primary radial stress: %.3f MPa\nMaximum value of the primary hoop stress: %.3f MPa\nPrimary axial stress: %.3f MPa" %(max(sigma_rL),max(sigma_tL),sigma_zL))
            if max(abs(sigma_r_totL)) > 3*Stress_Intensity:
                output_lines.append("\nThe maximum value of the total radial stress exceeds allowable stress.")
            if max(abs(sigma_t_totL)) > 3*Stress_Intensity:
                output_lines.append("\nThe maximum value of the total hoop stress exceeds allowable stress.")
            if max(abs(sigma_z_totL)) > 3*Stress_Intensity:
                output_lines.append("\nThe maximum value of the total axial stress exceeds allowable stress.")
            if max(sigma_rL) > Stress_Intensity:
                output_lines.append("\nThe maximum value of the primary radial stress exceeds allowable stress.")
            if max(sigma_tL) > Stress_Intensity:
                output_lines.append("\nThe maximum value of the primary hoop stress exceeds allowable stress.")
            if sigma_zL > Stress_Intensity:
                output_lines.append("\nThe primary axial stress exceeds allowable stress.")
                
            output_lines.append("\n============================================================================================================================")
            output_lines.append("\nAccording to Mariotte:")
            output_lines.append("Maximum absolute value of the total radial stress: %.3f MPa\nMaximum absolute value of the total hoop stress: %.3f MPa\nMaximum absolute value of the total axial stress: %.3f MPa" %(max(abs(sigma_r_totM)),max(abs(sigma_t_totM)),max(abs(sigma_z_totM))))
            output_lines.append("\nPrimary radial stress: %.3f MPa\nPrimary hoop stress: %.3f MPa\nPrimary axial stress: %.3f MPa" %(sigma_rM,sigma_tM,sigma_zM))
            if max(abs(sigma_r_totM)) > 3*Stress_Intensity:
                output_lines.append("\nThe maximum value of the total radial stress exceeds allowable stress.")
            if max(abs(sigma_t_totM)) > 3*Stress_Intensity:
                output_lines.append("\nThe maximum value of the total hoop stress exceeds allowable stress.")
            if max(abs(sigma_z_totM)) > 3*Stress_Intensity:
                output_lines.append("\nThe maximum value of the total axial stress exceeds allowable stress.")
            if sigma_rM > Stress_Intensity:
                output_lines.append("\nThe primary radial stress exceeds allowable stress.")
            if sigma_tM > Stress_Intensity:
                output_lines.append("\nThe primary hoop stress exceeds allowable stress.")
            if sigma_zM > Stress_Intensity:
                output_lines.append("\nThe primary axial stress exceeds allowable stress.")
            output_lines.append("\n============================================================================================================================")
            output_lines.append("WARNING: The current stress state in the vessel is not acceptable.")
            output_lines.append("============================================================================================================================")
    
        elif not flag_primsec and not flag_prim:
            output_lines.append("\nAccording to Lamé:")
            output_lines.append("Maximum absolute value of the total radial stress: %.3f MPa\nMaximum absolute value of the total hoop stress: %.3f MPa\nMaximum absolute value of the total axial stress: %.3f MPa" %(max(abs(sigma_r_totL)),max(abs(sigma_t_totL)),max(abs(sigma_z_totL))))
            output_lines.append("\nAll are lower than 3Sm = %.3f MPa" %(3*Stress_Intensity))         
            output_lines.append("\nMaximum value of the primary radial stress: %.3f MPa\nMaximum value of the primary hoop stress: %.3f MPa\nPrimary axial stress: %.3f MPa" %(max(sigma_rL),max(sigma_tL),sigma_zL))
            output_lines.append("\nAll are lower than Sm = %.3f MPa" %Stress_Intensity)
            output_lines.append("\n============================================================================================================================")
            output_lines.append("\nAccording to Mariotte:")
            output_lines.append("Maximum absolute value of the total radial stress: %.3f MPa\nMaximum absolute value of the total hoop stress: %.3f MPa\nMaximum absolute value of the total axial stress: %.3f MPa" %(max(abs(sigma_r_totM)),max(abs(sigma_t_totM)),max(abs(sigma_z_totM))))
            output_lines.append("\nAll are lower than 3Sm = %.3f MPa" %(3*Stress_Intensity)) 
            output_lines.append("\nPrimary radial stress: %.3f MPa\nPrimary hoop stress: %.3f MPa\nPrimary axial stress: %.3f MPa" %(sigma_rM,sigma_tM,sigma_zM))
            output_lines.append("\nAll are lower than Sm = %.3f MPa" %Stress_Intensity)
            output_lines.append("\n============================================================================================================================")

        if (sigma_cTR_L_PO < Stress_Intensity):
            output_lines.append("\nThe comparison stress of primary stresses only according to Tresca-Lamé Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_L_PO, Stress_Intensity))
        else:
            output_lines.append("\n============================================================================================================================")
            output_lines.append("WARNING: The comparison stress of primary stresses only according to Tresca-Lamé Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_L_PO, Stress_Intensity))
            output_lines.append("============================================================================================================================")
        if (sigma_cTR_M_PO < Stress_Intensity):
            output_lines.append("The comparison stress of primary stresses only according to Tresca-Mariotte Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_M_PO, Stress_Intensity))
            output_lines.append("\n============================================================================================================================")
        else:
            output_lines.append("\n============================================================================================================================")
            output_lines.append("WARNING: The comparison stress of primary stresses only according to Tresca-Mariotte Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_M_PO, Stress_Intensity))
            output_lines.append("============================================================================================================================")
            
        if (sigma_cTR_L < 3*Stress_Intensity):
            output_lines.append("\nThe comparison stress according to Tresca-Lamé Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_L, 3*Stress_Intensity))
        else:
            output_lines.append("\n============================================================================================================================")
            output_lines.append("WARNING: The comparison stress according to Tresca-Lamé Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_L, 3*Stress_Intensity))
            output_lines.append("============================================================================================================================")
        if (sigma_cTR_M < 3*Stress_Intensity):
            output_lines.append("The comparison stress according to Tresca-Mariotte Sc = %.3f MPa is lower than the allowable stress Sa = %.3f MPa" %(sigma_cTR_M, 3*Stress_Intensity))
            output_lines.append("\n============================================================================================================================")
        else:
            output_lines.append("\n============================================================================================================================")
            output_lines.append("WARNING: The comparison stress according to Tresca-Mariotte Sc = %.3f MPa is higher than the allowable stress Sa = %.3f MPa" %(sigma_cTR_M, 3*Stress_Intensity))
            output_lines.append("============================================================================================================================")
        
        output_lines.append("\n\n\n\n######################################################### Buckling #########################################################")
        output_lines.append("============================================================================================================================")
        output_lines.append("\nAccording to the Corradi Design Procedure:")
        output_lines.append("Current slenderness: %.3f    -   Critical slenderness: %.3f" %(Current_Slenderness, Dt_Crit_Ratio))
        output_lines.append("\nThe theoretical limit for collapse pressure, accounting for ovality, is: q_c = %.3f MPa = %.3f bar" %(Corradi_vessel[0], 10*Corradi_vessel[0]))
        output_lines.append("A safety factor s = %.3f was assumed. \nThe allowable external pressure is thus: q_a = %.3f MPa = %.3f bar" %(Corradi_vessel[2], Corradi_vessel[1], 10*Corradi_vessel[1]))
        if buckling_flag:
            output_lines.append("The given external pressure of %.3f bar is lower than the allowable pressure of %.3f bar" %(P_cpp, 10*Corradi_vessel[1]))
            output_lines.append("\n============================================================================================================================")
        elif not buckling_flag:
            output_lines.append("\n============================================================================================================================")
            output_lines.append("WARNING: The given external pressure of %.3f bar is higher than the allowable pressure of %.3f bar: a change in thickness is required!" %(P_cpp, 10*Corradi_vessel[1]))
            output_lines.append("============================================================================================================================")
            
        if buckling_flag and sigma_cTR_L_PO < Stress_Intensity and sigma_cTR_M_PO < Stress_Intensity and sigma_cTR_L < 3*Stress_Intensity and sigma_cTR_M < 3*Stress_Intensity and not creep_flag_V:
            output_lines.append("\n\n\n\n############################################################################################################################")
            output_lines.append("The vessel's integrity is ensured: the design is correct!")
            output_lines.append("############################################################################################################################")
    
        for line in output_lines:
            file.write(line + '\n')
        shutil.move(TS_plots_directory_path, case_directory_path)
        
        print("\n\n\033[32m############################################################################################################################\033[0m")
        print("\033[32mResults have been saved at: %s\033[0m" %case_directory_path)
        print("\033[32m############################################################################################################################\033[0m\n\n")