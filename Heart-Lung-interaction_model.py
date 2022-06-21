import math
from math import cos
from math import floor

# Import numpy
from typing import Optional, Tuple, Iterable
import itertools
import numpy as np
import matplotlib.pyplot as plt
from numpy import ndarray
from switch import Switch
from math import pi
from math import exp
from math import sqrt
from scipy.integrate import odeint
import csv
import pandas as pd
import os
import shutil
from aether.diagnostics import set_diagnostics_on
from aether.indices import perfusion_indices, get_ne_radius
from aether.filenames import read_geometry_main, get_filename
from aether.geometry import append_units, define_node_geometry, define_1d_elements, define_rad_from_geom, \
    add_matching_mesh
from aether.exports import export_1d_elem_geometry, export_node_geometry, export_1d_elem_field, export_node_field, \
    export_terminal_perfusion
from aether.pressure_resistance_flow import evaluate_prq
from csv import writer


class Heart_Lung:
    def __init__(self):
        #Model parameters
        self.T_vc = 0.34  # The duration of ventricles contraction (dimensionless)(in Model)
        self.T_vr = 0.15  # The duration of ventricles relaxation (dimensionless)(in Model)
        self.t_ar = 0.97  # The time when the atria start to relax (dimensionless)(in Model)
        self.T_ar = 0.17  # The duration of atria relaxation (dimensionless)(in Model)
        self.t_ac = 0.80  # The time when the atria start to contraction (dimensionless)(in Model)
        self.T_ac = 0.17  # The duration of atria contraction (dimensionless)(in Model)


        # blood_pressure_Atria_ventricles
        self.E_ra_A = 7.998e+6  # amplitude value of the RA elastance (J_per_m6)
        self.E_ra_B = 9.331e+6  # baseline value of the RA elastance (J_per_m6)
        # self.E_ra_A = 0.06  # amplitude value of the RA elastance (mmHg_per_mL(J_per_m6*0.000000008))
        # self.E_ra_B = 0.07  # baseline value of the RA elastance (mmHg_per_mL(J_per_m6*0.000000008))
        #self.V_ra = 20.0e-6  # Initial blood volume of RA
        self.V_ra_0 = 4.0e-6  # dead blood volume of RA (m3)
        # self.V_ra_0 = 4  # dead blood volume of RA (mL(m3*1000000))


        self.E_rv_A = 73.315e+6  # amplitude value of the RV elastance (J_per_m6)
        self.E_rv_B = 6.665e+6  # baseline value of the RV elastance (J_per_m6)
        # self.E_rv_A = 0.58  # amplitude value of the RV elastance (mmHg_per_mL(J_per_m6*0.000000008))
        # self.E_rv_B = 0.05  # baseline value of the RV elastance (mmHg_per_mL(J_per_m6*0.000000008))
        #self.V_rv = 500.0e-6  # Initial blood volume of RV
        self.V_rv_0 = 10.0e-6  # dead blood volume of RV (m3)
        # self.V_rv_0 = 10  # dead blood volume of RV (mL(m3*1000000))

        self.E_la_A = 9.331e+6  # amplitude value of the LA elastance (J_per_m6)
        self.E_la_B = 11.997e+6  # baseline value of the LA elastance (J_per_m6)
        # self.E_la_A = 0.07  # amplitude value of the LA elastance (mmHg_per_mL(J_per_m6*0.000000008))
        # self.E_la_B = 0.09  # baseline value of the LA elastance (mmHg_per_mL(J_per_m6*0.000000008))
        #self.V_la = 20.0e-6  # blood volume of LA
        self.V_la_0 = 4.0e-6  # dead blood volume of LA (m3)
        # self.V_la_0 = 4  # dead blood volume of LA (mL(m3*1000000))


        self.E_lv_A = 366.575e+6  # amplitude value of the LV elastance (J_per_m6)
        self.E_lv_B = 10.664e+6  # baseline value of the LV elastance (J_per_m6)
        # self.E_lv_A = 366.575e+6  # amplitude value of the LV elastance (J_per_m6)
        # self.E_lv_B = 10.664e+6  # baseline value of the LV elastance (J_per_m6)
        # self.E_lv_A = 2.9  # amplitude value of the LV elastance (mmHg_per_mL(J_per_m6*0.000000008))
        # self.E_lv_B = 0.08  # baseline value of the LV elastance (mmHg_per_mL(J_per_m6*0.000000008))
        # self.E_lv_A = 325.0e+6  # amplitude value of the LV elastance (J_per_m6)(in PH)
        # self.E_lv_B = 75.0e+6  # baseline value of the LV elastance (J_per_m6)(in PH)
        #self.V_lv = 500.0e-6  # blood volume of LV
        self.V_lv_0 = 5.0e-6  # dead blood volume of LV (m3)
        # self.V_lv_0 = 5  # dead blood volume of LV (mL(m3*1000000))

        # blood_flow_atria_ventricles
        self.CQ_trv = 17.6427e-6  # triscupid valve coefficient (UnitValve)
        self.CQ_puv = 14.3124e-6  # pulmonary valve coefficient (UnitValve)
        self.CQ_miv = 17.6427e-6  # mitral valve coefficient (UnitValve)
        self.CQ_aov = 14.3124e-6  # aortic valve coefficient (UnitValve)
        # self.CQ_trv = 34.6427e-6  # triscupid valve coefficient (UnitValve)
        # self.CQ_puv = 30.3124e-6  # pulmonary valve coefficient (UnitValve)
        # self.CQ_miv = 34.6427e-6  # mitral valve coefficient (UnitValve)
        # self.CQ_aov = 30.3124e-6  # aortic valve coefficient (UnitValve)



        # pulmonary circulation
        # self.C_pulmonary_artery = 61.0e-9  # artery compliance (m6_per_J)(exercise value)
        # self.C_pulmonary_artery = 3.0e-9  # artery compliance (m6_per_J)(PH value)
        self.C_pulmonary_artery = 0.0309077e-6  # artery compliance (m6_per_J)(model value)
        # self.C_pulmonary_artery = 4.12 # artery compliance (m6_per_J)(model value)(mL_per_mmHg(m6_per_J*133322368.421))

        # self.I_pulmonary_artery = 1.0e+6  # artery inductance (Js2_per_m6)
        # self.R_pulmonary_artery = 8750000  # artery resistance (Js_per_m6)(exercise value)
        # self.R_pulmonary_artery = 22500000  # artery resistance (Js_per_m6)(PH value)
        self.R_pulmonary_artery = 10.664e+6  # artery resistance (Js_per_m6)(model value)
        # self.R_pulmonary_artery = 0.08  # artery resistance (Js_per_m6)(model value)(mmHgs_per_mL(Js_per_m6*0.000000008))

        # self.C_pulmonary_vein = 9.0e-8  # vein compliance (m6_per_J)(exercise value)
        # self.C_pulmonary_vein = 29.0e-9  # vein compliance (m6_per_J)(PH value)
        self.C_pulmonary_vein = 0.060015e-6  # vein compliance (m6_per_J)(model value)
        # self.C_pulmonary_vein = 7.99  # vein compliance (m6_per_J)(model value)(mL_per_mmHg(m6_per_J*133322368.421))

        # self.R_pulmonary_vein = 1100000  # vein resistance (Js_per_m6)(exercise value)
        # self.R_pulmonary_vein = 13750000  # vein resistance (Js_per_m6)(PH value)
        self.R_pulmonary_vein = 1.333e+6  # vein resistance (Js_per_m6)(model value)
        # self.R_pulmonary_vein = 1.333e+6  # vein resistance (Js_per_m6)(model value)
        # self.R_pulmonary_vein = 0.01  # vein resistance (Js_per_m6)(model value)(mmHgs_per_mL(Js_per_m6*0.000000008))
        # self.I_pulmonary_vein = 1.0e+6  # vein inductance (Js2_per_m6)


        # Systemic circulation      
        # self.C_systemic_artery = 0.0309077e-6  # artery compliance (m6_per_J)
        self.C_systemic_artery = 21.0e-9  # vein compliance (m6_per_J)
        # self.C_systemic_artery = 2.79  # vein compliance (m6_per_J)(mL_per_mmHg(m6_per_J*133322368.421))

        # self.I_systemic_artery = 1.0e+6   # artery inductance (Js2_per_m6)
        # self.R_systemic_artery = 10.664e+6 # artery resistance (Js_per_m6)
        self.R_systemic_artery = 1.0e+8 # artery resistance (Js_per_m6)
        # self.R_systemic_artery = 0.8 # artery resistance (Js_per_m6)(mmHgs_per_mL(Js_per_m6*0.000000008))

        # self.C_systemic_vein = 0.060015e-6  # vein compliance (m6_per_J)
        self.C_systemic_vein = 15.0e-9  # vein compliance (m6_per_J)
        # self.C_systemic_vein = 1.99  # vein compliance (m6_per_J)(mL_per_mmHg(m6_per_J*133322368.421))

        # self.R_systemic_vein = 1.333e+6   # vein resistance (Js_per_m6)
        self.R_systemic_vein = 14.2e+7   # vein resistance (Js_per_m6)
        # self.R_systemic_vein = 1.13  # vein resistance (Js_per_m6)(mmHgs_per_mL(Js_per_m6*0.000000008))




        self.T = 1  # duration of a cardiac cycle in normal case (second)
        # self.T = 0.5  # duration of a cardiac cycle in exercise(second)

        
        
        ######INITIAL CONDITIONS####################
        # Initial value for v_ra, v_rv, v_la, and v_lv
        self.V_ra_initial = 20.0e-6  # blood volume of RA (m3)
        # self.V_ra_initial = 20  # blood volume of RA (mL(m3*1000000))
        self.V_rv_initial = 500.0e-6 # blood volume of RV (m3)
        # self.V_rv_initial = 500 # blood volume of RV (mL(m3*1000000))
        # self.V_la_initial = 20.0e-6  # blood volume of LA (m3)
        self.V_la_initial = 20.0e-6  # blood volume of LA (m3)
        # self.V_la_initial = 20  # blood volume of LA (mL(m3*1000000))
        self.V_lv_initial = 500.0e-6 # blood volume of LV (m3)
        # self.V_lv_initial = 500 # blood volume of LV (mL(m3*1000000))
        # self.P_pulmonary_artery_initial = 4000.  # pulmonary artery pressure (J_per_m3)(model value) assume by myself
        self.P_pulmonary_artery_initial = 1000.  # pulmonary artery pressure (J_per_m3)(model value)
        # self.P_pulmonary_artery_initial = 2000.  # pulmonary artery pressure (J_per_m3)(model value)
        # self.P_pulmonary_artery_initial = 30.  # pulmonary artery pressure (mmHg)(model value)
        # self.P_pulmonary_artery_initial = 1946.  # pulmonary artery pressure (J_per_m3)(exercise value)

        # self.P_pulmonary_vein_initial = 2000.    # pulmonary vein pressure (J_per_m3) assumed by myself
        self.P_pulmonary_vein_initial = 500.    # pulmonary vein pressure (J_per_m3)
        # self.P_pulmonary_vein_initial = 1000.    # pulmonary vein pressure (J_per_m3)
        # self.P_pulmonary_vein_initial = 15.    # pulmonary vein pressure (mmHg)(model value)
        self.Q_pulmonary_artery_initial = 0. # pulmonary artery flow (m3_per_s)
        self.Q_pulmonary_vein_initial = 0.   # pulmonary vein flow (m3_per_s)

        self.P_systemic_artery_initial = 2000  # pulmonary artery pressure (J_per_m3)
        # self.P_systemic_artery_initial = 15  # pulmonary artery pressure (mmHg)

        # self.P_systemic_artery_initial = 13332  # pulmonary artery pressure (J_per_m3)
        # self.P_systemic_artery_initial = 100  # pulmonary artery pressure (mmHg)(model value)
        self.P_systemic_vein_initial = 0.  # pulmonary vein pressure (J_per_m3)
        self.Q_systemic_artery_initial = 0.  # pulmonary artery flow (m3_per_s)
        self.Q_systemic_vein_initial = 0.  # pulmonary vein flow (m3_per_s)
        
        
        self.V_ra = self.V_ra_initial 
        self.V_rv = self.V_rv_initial 
        self.V_la = self.V_la_initial 
        self.V_lv = self.V_lv_initial 
        self.P_pulmonary_artery = self.P_pulmonary_artery_initial 
        self.P_pulmonary_vein = self.P_pulmonary_vein_initial 
        self.Q_pulmonary_artery = self.Q_pulmonary_artery_initial 
        self.Q_pulmonary_vein = self.Q_pulmonary_vein_initial

        self.P_systemic_artery = self.P_systemic_artery_initial
        self.P_systemic_vein = self.P_systemic_vein_initial
        self.Q_systemic_artery = self.Q_systemic_artery_initial
        self.Q_systemic_vein = self.Q_systemic_vein_initial
        
        


    def define_t_array(self, a, b, num_int):
        ''' This function considers the time as an array '''
        self.t = np.linspace(a, b, num_int)

    def activation_ventricles(self, mt):
        '''  This function calculates the activation ventricles '''
        if (mt >= 0) and (mt <= self.T_vc * self.T):
            self.e_v = 0.5 * (1 - cos(pi * mt / (self.T_vc * self.T)))
        elif (mt > self.T_vc * self.T) and (mt <= (self.T_vc + self.T_vr) * self.T):
            self.e_v = 0.5 * (1 + cos(pi * (mt - self.T_vc * self.T) / (self.T_vr * self.T)))
        elif (mt > (self.T_vc + self.T_vr) * self.T) and (mt < self.T):
            self.e_v = 0

        return self.e_v

    def activation_atria(self, mt):
        ''' This function calculates the  activation atria '''
        if (mt >= 0) and (mt <= (self.t_ar + self.T_ar) * self.T - self.T):
            self.e_a = 0.5 * (1 + cos(pi * (mt + self.T - self.t_ar * self.T) / (self.T_ar * self.T)))
        elif (mt > (self.t_ar + self.T_ar) * self.T - self.T) and (mt <= self.t_ac * self.T):
            self.e_a = 0
        elif (mt > self.t_ac * self.T) and (mt <= (self.t_ac + self.T_ac) * self.T):
            self.e_a = 0.5 * (1 - cos(pi * (mt - self.t_ac * self.T) / (self.T_ac * self.T)))
        elif (mt > (self.t_ac + self.T_ac) * self.T) and (mt <= self.T):
            self.e_a = 0.5 * (1 + cos(pi * (mt - self.t_ar * self.T) / (self.T_ar * self.T)))

        return self.e_a

    def Blood_Press_Atria_Ventricles(self, V_ra, V_rv, V_la, V_lv):
        ''' This function calculates the blood pressure in atria and ventricles '''

        self.P_ra = (self.e_a * self.E_ra_A + self.E_ra_B) * (V_ra - self.V_ra_0)

        # RV pressure
        self.P_rv = (self.e_v * self.E_rv_A + self.E_rv_B) * (V_rv - self.V_rv_0)

        # LA pressure
        self.P_la = (self.e_a * self.E_la_A + self.E_la_B) * (V_la - self.V_la_0)

        # LV pressure
        self.P_lv = (self.e_v * self.E_lv_A + self.E_lv_B) * (V_lv - self.V_lv_0)

        return (self.P_ra, self.P_rv, self.P_la, self.P_lv)

    def Blood_Flow_Atria_ventricles(self):
        '''   This function calculates the blood flow in atria and ventricles '''

        if self.P_ra >= self.P_rv:
            self.Q_ra = self.CQ_trv * sqrt(self.P_ra - self.P_rv)

        else:
            self.Q_ra = 0.

        # RV blood flow
        if self.P_rv >= self.P_pulmonary_artery:  # self.P_pulmonary_artery 
            self.Q_rv = self.CQ_puv * sqrt(self.P_rv - self.P_pulmonary_artery)
        else:
            self.Q_rv = 0.

        # LA blood flow
        if self.P_la >= self.P_lv:
            self.Q_la = self.CQ_miv * sqrt(self.P_la - self.P_lv)
        else:
            self.Q_la = 0.

        # LV blood flow
        if self.P_lv >= self.P_systemic_artery:
            self.Q_lv = self.CQ_aov * sqrt(self.P_lv - self.P_systemic_artery)
        else:
            self.Q_lv = 0.
        return (self.Q_ra, self.Q_rv, self.Q_la, self.Q_lv)


    def Blood_Volume_changes_Atria_ventricles(self):
        ''' This function calculates the blood volume changes in atria and ventricles '''

        # blood volume changes in RA
        # self.der_volume_RA = self.Q_sup_venacava + self.Q_inf_venacava - self.Q_ra
        self.der_volume_RA = self.Q_systemic_vein - self.Q_ra


        # blood volume changes in RV
        self.der_volume_RV = self.Q_ra - self.Q_rv

        # blood volume changes of LA
        self.der_volume_LA = self.Q_pulmonary_vein - self.Q_la

        # blood volume changes of LV
        self.der_volume_LV = self.Q_la - self.Q_lv
        

        return (self.der_volume_RA, self.der_volume_RV, self.der_volume_LA, self.der_volume_LV)
        
    def finite_difference_update(self,start_time,end_time,derivative,start_value):
    
         end_value = start_value + (end_time-start_time)*derivative 
    
         return end_value


    def Blood_Volume_Atria_ventricles_FiniteMethod(self, a, b, der_volume_RA, der_volume_RV, der_volume_LA, der_volume_LV, V_ra,
            V_rv, V_la, V_lv):
        ''' This function calculates the blood volume changes in atria and ventricles '''

        # blood volume changes in RA
        self.V_ra = (b - a) * der_volume_RA + V_ra

        # blood volume changes in RV
        self.V_rv = (b - a) * der_volume_RV + V_rv

        # blood volume changes of LA
        self.V_la = (b - a) * der_volume_LA + V_la

        # blood volume changes of LV
        self.V_lv = (b - a) * der_volume_LV + V_lv

        return (self.V_ra, self.V_rv, self.V_la, self.V_lv)

    def pulmonary_circulation(self):
        ''' This function calculates the blood pressure and volume changes in artery and vein in pulmonary circulation'''

        self.der_pulmonary_press_artery = (self.Q_rv - self.Q_pulmonary_artery) / self.C_pulmonary_artery  # pressure changes in artery
        
        # print('pulm circulation1',self.Q_rv,self.Q_pulmonary_artery,self.C_pulmonary_artery,self.der_pulmonary_press_artery)

        self.der_pulmonary_press_vein = (self.Q_pulmonary_artery - self.Q_pulmonary_vein) / self.C_pulmonary_vein  # pressure changes in vein
        
        # print('pulm circulation2',self.Q_pulmonary_artery,self.Q_pulmonary_vein,self.C_pulmonary_vein,self.der_pulmonary_press_vein)
        # self.der_pulmonary_flow_artery = (self.P_pulmonary_artery - self.P_pulmonary_vein - self.Q_pulmonary_artery * self.R_pulmonary_artery) / self.I_pulmonary_artery  # flow changes in artery
        # self.der_pulmonary_flow_artery = (self.P_pulmonary_artery - self.P_pulmonary_vein ) / self.R_pulmonary_artery  # flow changes in artery


        # print('pulm circulation3',self.P_pulmonary_artery,self.P_pulmonary_vein,self.Q_pulmonary_artery,self.R_pulmonary_artery,self.I_pulmonary_artery,self.der_pulmonary_flow_artery)
        # self.der_pulmonary_flow_vein = (self.P_pulmonary_vein - self.P_la - self.Q_pulmonary_vein * self.R_pulmonary_vein) / self.I_pulmonary_vein  # flow changes in vein
        # self.der_pulmonary_flow_vein = (self.P_pulmonary_vein - self.P_la ) / self.R_pulmonary_vein  # flow changes in vein

        #print('pulm circulation4',self.P_pulmonary_vein,self.P_la,self.Q_pulmonary_vein,self.R_pulmonary_vein,self.I_pulmonary_vein,self.der_pulmonary_flow_artery)

        return (self.der_pulmonary_press_artery, self.der_pulmonary_press_vein)#, self.der_pulmonary_flow_artery, self.der_pulmonary_flow_vein)



    def Finite_diff_meth_Pulmonary_press_artery(self, *args):

        ''' This function calculate the Pulmonary artery pressure with using previous points. If we want to calculate Pulmonary artery pressure for i=1, we use
        the previous point. If we want to calculate the Pulmonary artery pressure for i=2, two previous points are required and if we intend to calulate Pulmonary artery pressure
        for t>=3, we use the 3 points before to calculate the new point
        if i==1
            P_pulmonary_artery[i] = P_pulmonary_artery[i-1] + h * der_pulmonary_press_artery[i-1]
        if i==2:
            P_pulmonary_artery[i] = (P_pulmonary_artery[i-2] + P_pulmonary_artery[i-1] + (3 * h * der_pulmonary_press_artery[i-1])) / 2
        if i>= 3:
            P_pulmonary_artery[i] = (P_pulmonary_artery[i-3] + P_pulmonary_artery[i-2] + P_pulmonary_artery[i-1] + (6 * h * der_pulmonary_press_artery[i-1])) / 3
        '''

        # Calculate P_pulmonary_artery from previous point
        if args[0] == 1:
            self.P_pulmonary_artery = args[3] + ((args[2] - args[1]) * args[4])
        # Calculate P_pulmonary_artery from two last previous points
        elif args[0] == 2:
            self.P_pulmonary_artery = (args[3] + args[4] + (3 * (args[2] - args[1]) * args[5]))/2
        # Calculate P_pulmonary_artery from three last previous points
        else:
            self.P_pulmonary_artery = (args[3] + args[4] + args[5] + (6 * (args[2] - args[1]) * args[6]))/3
        # self.P_pulmonary_artery = args[2] + ((args[1] - args[0]) * args[3])
        return self.P_pulmonary_artery

    def Finite_diff_meth_Pulmonary_press_vein(self, *args):
        ''' This function calculate the Pulmonary vein pressure with using previous points. If we want to calculate Pulmonary vein pressure for i=1, we use
            the previous point. If we want to calculate the Pulmonary vein pressure for i=2, two previous points are required and if we intend to calulate Pulmonary vein pressure
            for t>=3, we use the 3 points before to calculate the new point
            if i==1
                P_pulmonary_vein[i] = P_pulmonary_vein[i-1] + h * der_pulmonary_press_vein[i-1]
            if i==2:
                P_pulmonary_vein[i] = (P_pulmonary_vein[i-2] + P_pulmonary_vein[i-1] + (3 * h * der_pulmonary_press_vein[i-1])) / 2
            if i>= 3:
                P_pulmonary_vein[i] = (P_pulmonary_vein[i-3] + P_pulmonary_vein[i-2] + P_pulmonary_vein[i-1] + (6 * h * der_pulmonary_press_vein[i-1])) / 3
            '''

        if args[0] == 1:
            self.P_pulmonary_vein = args[3] + (
                        (args[2] - args[1]) * args[4])
        # Calculate P_pulmonary_vein from two last previous points
        elif args[0] == 2:
            self.P_pulmonary_vein = (args[3] + args[4] + (3 * (args[2] - args[1]) * args[
                5])) / 2
        # Calculate P_pulmonary_vein from three last previous points
        else:
            self.P_pulmonary_vein = (args[3] + args[4] + args[5] + (6 * (args[2] - args[1]) * args[6])) / 3
        # self.P_pulmonary_vein = args[2] + ((args[1] - args[0]) * args[3])
        return self.P_pulmonary_vein

    # def Finite_diff_meth_Pulmonary_flow_artery(self, *args):
    #
    #     ''' This function calculate the Pulmonary artery flow with using previous points. If we want to calculate Pulmonary artery flow for t=1, we use
    #     the previous point. If we want to calculate the Pulmonary artery flow for t=2, two previous points are required and if we intend to calulate Pulmonary artery flow
    #     for t>=3, we use the 3 points before to calculate the new point
    #      if i==1
    #             Q_pulmonary_artery[i] = Q_pulmonary_artery[i-1] + h * der_pulmonary_flow_artery[i-1]
    #         if i==2:
    #             Q_pulmonary_artery[i] = (Q_pulmonary_artery[i-2] + Q_pulmonary_artery[i-1] + (3 * h * der_pulmonary_flow_artery[i-1])) / 2
    #         if i>= 3:
    #             Q_pulmonary_artery[i] = (Q_pulmonary_artery[i-3] + Q_pulmonary_artery[i-2] + Q_pulmonary_artery[i-1] + (6 * h * der_pulmonary_flow_artery[i-1])) / 3
    #         '''
    #
    #     # Calculate Q_pulmonary_artery from previous point
    #     if args[0] == 1:
    #         self.Q_pulmonary_artery = args[3] + (
    #                     (args[2] - args[1]) * args[4])
    #     # Calculate Q_pulmonary_artery from two last previous points
    #     elif args[0] == 2:
    #         self.Q_pulmonary_artery = (args[3] + args[4] + (3 * (args[2] - args[1]) * args[
    #             5])) / 2
    #     # Calculate Q_pulmonary_artery from three last previous points
    #     else:
    #         self.Q_pulmonary_artery = (args[3] + args[4] + args[5] + (6 * (args[2] - args[1]) * args[6])) / 3
    #     # self.Q_pulmonary_artery = args[2] + ((args[1] - args[0]) * args[3])
    #
    #     return self.Q_pulmonary_artery
    #
    # def Finite_diff_meth_Pulmonary_flow_vein(self, *args):
    #     ''' This function calculate the Pulmonary vein flow with using previous points. If we want to calculate Pulmonary vein flow for t=1, we use
    #          the previous point. If we want to calculate the Pulmonary vein flow for t=2, two previous points are required and if we intend to calulate Pulmonary vein flow
    #          for t>=3, we use the 3 points before to calculate the new point
    #           if i==1
    #                  Q_pulmonary_vein[i] = Q_pulmonary_vein[i-1] + h * der_pulmonary_flow_vein[i-1]
    #           if i==2:
    #                  Q_pulmonary_vein[i] = (Q_pulmonary_vein[i-2] + Q_pulmonary_vein[i-1] + (3 * h * der_pulmonary_flow_vein[i-1])) / 2
    #           if i>= 3:
    #                  Q_pulmonary_vein[i] = (Q_pulmonary_vein[i-3] + Q_pulmonary_vein[i-2] + Q_pulmonary_vein[i-1] + (6 * h * der_pulmonary_flow_vein[i-1])) / 3
    #              '''
    #
    #     # Calculate Q_pulmonary_vein from previous point
    #     if args[0] == 1:
    #         self.Q_pulmonary_vein = args[3] + (
    #                     (args[2] - args[1]) * args[4])
    #
    #     # Calculate Q_pulmonary_vein from two last previous points
    #     elif args[0] == 2:
    #         self.Q_pulmonary_vein = (args[3] + args[4] + (3 * (args[2] - args[1]) * args[
    #             5])) / 2
    #     # Calculate Q_pulmonary_vein from three last previous points
    #     else:
    #         self.Q_pulmonary_vein = (args[3] + args[4] + args[5] + (6 * (args[2] - args[1]) * args[6])) / 3
    #     # self.Q_pulmonary_vein = args[2] + ((args[1] - args[0]) * args[3])
    #
    #     return self.Q_pulmonary_vein


    def Pulmonary_flow_vein_artery(self):
        ''' This function calculates the blood flow in pulmonary artery and vein based on resistance'''

        self.Q_pulmonary_artery = (self.P_pulmonary_artery - self.P_pulmonary_vein) / self.R_pulmonary_artery  # blood flow in artery
        self.Q_pulmonary_vein = (self.P_pulmonary_vein - self.P_la ) / self.R_pulmonary_vein  # blood flow in vein

        return (self.Q_pulmonary_artery, self.Q_pulmonary_vein)

    def systemic_circulation(self):
        ''' This function calculates the blood pressure and volume changes in artery and vein in systemic circulation'''

        self.der_systemic_press_artery = (self.Q_lv - self.Q_systemic_artery) / self.C_systemic_artery  # pressure changes in artery

        self.der_systemic_press_vein = ( self.Q_systemic_artery - self.Q_systemic_vein) / self.C_systemic_vein  # pressure changes in vein


        return (self.der_systemic_press_artery, self.der_systemic_press_vein)#, self.der_systemic_flow_artery)#, self.der_pulmonary_flow_vein)

    def Finite_diff_meth_systemic_press_artery(self, *args):

        ''' This function calculate the Pulmonary artery pressure with using previous points. If we want to calculate Pulmonary artery pressure for i=1, we use
        the previous point. If we want to calculate the Pulmonary artery pressure for i=2, two previous points are required and if we intend to calulate Pulmonary artery pressure
        for t>=3, we use the 3 points before to calculate the new point
        if i==1
            P_pulmonary_artery[i] = P_pulmonary_artery[i-1] + h * der_pulmonary_press_artery[i-1]
        if i==2:
            P_pulmonary_artery[i] = (P_pulmonary_artery[i-2] + P_pulmonary_artery[i-1] + (3 * h * der_pulmonary_press_artery[i-1])) / 2
        if i>= 3:
            P_pulmonary_artery[i] = (P_pulmonary_artery[i-3] + P_pulmonary_artery[i-2] + P_pulmonary_artery[i-1] + (6 * h * der_pulmonary_press_artery[i-1])) / 3
        '''

        # Calculate P_pulmonary_artery from previous point
        if args[0] == 1:
            self.P_systemic_artery = args[3] + ((args[2] - args[1]) * args[4])
        # Calculate P_pulmonary_artery from two last previous points
        elif args[0] == 2:
            self.P_systemic_artery = (args[3] + args[4] + (3 * (args[2] - args[1]) * args[5]))/2
        # Calculate P_pulmonary_artery from three last previous points
        else:
            self.P_systemic_artery = (args[3] + args[4] + args[5] + (6 * (args[2] - args[1]) * args[6]))/3
        # self.P_pulmonary_artery = args[2] + ((args[1] - args[0]) * args[3])
        return self.P_systemic_artery

    def Finite_diff_meth_systemic_press_vein(self, *args):
        ''' This function calculate the Pulmonary vein pressure with using previous points. If we want to calculate Pulmonary vein pressure for i=1, we use
            the previous point. If we want to calculate the Pulmonary vein pressure for i=2, two previous points are required and if we intend to calulate Pulmonary vein pressure
            for t>=3, we use the 3 points before to calculate the new point
            if i==1
                P_pulmonary_vein[i] = P_pulmonary_vein[i-1] + h * der_pulmonary_press_vein[i-1]
            if i==2:
                P_pulmonary_vein[i] = (P_pulmonary_vein[i-2] + P_pulmonary_vein[i-1] + (3 * h * der_pulmonary_press_vein[i-1])) / 2
            if i>= 3:
                P_pulmonary_vein[i] = (P_pulmonary_vein[i-3] + P_pulmonary_vein[i-2] + P_pulmonary_vein[i-1] + (6 * h * der_pulmonary_press_vein[i-1])) / 3
            '''

        if args[0] == 1:
            self.P_systemic_vein = args[3] + (
                        (args[2] - args[1]) * args[4])
        # Calculate P_pulmonary_vein from two last previous points
        elif args[0] == 2:
            self.P_systemic_vein = (args[3] + args[4] + (3 * (args[2] - args[1]) * args[
                5])) / 2
        # Calculate P_pulmonary_vein from three last previous points
        else:
            self.P_systemic_vein = (args[3] + args[4] + args[5] + (6 * (args[2] - args[1]) * args[6])) / 3
        # self.P_pulmonary_vein = args[2] + ((args[1] - args[0]) * args[3])
        return self.P_systemic_vein

    def Finite_diff_meth_systemic_flow_artery(self, *args):
        ''' This function calculate the Pulmonary artery flow with using previous points. If we want to calculate Pulmonary artery flow for t=1, we use
        the previous point. If we want to calculate the Pulmonary artery flow for t=2, two previous points are required and if we intend to calulate Pulmonary artery flow
        for t>=3, we use the 3 points before to calculate the new point
         if i==1
                Q_pulmonary_artery[i] = Q_pulmonary_artery[i-1] + h * der_pulmonary_flow_artery[i-1]
            if i==2:
                Q_pulmonary_artery[i] = (Q_pulmonary_artery[i-2] + Q_pulmonary_artery[i-1] + (3 * h * der_pulmonary_flow_artery[i-1])) / 2
            if i>= 3:
                Q_pulmonary_artery[i] = (Q_pulmonary_artery[i-3] + Q_pulmonary_artery[i-2] + Q_pulmonary_artery[i-1] + (6 * h * der_pulmonary_flow_artery[i-1])) / 3
            '''

        # Calculate Q_pulmonary_artery from previous point
        if args[0] == 1:
            self.Q_systemic_artery = args[3] + (
                        (args[2] - args[1]) * args[4])
        # Calculate Q_pulmonary_artery from two last previous points
        elif args[0] == 2:
            self.Q_systemic_artery = (args[3] + args[4] + (3 * (args[2] - args[1]) * args[
                5])) / 2
        # Calculate Q_pulmonary_artery from three last previous points
        else:
            self.Q_systemic_artery = (args[3] + args[4] + args[5] + (6 * (args[2] - args[1]) * args[6])) / 3
        # self.Q_pulmonary_artery = args[2] + ((args[1] - args[0]) * args[3])

        return self.Q_systemic_artery

    def systemic_flow_vein_artery(self):
        ''' This function calculates the blood pressure and volume changes in artery and vein in systemic circulation'''
        self.Q_systemic_artery = (self.P_systemic_artery - self.P_systemic_vein) / self.R_systemic_artery  # pressure changes in artery
        self.Q_systemic_vein = (self.P_systemic_vein - self.P_ra) / self.R_systemic_vein  # pressure changes in artery

        return (self.Q_systemic_artery, self.Q_systemic_vein)


def perfusion(P_rv,P_la,count):
    set_diagnostics_on(False)

    export_directory = 'output'

    if not os.path.exists(export_directory):
        os.makedirs(export_directory)

    # define model geometry and indices
    perfusion_indices()
    if count == 1:
        # Read in geometry files
        define_node_geometry('../geometry/P2BRP268-H12816_Artery_Full.ipnode')
        define_1d_elements('../geometry/P2BRP268-H12816_Artery_Full.ipelem')
        append_units()
        add_matching_mesh()

    # define radius by Strahler order
    mpa_rad = 11.2  # main pulmonary artery radius, needs to be unstrained, so if defining from CT scale back to zero pressure
    s_ratio = 1.54  # straheler diameter ratio
    order_system = 'strahler'
    order_options = 'all'
    name = 'inlet'
    define_rad_from_geom(order_system, s_ratio, name, mpa_rad, order_options, '')

    s_ratio_ven = 1.56
    inlet_rad_ven = 14.53
    order_system = 'strahler'
    order_options = 'list'
    name = 'inlet'
    define_rad_from_geom(order_system, s_ratio_ven, '61361', inlet_rad_ven, order_options, '122720')  # matched arteries

    ##Call solve
    mesh_type = 'full_plus_ladder'
    vessel_type = 'elastic_g0_beta'
    grav_dirn = 2
    grav_factor = 1.0
    bc_type = 'pressure'
    inlet_bc = P_rv  #flow     #749.7438405387509
    outlet_bc = P_la      # 1161.738365326976

    evaluate_prq(mesh_type, vessel_type, grav_dirn, grav_factor, bc_type, inlet_bc, outlet_bc)

    ##export geometry
    group_name = 'perf_model'
    filename = export_directory + '/P2BRP268-H12816_Artery.exelem'
    export_1d_elem_geometry(filename, group_name)
    filename = export_directory + '/P2BRP268-H12816_Artery.exnode'
    export_node_geometry(filename, group_name)

    # export element field for radius
    field_name = 'radius_perf'
    ne_radius = get_ne_radius()
    filename = export_directory + '/P2BRP268-H12816_radius_perf.exelem'
    export_1d_elem_field(ne_radius, filename, name, field_name)

    # export flow element
    filename = export_directory + '/P2BRP268-H12816_flow_perf.exelem'
    field_name = 'flow'
    export_1d_elem_field(7, filename, group_name, field_name)

    # export node field for pressure
    filename = export_directory + '/P2BRP268-H12816_pressure_perf.exnode'
    field_name = 'pressure_perf'
    export_node_field(1, filename, group_name, field_name)

    # Export terminal solution
    filename = export_directory + '/P2BRP268-H12816_terminal.exnode'
    export_terminal_perfusion(filename, group_name)

    shutil.move('micro_flow_unit.out', export_directory + '/micro_flow_unit.out')
    shutil.move('micro_flow_ladder.out', export_directory + '/micro_flow_ladder.out')
    ne_radius = []

def read_file(
        path="/home/arah607/lung-group-examples/perfusion_Clark2011/output/" + "P2BRP268-H12816_flow_perf.exelem"):
    '''This function read the txt file from output of perfusion_clark2011 code '''
    flow_file = open(path)
    read_line = [10]  # This line shows the 11th line value in P2BRP268-H12816_flow_perf.exelem (output)

    for position, line in enumerate(flow_file):  # position shows the row and line shows the value in that row
        if position in read_line:
            flow = line
            break

    flow = flow.split()  # if the value has some strings, this line slipt them
    flow = float(flow[0])
    # flow = flow.split(" ")
    return flow

def plot_results(dic_arrays):
    ''' This function plot the result found solving systems'''

    for i in range(len(dic_arrays)):
        dic_vals = dic_arrays[i]['function'].items()
        x, y = zip(*dic_vals)
        split_num=len(y)
        
        x = np.linspace(5, 10, 50000)

        plt.plot(x, y, label=dic_arrays[i]['function_identity'])
    plt.xlabel(dic_arrays[0]['x_lable']) # write x lable based on the first dic of array
    plt.ylabel(dic_arrays[0]['y_lable']) #  write y lable based on the first dic of array

    plt.legend(loc = "upper right")
    plt.show()

    # if dic_array_2 == {}:
    #     dic_vals = dic_array.items()
    #     x, y = zip(*dic_vals)
    #     plt.plot(x, y)
    #     plt.xlabel(x_lable)
    #     plt.ylabel(y_lable)
    #     plt.title(plot_title)
    # else:
    #     dic_vals = dic_array.items()
    #     x, y = zip(*dic_vals)
    #     dic_vals_2 = dic_array_2.items()
    #     x_2, y_2 = zip(*dic_vals_2)
    #     plt.plot(x, y)
    #     plt.plot(x_2, y_2)
    #     plt.xlabel(x_lable)
    #     plt.ylabel(y_lable)
    #     plt.title(plot_title)
    #
    # # plt.legend()
    #
    # plt.show()


def print_results(*arg):
    ''' This function print output values'''
    print("e_a:" + str([(k, arg[0][k]) for k in arg[0]]))
    print("e_v:" + str([(k, arg[1][k]) for k in arg[1]]))

    print("P_ra" + str([(k, arg[2][k]) for k in arg[2]]))
    print("P_rv:" + str([(k, arg[3][k]) for k in arg[3]]))
    print("P_la:" + str([(k, arg[4][k]) for k in arg[4]]))
    print("P_lv:" + str([(k, arg[5][k]) for k in arg[5]]))

    print("Q_la:" + str([(k, arg[6][k]) for k in arg[6]]))
    print("Q_lv:" + str([(k, arg[7][k]) for k in arg[7]]))
    print("Q_ra:" + str([(k, arg[8][k]) for k in arg[8]]))
    print("Q_rv:" + str([(k, arg[9][k]) for k in arg[9]]))

    print("der_volume_RA:" + str([(k, arg[10][k]) for k in arg[10]]))
    print("der_volume_RV:" + str([(k, arg[11][k]) for k in arg[11]]))
    print("der_volume_LA:" + str([(k, arg[12][k]) for k in arg[12]]))
    print("der_volume_LV:" + str([(k, arg[13][k]) for k in arg[13]]))

    print("V_ra:" + str([(k, arg[14][k]) for k in arg[14]]))
    print("V_rv:" + str([(k, arg[15][k]) for k in arg[15]]))
    print("V_la:" + str([(k, arg[16][k]) for k in arg[16]]))
    print("V_lv:" + str([(k, arg[17][k]) for k in arg[17]]))

    print("der_systemic_press_artery:" + str([(k, arg[18][k]) for k in arg[18]]))
    print("der_systemic_press_vein:" + str([(k, arg[19][k]) for k in arg[19]]))
    # print("der_pulmonary_flow_artery:" + str([(k, arg[20][k]) for k in arg[20]]))
    # print("der_pulmonary_flow_vein:" + str([(k, arg[21][k]) for k in arg[21]]))

    print("P_systemic_artery:" + str([(k, arg[20][k]) for k in arg[20]]))
    print("P_systemic_vein:" + str([(k, arg[21][k]) for k in arg[21]]))
    print("Q_systemic_artery:" + str([(k, arg[22][k]) for k in arg[22]]))
    print("Q_systemic_vein:" + str([(k, arg[23][k]) for k in arg[23]]))

    print("der_pulmonary_press_artery:" + str([(k, arg[24][k]) for k in arg[24]]))
    print("der_pulmonary_press_vein:" + str([(k, arg[25][k]) for k in arg[25]]))
    # print("der_pulmonary_flow_artery:" + str([(k, arg[20][k]) for k in arg[20]]))
    # print("der_pulmonary_flow_vein:" + str([(k, arg[21][k]) for k in arg[21]]))

    # print("P_pulmonary_artery:" + str([(k, arg[22][k]) for k in arg[22]]))
    # print("P_pulmonary_vein:" + str([(k, arg[23][k]) for k in arg[23]]))
    # print("Q_pulmonary_artery:" + str([(k, arg[24][k]) for k in arg[24]]))
    # print("Q_pulmonary_vein:" + str([(k, arg[25][k]) for k in arg[25]]))

    print("P_pulmonary_artery:" + str([(k, arg[26][k]) for k in arg[26]]))
    print("P_pulmonary_vein:" + str([(k, arg[27][k]) for k in arg[27]]))
    print("Q_pulmonary_artery:" + str([(k, arg[28][k]) for k in arg[28]]))
    print("Q_pulmonary_vein:" + str([(k, arg[29][k]) for k in arg[29]]))
def max_min_values(array,array_name):
    array=array.values()
    min_val= min(array)
    max_val=max(array)
    print('the maximum value' + array_name + 'is : ' + str(max_val))
    print('the min value' + array_name + 'is : ' + str(min_val))





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # define an object regarding heart_lung class
    heart_lung_obj = Heart_Lung()

    # activation_ventricles and activation_atria
    e_v = {}
    e_a = {}

    # blood_press_Atria_ventricles
    P_ra = {}
    P_rv = {}
    P_la = {}
    P_lv = {}

    # blood_flow_atria_ventricles
    Q_ra = {}
    Q_rv = {}
    Q_la = {}
    Q_lv = {}

    # blood_volume_changes_Atria_ventricles
    der_volume_RA = {}
    der_volume_RV = {}
    der_volume_LA = {}
    der_volume_LV = {}

    # # blood_volume_Atria_ventricles
    V_ra = {}
    V_rv = {}
    V_la = {}
    V_lv = {}

    # pulmonary circulation
    der_pulmonary_press_artery = {}
    der_pulmonary_press_vein = {}
    der_pulmonary_flow_artery = {}
    # der_pulmonary_flow_vein = {}

    # systemic circulation
    der_systemic_press_artery = {}
    der_systemic_press_vein = {}
    # der_systemic_flow_artery = {}
    # der_systemic_flow_vein = {}
    #
    # pulmonary pressure and flow
    P_pulmonary_artery = {}
    P_pulmonary_vein = {}
    Q_pulmonary_artery = {}
    Q_pulmonary_vein = {}

    # systemic pressure and flow
    P_systemic_artery = {}
    P_systemic_vein = {}
    Q_systemic_artery = {}
    Q_systemic_vein = {}

    # receive an the first and the last element of interval and number of equal parts from user to create an array
    first_int = input("please enter the first value of interval [0]:") #First time in time interval
    end_int = input("please enter the last value of interval [1]:") #Last time in time interval
    num_split = input("please enter the number of splitting interval [50]:") #discretisation
    
    #Set inputs to integers or default values if not specified
    if first_int == '':
        first_int = 0
    else:
        first_int = float(first_int) #ARC: as these are times i think that these should be real numbers
    if end_int == '':
        end_int = 1
    else:
        end_int = float(end_int)
    if num_split == '':
        num_split = 50
    else:
        num_split = int(num_split)
        
    #Define overall time array heart_lung_obj.t
    heart_lung_obj.define_t_array(first_int, end_int, num_split)
    
    #print(heart_lung_obj.t)
    # with open('heart-persion-results.csv', 'w', newline='') as write_obj:
    #     csv_writer = writer(write_obj)
    value = []
    for i in range(0, len(heart_lung_obj.t)):
      with open('heart-persion-results.csv', 'w', newline='') as write_obj:
        csv_writer = writer(write_obj)
        #stonep through time
        if i == 0:  # In this loop we want to run the heart model
            # define time parameter
            mt = heart_lung_obj.t[i] - heart_lung_obj.T * floor(heart_lung_obj.t[i] / heart_lung_obj.T)

            # Calculate activation ventricles value regarding its time
            e_v[i] = heart_lung_obj.activation_ventricles(mt)

            # Calculate activation atria value regarding its time
            e_a[i] = heart_lung_obj.activation_atria(mt)
            

            # Initialise parameters that will have derivative updates 
            V_ra[i] = heart_lung_obj.V_ra_initial
            V_rv[i] = heart_lung_obj.V_rv_initial
            V_la[i] = heart_lung_obj.V_la_initial
            V_lv[i] = heart_lung_obj.V_lv_initial

            # Initial values for P_pulmonary_artery, P_pulmonary_vein, Q_pulmonary_artery, and Q_pulmonary_vein
            P_pulmonary_artery[i] = heart_lung_obj.P_pulmonary_artery_initial
            P_pulmonary_vein[i] = heart_lung_obj.P_pulmonary_vein_initial
            Q_pulmonary_artery[i] = heart_lung_obj.Q_pulmonary_artery_initial
            Q_pulmonary_vein[i] = heart_lung_obj.Q_pulmonary_vein_initial

            P_systemic_artery[i] = heart_lung_obj.P_systemic_artery_initial
            P_systemic_vein[i] = heart_lung_obj.P_systemic_vein_initial
            Q_systemic_artery[i] = heart_lung_obj.Q_systemic_artery_initial
            Q_systemic_vein[i] = heart_lung_obj.Q_systemic_vein_initial


            # # Calculate initial blood pressure in atria and ventricles 
            [P_ra[i], P_rv[i], P_la[i], P_lv[i]] = heart_lung_obj.Blood_Press_Atria_Ventricles(V_ra[i], V_rv[i],
                                                                                               V_la[i], V_lv[i])

            # Calculate blood flow in atria and ventricles regarding their time
            [Q_ra[i], Q_rv[i], Q_la[i], Q_lv[i]] = heart_lung_obj.Blood_Flow_Atria_ventricles()

            # Calculate blood volume changes in atria and ventricles regarding their time
            [der_volume_RA[i], der_volume_RV[i], der_volume_LA[i],
             der_volume_LV[i]] = heart_lung_obj.Blood_Volume_changes_Atria_ventricles()

            # Calculate blood pressure and volume in systemic circulation regarding its time
            # [der_systemic_press_artery[i], der_systemic_press_vein[i]] = heart_lung_obj.systemic_circulation()

            # Calculate blood pressure and volume in pulmonary circulation regarding its time
            [der_pulmonary_press_artery[i], der_pulmonary_press_vein[i]] = heart_lung_obj.pulmonary_circulation()

            # P_ra.update({i: 0.00750061682704466 * P_ra[i] for i in P_ra.keys()})
            # P_rv.update({i: 0.00750061682704466 * P_rv[i] for i in P_rv.keys()})
            # P_la.update({i: 0.00750061682704466 * P_la[i] for i in P_la.keys()})
            # P_lv.update({i: 0.00750061682704466 * P_lv[i] for i in P_lv.keys()})

            # Q_ra.update({i: 1000000000 * Q_ra[i] for i in Q_ra.keys()})
            # Q_rv.update({i: 1000000000 * Q_rv[i] for i in Q_rv.keys()})
            # Q_la.update({i: 1000000000 * Q_la[i] for i in Q_la.keys()})
            # Q_lv.update({i: 1000000000 * Q_lv[i] for i in Q_lv.keys()})
            #
            # V_ra.update({i: 1000000000 * V_ra[i] for i in V_ra.keys()})
            # V_rv.update({i: 1000000000 * V_rv[i] for i in V_rv.keys()})
            # V_la.update({i: 1000000000 * V_la[i] for i in V_la.keys()})
            # V_lv.update({i: 1000000000 * V_lv[i] for i in V_lv.keys()})
            #
            # P_pulmonary_artery.update({i: 0.00750061682704466 * P_pulmonary_artery[i] for i in P_pulmonary_artery.keys()})
            # P_pulmonary_vein.update({i: 0.00750061682704466 * P_pulmonary_vein[i] for i in P_pulmonary_vein.keys()})
            # Q_pulmonary_artery.update({i: 1000000000 * Q_pulmonary_artery[i] for i in Q_pulmonary_artery.keys()})
            # Q_pulmonary_vein.update({i: 1000000000 * Q_pulmonary_vein[i] for i in Q_pulmonary_vein.keys()})



            headerList = ['e_a', 'e_v', 'V_la', 'V_lv', 'V_rv', 'V_ra', 'P_la', 'P_lv', 'P_rv', 'P_ra', 'Q_la', 'Q_lv', 'Q_rv', 'Q_ra', 'P_pulmonary_artery', 'P_pulmonary_vein', 'Q_pulmonary_artery', 'Q_pulmonary_vein']
            value.append(headerList)
            n0 = [e_a[i], e_v[i], V_la[i], V_lv[i], V_rv[i], V_ra[i], P_la[i], P_lv[i], P_rv[i], P_ra[i], Q_la[i], Q_lv[i], Q_rv[i], Q_ra[i], P_pulmonary_artery[i], P_pulmonary_vein[i], Q_pulmonary_artery[i], Q_pulmonary_vein[i]]
            value.append(n0)


        else:

            perfusion(P_rv[i - 1], P_la[i - 1], i)
            flow = read_file()
            print(flow)
            flow = flow * 0.000000001 # convert mm3/s to m3/s
            Q_pulmonary_vein = flow
            Q_pulmonary_artery = flow
            heart_lung_obj.Q_pulmonary_artery = float(flow)
            heart_lung_obj.Q_pulmonary_vein = float(flow)
            heart_lung_obj.P_pulmonary_artery = P_rv[i - 1]
            heart_lung_obj.P_pulmonary_vein = P_la[i - 1]
            # heart_lung_obj.P_systemic_artery = 15 #(mmHg)


            #heart_lung_obj.P_pulmonary_artery = P_pulmonary_artery
            #heart_lung_obj.P_pulmonary_vein = P_pulmonary_vein
            print(flow)


            mt = heart_lung_obj.t[i] - heart_lung_obj.T * floor(heart_lung_obj.t[i] / heart_lung_obj.T)
            # print(mt)

            # Calculate activation ventricles value regarding its time
            e_v[i] = heart_lung_obj.activation_ventricles(mt)

            # Calculate activation atria value regarding its time
            e_a[i] = heart_lung_obj.activation_atria(mt)

            #Calculate blood volume in atria and ventricles regarding their time
            [V_ra[i], V_rv[i], V_la[i], V_lv[i]] = heart_lung_obj.Blood_Volume_Atria_ventricles_FiniteMethod(heart_lung_obj.t[i - 1], heart_lung_obj.t[i],
                                                                                                der_volume_RA[i - 1],
                                                                                                der_volume_RV[i - 1],
                                                                                                der_volume_LA[i - 1],
                                                                                                der_volume_LV[i - 1],
                                                                                                V_ra[i - 1],
                                                                                                V_rv[i - 1],
                                                                                                V_la[i - 1],
                                                                                                V_lv[i - 1])
            heart_lung_obj.V_ra = V_ra[i]
            heart_lung_obj.V_rv = V_rv[i]
            heart_lung_obj.V_la = V_la[i]
            heart_lung_obj.V_lv = V_lv[i]




            # Calculate blood pressure in atria and ventricles regarding their time
            [P_ra[i], P_rv[i], P_la[i], P_lv[i]] = heart_lung_obj.Blood_Press_Atria_Ventricles(V_ra[i], V_rv[i],
                                                                                               V_la[i], V_lv[i])

            # # Calculate blood pressure in systemic arteries and veins regarding their time
            # if i == 1:
            #     P_systemic_artery[i] = heart_lung_obj.Finite_diff_meth_systemic_press_artery(i,
            #                                                                                    heart_lung_obj.t[i - 1],
            #                                                                                    heart_lung_obj.t[i],
            #                                                                                    P_systemic_artery[
            #                                                                                        i - 1],
            #                                                                                    der_systemic_press_artery[
            #                                                                                        i - 1])
            #     P_systemic_vein[i] = heart_lung_obj.Finite_diff_meth_systemic_press_vein(i, heart_lung_obj.t[i - 1],
            #                                                                                heart_lung_obj.t[i],
            #                                                                                P_systemic_vein[i - 1],
            #                                                                                der_systemic_press_vein[
            #                                                                                    i - 1])  # Q_pulmonary_artery[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_flow_artery(i,heart_lung_obj.t[i-1],heart_lung_obj.t[i], Q_pulmonary_artery[i-1], der_pulmonary_flow_artery[i-1])  # Q_pulmonary_vein[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_flow_vein(i,heart_lung_obj.t[i-1],heart_lung_obj.t[i], Q_pulmonary_vein[i-1], der_pulmonary_flow_vein[i-1])
            #     # Q_systemic_artery[i] = heart_lung_obj.Finite_diff_meth_systemic_flow_artery(i,heart_lung_obj.t[i-1],heart_lung_obj.t[i], Q_systemic_artery[i-1], der_systemic_flow_artery[i-1])
            #
            # elif i == 2:
            #     P_systemic_artery[i] = heart_lung_obj.Finite_diff_meth_systemic_press_artery(i,
            #                                                                                    heart_lung_obj.t[i - 1],
            #                                                                                    heart_lung_obj.t[i],
            #                                                                                    P_systemic_artery[
            #                                                                                        i - 2],
            #                                                                                    P_systemic_artery[
            #                                                                                        i - 1],
            #                                                                                    der_systemic_press_artery[
            #                                                                                        i - 1])
            #     P_systemic_vein[i] = heart_lung_obj.Finite_diff_meth_systemic_press_vein(i, heart_lung_obj.t[i - 1],
            #                                                                                heart_lung_obj.t[i],
            #                                                                                P_systemic_vein[i - 2],
            #                                                                                P_systemic_vein[i - 1],
            #                                                                                der_systemic_press_vein[
            #                                                                                    i - 1])  # Q_pulmonary_artery[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_flow_artery(i, heart_lung_obj.t[i - 1], heart_lung_obj.t[i], Q_pulmonary_artery[i - 2], Q_pulmonary_artery[i - 1],  #                                                                                   der_pulmonary_flow_artery[i - 1])  # Q_pulmonary_vein[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_flow_vein(i, heart_lung_obj.t[i - 1], heart_lung_obj.t[i], Q_pulmonary_vein[i - 2], Q_pulmonary_vein[i - 1],  #                                                                                   der_pulmonary_flow_vein[i - 1])
            #     # Q_systemic_artery[i] = heart_lung_obj.Finite_diff_meth_systemic_flow_artery(i,
            #     #                                                                               heart_lung_obj.t[i - 1],
            #     #                                                                               heart_lung_obj.t[i],
            #     #                                                                               Q_systemic_artery[i - 2],
            #     #                                                                               Q_systemic_artery[i - 1],
            #     #                                                                               der_systemic_flow_artery[
            #     #                                                                                   i - 1])
            #
            # else:
            #     P_systemic_artery[i] = heart_lung_obj.Finite_diff_meth_systemic_press_artery(i,
            #                                                                                    heart_lung_obj.t[i - 1],
            #                                                                                    heart_lung_obj.t[i],
            #                                                                                    P_systemic_artery[
            #                                                                                        i - 3],
            #                                                                                    P_systemic_artery[
            #                                                                                        i - 2],
            #                                                                                    P_systemic_artery[
            #                                                                                        i - 1],
            #                                                                                    der_systemic_press_artery[
            #                                                                                        i - 1])
            #
            #     P_systemic_vein[i] = heart_lung_obj.Finite_diff_meth_systemic_press_vein(i, heart_lung_obj.t[i - 1],
            #                                                                                heart_lung_obj.t[i],
            #                                                                                P_systemic_vein[i - 3],
            #                                                                                P_systemic_vein[i - 2],
            #                                                                                P_systemic_vein[i - 1],
            #                                                                                der_systemic_press_vein[
            #                                                                                    i - 1])
            #
            #     # Q_systemic_artery[i] = heart_lung_obj.Finite_diff_meth_systemic_flow_artery(i, heart_lung_obj.t[i - 1], heart_lung_obj.t[i],
            #     #                                                                                Q_systemic_artery[
            #     #                                                                                    i - 3],
            #     #                                                                                Q_systemic_artery[
            #     #                                                                                    i - 2],
            #     #                                                                                Q_systemic_artery[i - 1],
            #     #                                                                                der_systemic_flow_artery[i - 1])
            # heart_lung_obj.P_systemic_artery = P_systemic_artery[i]
            # heart_lung_obj.P_systemic_vein = P_systemic_vein[i]
            # # heart_lung_obj.Q_systemic_artery = Q_systemic_artery[i]
            #
            # # Calculate blood flow in pulmonary arteries and veins regarding their time
            # [Q_systemic_artery[i], Q_systemic_vein[i]] = heart_lung_obj.systemic_flow_vein_artery()
            # heart_lung_obj.Q_systemic_artery = Q_systemic_artery[i]
            # heart_lung_obj.Q_systemic_vein = Q_systemic_vein[i]




            # #Calculate blood pressure in pulmonary arteries and veins regarding their time
            # if i == 1:
            #     P_pulmonary_artery[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_press_artery(i,heart_lung_obj.t[i-1],heart_lung_obj.t[i], P_pulmonary_artery[i-1], der_pulmonary_press_artery[i-1])
            #     P_pulmonary_vein[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_press_vein(i,heart_lung_obj.t[i-1],heart_lung_obj.t[i], P_pulmonary_vein[i-1], der_pulmonary_press_vein[i-1])
            #     # Q_pulmonary_artery[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_flow_artery(i,heart_lung_obj.t[i-1],heart_lung_obj.t[i], Q_pulmonary_artery[i-1], der_pulmonary_flow_artery[i-1])
            #     #Q_pulmonary_vein[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_flow_vein(i,heart_lung_obj.t[i-1],heart_lung_obj.t[i], Q_pulmonary_vein[i-1], der_pulmonary_flow_vein[i-1])
            # elif i == 2:
            #    P_pulmonary_artery[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_press_artery(i, heart_lung_obj.t[i - 1], heart_lung_obj.t[i], P_pulmonary_artery[i - 2], P_pulmonary_artery[i - 1],
            #                                                                                       der_pulmonary_press_artery[i-1])
            #    P_pulmonary_vein[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_press_vein(i, heart_lung_obj.t[i - 1], heart_lung_obj.t[i], P_pulmonary_vein[i - 2], P_pulmonary_vein[i - 1],
            #                                                                                       der_pulmonary_press_vein[i-1])
            #    # Q_pulmonary_artery[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_flow_artery(i, heart_lung_obj.t[i - 1], heart_lung_obj.t[i], Q_pulmonary_artery[i - 2], Q_pulmonary_artery[i - 1],
            #    #                                                                                   der_pulmonary_flow_artery[i - 1])
            #    # Q_pulmonary_vein[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_flow_vein(i, heart_lung_obj.t[i - 1], heart_lung_obj.t[i], Q_pulmonary_vein[i - 2], Q_pulmonary_vein[i - 1],
            #    #                                                                                   der_pulmonary_flow_vein[i - 1])
            #
            # else:
            #     P_pulmonary_artery[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_press_artery(i, heart_lung_obj.t[i - 1], heart_lung_obj.t[i],
            #                                                                                    P_pulmonary_artery[
            #                                                                                        i - 3],
            #                                                                                    P_pulmonary_artery[
            #                                                                                        i - 2],
            #                                                                                    P_pulmonary_artery[i - 1],
            #                                                                                    der_pulmonary_press_artery[i - 1])
            #
            #     P_pulmonary_vein[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_press_vein(i, heart_lung_obj.t[i - 1], heart_lung_obj.t[i],
            #                                                                                    P_pulmonary_vein[
            #                                                                                        i - 3],
            #                                                                                    P_pulmonary_vein[
            #                                                                                        i - 2],
            #                                                                                    P_pulmonary_vein[i - 1],
            #                                                                                    der_pulmonary_press_vein[i - 1])
            #
            #     # Q_pulmonary_artery[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_flow_artery(i, heart_lung_obj.t[i - 1], heart_lung_obj.t[i],
            #     #                                                                                Q_pulmonary_artery[
            #     #                                                                                    i - 3],
            #     #                                                                                Q_pulmonary_artery[
            #     #                                                                                    i - 2],
            #     #                                                                                Q_pulmonary_artery[i - 1],
            #     #                                                                                der_pulmonary_flow_artery[i - 1])
            #     #
            #     # Q_pulmonary_vein[i] = heart_lung_obj.Finite_diff_meth_Pulmonary_flow_vein(i, heart_lung_obj.t[i - 1], heart_lung_obj.t[i],
            #     #                                                                                Q_pulmonary_vein[
            #     #                                                                                    i - 3],
            #     #                                                                                Q_pulmonary_vein[
            #     #                                                                                    i - 2],
            #     #                                                                                Q_pulmonary_vein[i - 1],
            #     #                                                                                der_pulmonary_flow_vein[i - 1])
            #
            # heart_lung_obj.P_pulmonary_artery = P_pulmonary_artery[i]
            # heart_lung_obj.P_pulmonary_vein = P_pulmonary_vein[i]
            #
            # # heart_lung_obj.Q_pulmonary_artery = Q_pulmonary_artery[i]
            # # heart_lung_obj.Q_pulmonary_vein = Q_pulmonary_vein[i]
            #
            #
            # #Calculate blood flow in pulmonary arteries and veins regarding their time
            # [Q_pulmonary_artery[i], Q_pulmonary_vein[i]] = heart_lung_obj.Pulmonary_flow_vein_artery()
            # heart_lung_obj.Q_pulmonary_artery = Q_pulmonary_artery[i]
            # heart_lung_obj.Q_pulmonary_vein = Q_pulmonary_vein[i]

            # Calculate blood flow in atria and ventricles regarding their time
            [Q_ra[i], Q_rv[i], Q_la[i], Q_lv[i]] = heart_lung_obj.Blood_Flow_Atria_ventricles()
            Q_rv[i] = flow
            Q_la[i] = flow
            heart_lung_obj.Q_rv = Q_rv[i]  # replace new values (perfusion) in flow function
            heart_lung_obj.Q_la = Q_la[i]  # replace new values (perfusion) in flow function
 


            # Calculate blood volume changes in atria and ventricles regarding their time
            [der_volume_RA[i], der_volume_RV[i], der_volume_LA[i],
             der_volume_LV[i]] = heart_lung_obj.Blood_Volume_changes_Atria_ventricles()

            # Calculate blood pressure and volume in systemic circulation regarding its time
            # [der_systemic_press_artery[i], der_systemic_press_vein[i]] = heart_lung_obj.systemic_circulation()

            # Calculate blood pressure and volume in pulmonary circulation regarding its time
            # [der_pulmonary_press_artery[i], der_pulmonary_press_vein[i]] = heart_lung_obj.pulmonary_circulation()
            # P_ra.update({i: 0.00750061682704466 * P_ra[i] for i in P_ra.keys()}) #mmHg
            # P_rv.update({i: 0.00750061682704466 * P_rv[i] for i in P_rv.keys()})
            # P_la.update({i: 0.00750061682704466 * P_la[i] for i in P_la.keys()})
            # P_lv.update({i: 0.00750061682704466 * P_lv[i] for i in P_lv.keys()})
            # 
            # Q_ra.update({i: 1000000 * Q_ra[i] for i in Q_ra.keys()}) #mL/s
            # # Q_rv.update({i: 1000000 * Q_rv[i] for i in Q_rv.keys()})
            # # Q_la.update({i: 1000000 * Q_la[i] for i in Q_la.keys()})
            # Q_lv.update({i: 1000000 * Q_lv[i] for i in Q_lv.keys()})
            # 
            # V_ra.update({i: 1000000 * V_ra[i] for i in V_ra.keys()}) #mL
            # V_rv.update({i: 1000000 * V_rv[i] for i in V_rv.keys()})
            # V_la.update({i: 1000000 * V_la[i] for i in V_la.keys()})
            # V_lv.update({i: 1000000 * V_lv[i] for i in V_lv.keys()})

            n0 = [e_a[i], e_v[i], V_la[i], V_lv[i], V_rv[i], V_ra[i], P_la[i], P_lv[i], P_rv[i], P_ra[i], Q_la[i],
                  Q_lv[i], Q_rv[i], Q_ra[i], heart_lung_obj.P_pulmonary_artery , heart_lung_obj.P_pulmonary_vein, Q_pulmonary_artery, Q_pulmonary_vein]
            value.append(n0)
        #
        #
        csv_writer.writerows(value)
    
       # Find the max and min from half of the time untill end (when the trend is preodic)
       #  n = len(P_ra) // 2



    # # draw plots
    # e_a = dict(list(e_a.items())[len(e_a) // 2:])
    # activation_atria_dic_array = []
    # activation_atria_dic_array.append({'function':e_a,'x_lable':'Time','y_lable':'value','function_identity':'e_a'})
    # plot_results(activation_atria_dic_array)
    #
    # e_v = dict(list(e_v.items())[len(e_v) // 2:])
    # activation_ventricles_dic_array = []
    # activation_ventricles_dic_array.append({'function':e_v,'x_lable':'Time','y_lable':'value','function_identity':'e_v'})
    # plot_results(activation_ventricles_dic_array)
    #
    # # convert the value to physiology
    # P_ra.update({i: 0.00750061682704466 * P_ra[i] for i in P_ra.keys()})
    # P_rv.update({i: 0.00750061682704466 * P_rv[i] for i in P_rv.keys()})
    # P_la.update({i: 0.00750061682704466 * P_la[i] for i in P_la.keys()})
    # P_lv.update({i: 0.00750061682704466 * P_lv[i] for i in P_lv.keys()})
    # # read the values from the half to end
    # P_ra = dict(list(P_ra.items())[len(P_ra) // 2:])
    # P_rv = dict(list(P_rv.items())[len(P_rv) // 2:])
    # P_la = dict(list(P_la.items())[len(P_la) // 2:])
    # P_lv = dict(list(P_lv.items())[len(P_lv) // 2:])
    #
    # max_min_values(dict(list(P_ra.items())), 'Right Atrium Pressure')
    # max_min_values(dict(list(P_rv.items())), 'Right Ventricle Pressure')
    # max_min_values(dict(list(P_la.items())), 'Left Atrium Pressure')
    # max_min_values(dict(list(P_lv.items())), 'Left Ventricle Pressure')
    #
    # # draw pressure
    # left_pressure_dic_array=[]
    # left_pressure_dic_array.append({'function': P_la, 'x_lable': 'Time', 'y_lable': 'Pressure(mmHg)', 'function_identity': 'P_la'})
    # left_pressure_dic_array.append({'function': P_lv, 'x_lable': 'Time', 'y_lable': 'Pressure(mmHg)', 'function_identity': 'P_lv'})
    # plot_results(left_pressure_dic_array)
    #
    # right_pressure_dic_array=[]
    # right_pressure_dic_array.append({'function':P_ra,'x_lable':'Time','y_lable':'Pressure(mmHg)','function_identity':'P_ra'})
    # right_pressure_dic_array.append({'function': P_rv, 'x_lable': 'Time', 'y_lable': 'Pressure(mmHg)', 'function_identity': 'P_rv'})
    # plot_results(right_pressure_dic_array)
    #
    #
    #
    #
    # # convert the value to physiology
    # Q_ra.update({i: 1000000 * Q_ra[i] for i in Q_ra.keys()})
    # Q_rv.update({i: 1000000 * Q_rv[i] for i in Q_rv.keys()})
    # Q_la.update({i: 1000000 * Q_la[i] for i in Q_la.keys()})
    # Q_lv.update({i: 1000000 * Q_lv[i] for i in Q_lv.keys()})
    # # read the values from the half to end
    # Q_ra = dict(list(Q_ra.items())[len(Q_ra) // 2:])
    # Q_rv = dict(list(Q_rv.items())[len(Q_rv) // 2:])
    # Q_la = dict(list(Q_la.items())[len(Q_la) // 2:])
    # Q_lv = dict(list(Q_lv.items())[len(Q_lv) // 2:])
    #
    # max_min_values(dict(list(Q_la.items())), 'Left Atrium Flow')
    # max_min_values(dict(list(Q_lv.items())), 'Left Ventricle Flow')
    # max_min_values(dict(list(Q_ra.items())), 'Right Atrium Flow')
    # max_min_values(dict(list(Q_rv.items())), 'Right Ventricle Flow')
    # # draw flow
    # left_flow_dic_array=[]
    # left_flow_dic_array.append({'function': Q_la, 'x_lable': 'Time', 'y_lable': 'Flow(mL/s)', 'function_identity': 'Q_la'})
    # left_flow_dic_array.append({'function': Q_lv, 'x_lable': 'Time', 'y_lable': 'Flow(mL/s)', 'function_identity': 'Q_lv'})
    # plot_results(left_flow_dic_array)
    #
    # right_flow_dic_array=[]
    # right_flow_dic_array.append({'function':Q_ra,'x_lable':'Time','y_lable':'Flow(mL/s)','function_identity':'Q_ra'})
    # right_flow_dic_array.append({'function': Q_rv, 'x_lable': 'Time', 'y_lable': 'Flow(mL/s)', 'function_identity': 'Q_rv'})
    # plot_results(right_flow_dic_array)
    # Q_rv1 = dict(list(Q_rv.items())[:100])
    # y=list(Q_rv1.values())
    # x = np.linspace(5, 6, 100)
    # plt.plot(x, y)
    # plt.show()
    #
    # # convert the value to physiology
    # V_ra.update({i: 1000000 * V_ra[i] for i in V_ra.keys()})
    # V_rv.update({i: 1000000 * V_rv[i] for i in V_rv.keys()})
    # V_la.update({i: 1000000 * V_la[i] for i in V_la.keys()})
    # V_lv.update({i: 1000000 * V_lv[i] for i in V_lv.keys()})
    # # read the values from the half to end
    # V_ra = dict(list(V_ra.items())[len(V_ra) // 2:])
    # V_rv = dict(list(V_rv.items())[len(V_rv) // 2:])
    # V_la = dict(list(V_la.items())[len(V_la) // 2:])
    # V_lv = dict(list(V_lv.items())[len(V_lv) // 2:])
    #
    # max_min_values(dict(list(V_ra.items())), 'Right Atrium Volume')
    # max_min_values(dict(list(V_rv.items())), 'Right Ventricle Volume')
    # max_min_values(dict(list(V_la.items())), 'Left Atrium Volume')
    # max_min_values(dict(list(V_lv.items())), 'Left Ventricle Volume')
    # # draw volume
    # left_volume_dic_array=[]
    # left_volume_dic_array.append({'function': V_la, 'x_lable': 'Time', 'y_lable': 'Volume(mL)', 'function_identity': 'V_la'})
    # left_volume_dic_array.append({'function': V_lv, 'x_lable': 'Time', 'y_lable': 'Volume(mL)', 'function_identity': 'V_lv'})
    # plot_results(left_volume_dic_array)
    #
    # right_volume_dic_array=[]
    # right_volume_dic_array.append({'function':V_ra,'x_lable':'Time','y_lable':'Volume(mL)','function_identity':'V_ra'})
    # right_volume_dic_array.append({'function': V_rv, 'x_lable': 'Time', 'y_lable': 'Volume(mL)', 'function_identity': 'V_rv'})
    # plot_results(right_volume_dic_array)


    # # convert the value to physiology
    # P_pulmonary_artery.update({i: 0.00750061682704466 * P_pulmonary_artery[i] for i in P_pulmonary_artery.keys()})
    # P_pulmonary_vein.update({i: 0.00750061682704466 * P_pulmonary_vein[i] for i in P_pulmonary_vein.keys()})
    # # read the values from the half to end
    # P_pulmonary_artery = dict(list(P_pulmonary_artery.items())[len(P_pulmonary_artery) // 2:])
    # P_pulmonary_vein = dict(list(P_pulmonary_vein.items())[len(P_pulmonary_vein) // 2:])
    # # draw pressure
    # pul_pressure_dic_array = []
    # pul_pressure_dic_array.append({'function': P_pulmonary_artery, 'x_lable': 'Time', 'y_lable': 'Pressure(mmHg)', 'function_identity': 'P_pulmonary_artery'})
    # pul_pressure_dic_array.append({'function': P_pulmonary_vein, 'x_lable': 'Time', 'y_lable': 'Pressure(mmHg)', 'function_identity': 'P_pulmonary_vein'})
    # plot_results(pul_pressure_dic_array)
    #
    #
    # # convert the value to physiology
    # Q_pulmonary_artery.update({i: 1000000 * Q_pulmonary_artery[i] for i in Q_pulmonary_artery.keys()})
    # Q_pulmonary_vein.update({i: 1000000 * Q_pulmonary_vein[i] for i in Q_pulmonary_vein.keys()})
    # # read the values from the half to end
    # Q_pulmonary_artery = dict(list(Q_pulmonary_artery.items())[len(Q_pulmonary_artery) // 2:])
    # Q_pulmonary_vein = dict(list(Q_pulmonary_vein.items())[len(Q_pulmonary_vein) // 2:])
    #
    # max_min_values(dict(list(P_pulmonary_artery.items())), 'Pulmonary Artery Pressure')
    # max_min_values(dict(list(P_pulmonary_vein.items())), 'Pulmonary Vein Pressure')
    # max_min_values(dict(list(Q_pulmonary_artery.items())), 'Pulmonary Artery Flow')
    # max_min_values(dict(list(Q_pulmonary_vein.items())), 'Pulmonary Vein Flow')
    # # draw flow
    # pul_flow_dic_array = []
    # pul_flow_dic_array.append({'function': Q_pulmonary_artery, 'x_lable': 'Time', 'y_lable': 'Flow(mL/s)','function_identity': 'Q_pulmonary_artery'})
    # pul_flow_dic_array.append({'function': Q_pulmonary_vein, 'x_lable': 'Time', 'y_lable': 'Flow(mL/s)','function_identity': 'Q_pulmonary_vein'})
    # plot_results(pul_flow_dic_array)





    # Store  the results  in an excel file!
    Store_array=[]
    Store_array.append(list(P_ra.values()))
    Store_array.append(list(P_rv.values()))
    Store_array.append(list(P_la.values()))
    Store_array.append(list(P_lv.values()))
    Store_array.append(list(Q_ra.values()))
    Store_array.append(list(Q_rv.values()))
    Store_array.append(list(Q_la.values()))
    Store_array.append(list(Q_lv.values()))
    Store_array.append(list(V_ra.values()))
    Store_array.append(list(V_rv.values()))
    Store_array.append(list(V_la.values()))
    Store_array.append(list(V_lv.values()))
    # Store_array.append(list(P_pulmonary_artery.values()))
    # Store_array.append(list(P_pulmonary_vein.values()))
    # Store_array.append(list(Q_pulmonary_artery.values()))
    # Store_array.append(list(Q_pulmonary_vein.values()))

    #new_list = [["first", "second"], ["third", "fourth"]]
    with open("Normal_new.csv", 'w') as f:
        fc = csv.writer(f, lineterminator='\n')
        fc.writerows(Store_array)

    #Store_array=dict(Store_array)
    #Store_array.to_excel("Normal.xlsx")



    # plot_results(pressure_dic_array)
    #
    # plot_results(P_rv, 'Time', 'Value', 'Right Ventricle Pressure')
    # plot_results(P_la, 'Time', 'Value', 'Left Atrium Pressure')
    # plot_results(P_lv, 'Time', 'Value', 'Left Ventricle Pressure')
    #
    # plot_results(Q_la, 'Time', 'Value (m3_per_s)', 'Left Atrium Flow')
    # plot_results(Q_lv, 'Time', 'Value (m3_per_s)', 'Left Ventricle Flow')
    # plot_results(Q_ra, 'Time', 'Value (m3_per_s)', 'Right atrium Flow')
    # plot_results(Q_rv, 'Time', 'Value(m3_per_s)', 'Right Ventricle Flow')
    #
    # plot_results(der_volume_RA, 'Time', 'Value', 'blood volume changes in RA')
    # plot_results(der_volume_RV, 'Time', 'Value', 'blood volume changes in RV')
    # plot_results(der_volume_LA, 'Time', 'Value', 'blood volume changes in LA')
    # plot_results(der_volume_LV, 'Time', 'Value', 'blood volume changes in LV')
    #
    # plot_results(V_ra, 'Time', 'Value (m3)', 'blood volume in RA')
    # plot_results(V_rv, 'Time', 'Value (m3)', 'blood volume in RV')
    # plot_results(V_la, 'Time', 'Value (m3)', 'blood volume in LA')
    # plot_results(V_lv, 'Time', 'Value (m3)', 'blood volume in LV')

    # plot_results(der_systemic_press_artery, 'Time', 'Value', 'Systemic Artery Pressure changes')
    # plot_results(der_systemic_press_vein, 'Time', 'Value', 'Systemic Vein Pressure changes')
    # plot_results(der_pulmonary_flow_artery, 'Time', 'Value', 'Pulmonary Artery Flow changes')
    # plot_results(der_pulmonary_flow_vein, 'Time', 'Value', 'Pulmonary Vein Flow changes')

    # plot_results(P_systemic_artery, 'Time', 'Value', 'Systemic Artery Pressure')
    # plot_results(P_systemic_vein, 'Time', 'Value', 'Systemic Vein Pressure')
    # plot_results(Q_systemic_artery, 'Time', 'Value', 'Systemic Artery Flow')
    # plot_results(Q_systemic_vein, 'Time', 'Value', 'Systemic Vein Flow')

    # plot_results(der_pulmonary_press_artery, 'Time', 'Value', 'Pulmonary Artery Pressure changes')
    # plot_results(der_pulmonary_press_vein, 'Time', 'Value', 'Pulmonary Vein Pressure changes')
    # # plot_results(der_pulmonary_flow_artery, 'Time', 'Value', 'Pulmonary Artery Flow changes')
    # # plot_results(der_pulmonary_flow_vein, 'Time', 'Value', 'Pulmonary Vein Flow changes')
    #
    # plot_results(P_pulmonary_artery, 'Time', 'Value(J_per_m3)', 'Pulmonary Artery Pressure')
    # plot_results(P_pulmonary_vein, 'Time', 'Value(J_per_m3)', 'Pulmonary Vein Pressure')
    # plot_results(Q_pulmonary_artery, 'Time', 'Value(m3_per_s', 'Pulmonary Artery Flow')
    # plot_results(Q_pulmonary_vein, 'Time', 'Value(m3_per_s)', 'Pulmonary Vein Flow')





    # print_results(e_a, e_v, P_ra, P_rv, P_la, P_lv, Q_la, Q_lv, Q_ra, Q_rv, der_volume_RA, der_volume_RV, der_volume_LA,
    #               der_volume_LV, V_ra, V_rv, V_la, V_lv, der_systemic_press_artery, der_systemic_press_vein,
    #                P_systemic_artery, P_systemic_vein, Q_systemic_artery, Q_systemic_vein, der_pulmonary_press_artery, der_pulmonary_press_vein,
    #                P_pulmonary_artery, P_pulmonary_vein, Q_pulmonary_artery, Q_pulmonary_vein)
