#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created by Alma Andersson TMBIM1 2018

Code used to solve the questions of Exercise 2 Homework 3 in SI2540 VT2018
All functions can be found in the file homework3_functions.py

"""
import sys
sys.path.append('../lib')
import homework3_functions as fun
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

if __name__ == '__main__':
#%% General Variables
    a,b = 0.1, 0.1
    c_interval = np.arange(4,20,0.1)
    var = ['x','y','z']
    dt = 0.01
#%% Toggle
    
    #Choose which question to generate results from by setting the variable to true
    #If rsyst is set to True this gives a projection of a system in the xy-phase space
    
    rsyst = False
    question_a = False
    question_b = False
    question_c = False
#%% Test System
    if rsyst:
        #----------Setup-Integrator----------#
            syst = fun.Rossler_System(a,b,19.5)
            integrator = fun.Integrator(syst.function,h=0.01)
            #-------------Integrate-------------#
            T = np.array([0.,3000.])
            y0 = np.array([3.0,0.0,0.0])
            integrator.generate_trajectory(T,y0)
            time, traj = integrator.time, integrator.traj
            #----------------Plot----------------#
            plt.figure()
            plt.plot(traj[-15000:,0],traj[-15000:,1],linewidth=3,color='k',alpha=0.8)
            plt.show()
            
#%% Question A    
    if question_a:
        lorenz_map_list = [[],[],[]]
        for c in c_interval:
            #----------Setup-Integrator----------#
            syst = fun.Rossler_System(a,b,c)
            integrator = fun.Integrator(syst.function,h=dt)
            #-------------Integrate-------------#
            T = np.array([0.,1000.])
            y0 = np.array([1.0,0.0,0.0])
            integrator.generate_trajectory(T,y0)
            time, traj = integrator.time, integrator.traj
            
            burn_in = int((T[-1]/dt)*0.10)
            for ii in xrange(3):
                lorenz_map_list[ii].append(fun.Lorenz.lorenz_map(traj[-burn_in:,ii]))
        #---------------Adjust---------------#
        lorenz_map_arr = []
        for jj in xrange(3):
            adj = np.min([len(ii) for ii in lorenz_map_list[jj]])
            lorenz_map_arr.append(np.array([ii[0:adj] for ii in lorenz_map_list[jj]]))
        #----------------Plot----------------#
        lorenz_fig, lorenz_ax = plt.subplots(1,3,facecolor='white')
        for jj in xrange(3):
            fun.Lorenz.lorenz_bifurcation_plot(lorenz_ax[jj],c_interval,lorenz_map_arr[jj],variable=var[jj])
        lorenz_fig.tight_layout()
        lorenz_fig.show()
        
#%% Question B
    if question_b:
            #---------Rossler-Variables---------#
            c_interval = np.arange(4.0,20.0,0.1)
            lyap_list = []
            for c in c_interval:
                #----------Setup-Integrator----------#
                syst = fun.Rossler_System(a,b,c)
                integrator = fun.Integrator_Lyapunov(syst.function,h=0.01)
                #-------------Integrate-------------#
                T = np.array([0.,1000.])
                y0 = np.array([3.0,0.0,0.0])
                integrator.generate_trajectory(T,y0,burn_in=int((0.75*T[-1])/dt))
                #----------Store-Variables----------#
                lyap = integrator.lyapunov
                lyap_list.append(lyap)
                
            lyap_arr = np.array(lyap_list)
            #----------------Plot----------------#
            lyap_fig, lyap_ax = plt.subplots(1,1,facecolor='white')
            fun.Lorenz.lyapunov_plot(lyap_ax,c_interval,lyap_arr)
            
#%% question C
    if question_c:
        rossler = True
        logistic = True
        
        #---------Rossler-attractor---------#
        if rossler:
            #----------Setup-Integrator----------#
            qc_c = 20.0
            syst = fun.Rossler_System(a,b,qc_c)
            integrator = fun.Integrator(syst.function,h=dt)
            #-------------Integrate-------------#
            T = np.array([0.,3000.])
            y0 = np.array([1.0,0.0,0.0])
            integrator.generate_trajectory(T,y0)
            time, traj = integrator.time, integrator.traj
            #-------------Lorenz-Map-------------#
            burn_in = int(np.floor(len(time)*0.75))
            lorenz_map = fun.Lorenz.lorenz_map(traj[burn_in::,0],visualize=False)
            l1,l2 = fun.Lorenz.lorenz_prepare(lorenz_map)
            #------------Rossler-Plot------------#
            if not logistic:
                lorenz_fig, lorenz_ax = plt.subplots(1,1,facecolor='white')
                fun.Lorenz.lorenz_map_plot(lorenz_ax,l1,l2)
                lorenz_fig.tight_layout()
                lorenz_fig.show()
                
        #------------Logistic-Map------------#
        if logistic:
            #---------Logistic-Variables---------#
            r,t0 = 3.9,0.3
            burn_in = 20
            niter = 100
            orbit = fun.Logistic_Map(r,t0,niter)
            o1,o2 = fun.Lorenz.lorenz_prepare(orbit[burn_in:-5])
            #-----------Logistic-Plot-----------#
            if not rossler:
                logistic_fig, logistic_ax = plt.subplots(1,1,facecolor='white')
                fun.Lorenz.lorenz_map_plot(logistic_ax,o1,o2)
                logistic_fig.tight_layout()
                
        #------------Double-Plot------------#
        if logistic and rossler:
             ros_log_fig, ros_log_ax = plt.subplots(1,2,facecolor='white')
             fun.Lorenz.lorenz_map_plot(ros_log_ax[0],l1,l2)
             fun.Lorenz.lorenz_map_plot(ros_log_ax[1],o1,o2)
             ros_log_ax[0].set_title(r'Rossler Attractor, $c=$ %s' %qc_c )
             ros_log_ax[1].set_title(r'Logistic Map, $r=$ %s' % r)
             ros_log_fig.tight_layout()