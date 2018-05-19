#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created by Alma Andersson TMBIM1 2018

Code used to solve the questions of Exercise 1 Homework 3 in SI2540 VT2018
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
    
    variable_box = fun.Driven_Variables(0.5,1.0)
    var = variable_box.triple_bif
    dt = np.pi/var['W']*0.01
#%% Toogle
    
    #Choose which question to generate results from by setting the variable to true
    
    question_a = False
    question_b1 = False
    question_b2 = False
    question_c = False
#%%  Question A
    if question_a:
        try:
            del syst, integrator, time, traj
        except NameError:
            pass
        #----------Setup---------#
        syst = fun.Driven_Pendulum(var['g'],var['w'],var['A'],2/3.)
        integrator = fun.Integrator(syst.function,h=dt)
        
        #----------Integrate---------#
        T = np.array([0.,3000.])
        y0 = np.array([1.0,0.0,0.0])
        integrator.generate_trajectory(T,y0,periodic=True)
        time, traj = integrator.time, integrator.traj
        
        #----------Plot---------#
        plt_figure,phi_ax = plt.subplots(1,3,figsize=(15,5),facecolor='white')
        fun.Plotter.add_trajectory(phi_ax[0],time,traj[:,0])
        fun.Plotter.add_labels(phi_ax[0],r'$\phi-plot$','time',r'$\phi$')
        fun.Plotter.add_trajectory(phi_ax[1],time,traj[:,1])
        fun.Plotter.add_labels(phi_ax[1],r'$\psi-plot$','time',r'$\psi$')
        fun.Plotter.add_trajectory(phi_ax[2],traj[:,0],traj[:,1],ltype='',mtype='.')
        fun.Plotter.add_labels(phi_ax[2],r'$\phi-\psi$-plot',r'$\phi$',r'$\psi$')
        phi_ax[-1].set_xlim([-np.pi,np.pi])
        var_fold = (var['g'],var['w'],var['A'],round(var['W'],3))
        sup = '$\gamma = $ %s , $\omega = $ %s , $A = $ %s , $\Omega = $ %s ' % tuple(var_fold)  
        plt_figure.suptitle(sup)
        plt_figure.tight_layout()
        plt_figure.show()
        
        
#%% Question b1
    if question_b1:
        try:
            del syst, integrator, time, traj
        except NameError:
            pass
        #----------Setup---------#
        syst = fun.Driven_Pendulum(var['g'],var['w'],var['A'],var['W'])
        integrator = fun.Integrator(syst.function,h=dt)
        #----------Integrate---------#
        T = np.array([0.,10**3.])
        y0 = np.array([1.0,0.0,0.0])
        integrator.generate_trajectory(T,y0,periodic=True)
        time, traj = integrator.time, integrator.traj
        #--------Stroboscobic Map----------#
        burn_in = int(0.5*len(traj[:,0]))
        strobe1, strobe2 = fun.Stroboscopic.stroboscopic_map(dt,time[burn_in::],traj[burn_in::,:],var['W'])
        #----------Count-Occurance----------#
        uni1,uni2,cnt = fun.Stroboscopic.count_stroboscopic(strobe1,strobe2)
        #----------------Plot----------------#
        strobe_fig, strobe_ax = plt.subplots(1,1,facecolor='white')
        fun.Plotter.add_trajectory(strobe_ax,traj[burn_in::,0],traj[burn_in::,1],ltype='',mtype='o',alph=0.05)
        prms = (var['g'],var['w'],var['A'],round(var['W'],3))
        fun.Stroboscopic.plot_stroboscobic_map_alt2(strobe_ax,strobe1,strobe2,prms)
        fun.Stroboscopic.plot_add_count(strobe_ax,uni1,uni2,cnt)
        strobe_fig.show()
#%% Question b2
    if question_b2:
        A_list = np.arange(1.0,1.5,0.01)
        strobe_list1,strobe_list2 = [], []
        traj_list = []
        for (k,amp) in enumerate(A_list):
            var_amp = np.array([var['g'],var['w'],amp,var['W']])
            syst = fun.Driven_Pendulum(*var_amp)
            integrator = fun.Integrator(syst.function,h=dt)
            #----------Integrate---------#
            T = np.array([0.,3000.])
            y0 = np.array([-1.0,0.0,0.0])
            integrator.generate_trajectory(T,y0,periodic=True)
            time, traj = integrator.time, integrator.traj
            traj_list.append(traj)
            #--------Stroboscopic----------#
            burn_in = int(2./3.*len(traj[:,0]))
            strobe1, strobe2 = fun.Stroboscopic.stroboscopic_map(dt,time[burn_in::],traj[burn_in::,:],var['W'])
            strobe_list1.append(strobe1)
            strobe_list2.append(strobe2)
            
        #----------------Plot----------------#
        strobe_list2 = np.array(strobe_list2)
        strobe_list1 = np.array(strobe_list1)
        wrapped_s1, wrapped_s2 = fun.Stroboscopic.wrap(strobe_list1), fun.Stroboscopic.wrap(strobe_list2)
        bif_fig, bif_ax = plt.subplots(1,2)
        fun.Plotter.bifurcation_plot(bif_ax,A_list,wrapped_s1,wrapped_s2)
        bif_fig.tight_layout()
#%% Question C
    if question_c:
        try:
            del syst, integrator, time, traj
        except NameError:
            pass
        #----------Setup---------#
        syst = fun.Driven_Pendulum(var['g'],var['w'],var['A'],var['W'])
        integrator = fun.Integrator(syst.function,h=dt)
        T = np.array([0.,3000.])
        y0 = np.array([1.0,0.0,0.0])
        dlist = (np.random.random((5,3))-np.random.random((5,3)))
        traj_list = []
        #---------Approach-Attractor---------#
        integrator.generate_trajectory(T,y0,periodic=True)
        time0, traj0 = integrator.time, integrator.traj
        y1 = traj0[-1,:]
        #----------Integrate-Bundle----------#
        for ii in xrange(dlist.shape[0]):
            d = dlist[ii,:]/np.linalg.norm(dlist[ii,:])*1e-7
            integrator.generate_trajectory(np.array([0,1000]) + time0[-1],y1 + d,periodic=True)
            time, traj = integrator.time, integrator.traj
            traj_list.append(traj)
        #----------------Plot----------------#
        bundle_fig, bundle_ax = plt.subplots(1,1,facecolor='white')
        fun.Plotter.plot_bundle(bundle_ax,time,traj_list)
        bundle_fig.show()
    
