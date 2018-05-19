#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created by Alma Andersson TMBIM1 2018

Functions and necessary classes used to solve Exercise 1 and 2 for Homework 3 in the cours
SI2540 VT2018.

"""

import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt
import clipboard
from scipy import signal
import peakutils as pks
#%% Common functions
class Integrator:
    """Class to use for numerical integration of trajectories, implementing the
    scipy ode solver dopri15. The class is to be initated with a function (required) 
    and a size for the timestep (optional, default = 0.01). If the argument 'periodic'
    is provided the system will adjust the values of of the first coordinate as to
    be bounded within the intercal [-pi,pi].
    """
    def __init__(self,f,**kwargs):
        self.f = f
        self.h = kwargs.pop('h', 0.01)
        self.r = ode(f).set_integrator('dopri5')
    def generate_trajectory(self,tt,y0,periodic=False):
        """
        Generate a trajectory from between provided initial and final time
        starting from the given initial values 
        """
        self.time,self.traj = [],[]
        self.r.set_initial_value(y0,tt[0])
        while self.r.successful() and self.r.t < tt[1]:
          self.r.integrate(self.r.t + self.h)
          self.traj.append(self.r.y)
          #--------Keep phi in interval -pi <= phi < pi
          if periodic:
              if self.traj[-1][0] < -np.pi:
                  while self.traj[-1][0] < -np.pi:
                      self.traj[-1][0] = self.traj[-1][0] + 2*np.pi
              elif self.traj[-1][0] > np.pi:
                   while self.traj[-1][0] > np.pi:
                       self.traj[-1][0] = self.traj[-1][0] - 2*np.pi
          self.time.append(self.r.t+self.h)
        self.traj = np.array(self.traj)
        self.time = np.array(self.time)

def make_separator(string):
    """
    Function to use for markup of scripts
    """
    string = string.replace(' ','-')
    length = len(string)
    padd = np.floor((36.-length)/2.)
    clipboard.copy('#'+'-'*int(padd) + string + '-'*int(padd)+'#')
#%% Rösler System
class Rossler_System:
    """
    Creates an environment for the Rössler system and constructs a function compatible
    with the Integrator class. All parameters can be adjusted (a,b,c) but have
    default values of 0.1
    """
    def __init__(self,a=0.1,b=0.1,c=0.1):
        self.a = a
        self.b = b
        self.c = c
    def function(self,t,y):
        return np.array([-y[1]-y[2], y[0]+self.a*y[1],self.b+y[2]*(y[0]-self.c)])
    
class Integrator_Lyapunov:
    """
    Used to compute the Largest Lyapunov exponent of a system. Similar to the Integrator class
    but trajectories and timesteps are not stored.
    
    The average lyapunov exponent for the whole
    trajectory is stored in attribute 'lyapunov'.
    """
    def __init__(self,f,**kwargs):
        self.f = f
        self.h = kwargs.pop('h', 0.01)
        self.r = ode(f).set_integrator('dopri5') #reference trajectory
        self.xi = ode(f).set_integrator('dopri5') #perturbed trajectory
        self.d0 = 1e-7 #norm of perturbation 
        
    def generate_trajectory(self,tt,y0,burn_in=0):
       """
       Generate trajectory between provided start and endpoint in time for
       given initial values. If argument burn_in is provided this number of
       timesteps will be executed before initializing the perturbed trajectory
       """
       self.r.set_initial_value(y0,tt[0])
       k = 0
       first = True
       self.ratio_list = []
       while self.r.successful() and self.r.t < tt[1]:
          k += 1
          self.r.integrate(self.r.t + self.h)
          if k > burn_in:
              if first:
                  #----Create-Parallell-Trajectory----#
                  mod_traj = np.array(self.r.y[:])
                  mod_traj = mod_traj + self.d0/np.sqrt(3)*np.ones(3)
                  self.xi.set_initial_value(mod_traj,self.r.t)
                  first = False
                  j = 0.
              else:
                  j += 1.
                  self.xi.integrate(self.xi.t + self.h)
                  #---------------Scale---------------#
                  p,q = np.array(self.r.y[:]), np.array(self.xi.y[:])
                  diff = q-p
                  d1 = np.linalg.norm(diff)
                  ratio = np.log(d1)-np.log(self.d0)
                  if j < 2:
                      lyap = ratio
                      jold = j
                  elif j >= 5.0/self.h and j % 5.0/self.h == 0:
                      dt = self.h*(j-jold)
                      lyap = (j-1)/j*lyap + 1./dt*ratio
                      jold = j
                      
                  self.ratio_list.append(ratio)
                  s = np.exp(-ratio)
                  self.xi.y[:] = p+s*diff
       self.lyapunov = lyap
       
class Lorenz(object):
    """
    Class containing different functions used to generate Lorenz Maps and 
    Plot the obtained results.
    """
    def __init__(self,):
        pass
    @staticmethod
    def lorenz_map(seq,visualize=False):
        """
        Return the values of the identified peaks of a provided trajectory
        """
        pos = pks.indexes(seq,thres=0.3,min_dist=0.5)
        L = seq[pos]
        if visualize:
            plt.figure()
            t = np.arange(len(seq))
            t_top = t[pos]
            plt.plot(t,seq)
            plt.plot(t_top,L,marker='o',linestyle='')
        return L
    @staticmethod
    def lorenz_prepare(Lmap):
        """
        Generate arrays to be used in visualization of the Lorenz map
        """
        return Lmap[0:-1],Lmap[1::]
    
    @staticmethod
    def lorenz_bifurcation_plot(ax,arr,lmap,variable='x'):
        """
        Layout for bifurcation plot obtained from a Lorenz map
        """
        ax.plot(arr,lmap,linestyle='',marker='o',markerfacecolor='k',markersize=1)
        ax.set_xlabel(r'$c$',fontsize=25)
        ax.set_ylabel(variable + r'$_{max}$',fontsize=25)
        
    @staticmethod
    def lorenz_map_plot(ax,l1,l2):
        """
        Plot Layout for Lorenz Map
        """
        ax.plot(l1,l2,linestyle='',marker='o',markerfacecolor='k',markersize=1)
        ax.set_xlabel(r'$x_{k}$',fontsize=25)
        ax.set_ylabel(r'$x_{k+1}$',fontsize=25)
        
    @staticmethod
    def logistic_map_plot(ax,l1,l2):
        """
        Plot Layout for Logisitic map
        """
        ax.plot(l1,l2,linestyle='',marker='o',markerfacecolor='k',markersize=1)
        ax.set_ylabel(r'$x_{k}$',fontsize=25)
        ax.set_xlabel(r'$r$',fontsize=25)
       
    @staticmethod
    def lyapunov_plot(ax,prm,lyap):
        """
        Plot Layout for Lyapunov exponent as function of parameter 
        """
        ax.plot(prm,lyap,marker='o',linestyle='-',markersize=2,color='k',linewidth=2)
        ax.axhline(y=0,color='k')
        ax.set_xlabel(r'$c$',fontsize=25)
        ax.set_ylabel(r'$\lambda_{1}$',fontsize=25)

#%% Pendulum Functions
class Driven_Pendulum:
    """
    Set up environment for driven damped pendulum, creates a function compatible with
    the integrator class
    """
    def __init__(self,g,w,A,W):
        self.A = A
        self.g = g
        self.w = w
        self.W = W
    def function(self,t,y):
        return np.array([y[1],self.A*np.cos(y[2])-self.w**2*np.sin(y[0])-self.g*y[1],self.W])
       
class Plotter(object):
    """
    Class for construction of plots necessary to visuzalize results for the damped driven pendulum
    system
    """
    def __init__(self,):
        pass
    @staticmethod
    def add_labels(ax,title,xlab,ylab):
        """
        Add labels and title to a plot
        """
        ax.set_title(title,fontsize=25)
        ax.set_xlabel(xlab,fontsize=25)
        ax.set_ylabel(ylab,fontsize=25)
        
    @staticmethod
    def add_trajectory(ax,time_array,traj_array,ltype='-',mtype='',alph=1.0):
        """
        Plot trajectory
        """
        ax.plot(time_array,traj_array,linestyle=ltype,marker=mtype,markersize=1,linewidth=1,color='k',alpha=alph)
    @staticmethod
    def bifurcation_plot(bif_ax,A_list,strobe_list1,strobe_list2):
        """
        Layout of bifurcation plot
        """
        bif_ax[0].plot(A_list,strobe_list1,linestyle='',marker='o',markerfacecolor='lightgrey',markersize=1)
        bif_ax[1].plot(A_list,strobe_list2,linestyle='',marker='o',markerfacecolor='lightgrey',markersize=1)
        bif_ax[0].set_xlabel(r'$A$',fontsize=25);bif_ax[0].set_ylabel(r'$\phi$',fontsize=25)
        bif_ax[1].set_xlabel(r'$A$',fontsize=25);bif_ax[1].set_ylabel(r'$\dot{\phi}$',fontsize=25)
        piax = np.array([np.pi*ii/4.0 for ii in xrange(0,8)])
        pilab = [r'$0$',r'$\pi/4$',r'$\pi/2$',r'$3\pi/4$',r'$\pi$',r'$5\pi$/4',r'$3\pi$/2',r'$7\pi$/4']
        bif_ax[0].set_yticks(piax)
        bif_ax[0].set_yticklabels(pilab)
    
    @staticmethod
    def plot_bundle(ax,time,tlist):
        """
        Layout for plot of trajectories initiated with different initial conditions
        """
        from matplotlib.pyplot import cm
        color=iter(cm.rainbow(np.linspace(0,1,len(tlist))))
        for ii in xrange(len(tlist)):
            ax.plot(tlist[ii][:,0],tlist[ii][:,1],linestyle='',
                    marker='o',color=next(color),markersize=3,
                    label=str(tlist[ii][0,0]))
        ax.set_xlabel(r'$\phi$')
        ax.set_ylabel(r'$\psi$')
        ax.set_xlim([-np.pi,np.pi])
        ax.legend()
            

class Stroboscopic(object):
    """
    Class containing functions necessary for generation of the stroboscopic map
    and visualization of the results
    """
    def __init__(self,):
        pass
    @staticmethod
    def stroboscopic_map(dt,time,traj,Omega,alt=1):
        """
        Generate stroboscopic map
        """
        tau = 2.0*np.pi/Omega
        tc = time[0]
        pos = [0,]
        for t in xrange(int(np.floor((time[-1]-time[0])/tau))):
          tc = tc + tau
          pos.append(np.argmin(np.abs(time-tc)))
        pos = np.array(pos)
        return traj[pos,0],traj[pos,1]
    
    @staticmethod
    def plot_stroboscobic_map_alt2(ax,s1,s2,prm):
        """
        Layout for visualization of stroboscopic map
        """
        ax.plot(s1,s2,marker='o',markersize=8,linestyle='',markerfacecolor='r')
        ax.set_xlim([-np.pi,np.pi])
        ax.set_ylim([-np.pi,np.pi])
        ax.set_xlabel('$\phi$')
        ax.set_ylabel(r'$\psi$')
        sup = 'Stroboscopic map: $\gamma = $ %s , $\omega = $ %s , $A = $ %s , $\Omega = $ %s ' % tuple(prm)  
        ax.set_title(sup)
    
    @staticmethod
    def plot_add_count(ax,vals1,vals2,cnt):
        """
        Add text label of number of occurences to plot
        """
        for ii in xrange(len(vals1)):
            plt.text(vals1[ii],vals2[ii],str(int(cnt[ii])))
        
    @staticmethod    
    def count_stroboscopic(s1,s2):
        """
        Count the number of times same value occurs in an array
        """
        rnd1 = np.round(s1,3)
        uni1 = np.unique(rnd1)
        cnt = np.zeros(uni1.shape[0])
        sax1, sax2 = np.zeros(uni1.shape[0]), np.zeros(uni1.shape[0])
        for ii in xrange(len(uni1)):
            pos = np.argmax(uni1[ii]==rnd1)
            sax1[ii],sax2[ii] = s1[pos],s2[pos]
            cnt[ii] = np.sum(rnd1 == uni1[ii])
        return sax1,sax2, cnt
    
    @staticmethod
    def wrap(arr):
        """
        Shift values by 2pi
        """
        pos = arr < 0.0
        arr[pos] = arr[pos] + 2*np.pi
        return arr
       
class Driven_Variables:
    """
    Different systems for the damped driven pendulum 
    """
    def __init__(self,gamma,omega):
        self.gamma, self.omega = gamma, omega
        self.good_1 = {'g': self.gamma,'w': self.omega,'A':0.2,'W': 0.3}
        self.good_2 = {'g': self.gamma,'w': self.omega,'A':1.0,'W':0.3}
        self.good_3 = {'g': self.gamma,'w': self.omega,'A':0.9,'W':2./3.}
        self.good_4 = {'g': self.gamma,'w': self.omega,'A':1.35,'W':2./3.}
        self.chaos_1 = {'g': self.gamma,'w': self.omega,'A':1.15,'W':2./3.}
        self.chaos_2 = {'g': self.gamma,'w': self.omega,'A':1.5,'W':2./3.}
        self.chaos_3 = {'g': self.gamma,'w': self.omega,'A':1.2,'W':2./3.}
        self.double_bif = {'g': self.gamma,'w': self.omega,'A':1.07,'W':2./3.}
        self.triple_bif = {'g': self.gamma,'w': self.omega,'A':1.47,'W':2./3.}
        self.arb_1 = {'g': self.gamma,'w': self.omega,'A':1.21,'W':0.32}
        self.arb_2 = {'g': self.gamma,'w': self.omega,'A':0.42,'W':1.32}
        
def Logistic_Map(r,x0,niter):
    """
    Function generating the logistic map. Given an initial value x0 and a value of the parameter r
    """
    x_list = [x0]
    lmap = lambda y: r*y*(1.-y)
    for ii in xrange(niter):
        x_list.append(lmap(x_list[-1]))
    return np.array(x_list)