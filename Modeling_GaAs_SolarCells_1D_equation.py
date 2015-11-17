from __future__ import division
from numpy import arange, pi,array, zeros
from pylab import plot, savefig, legend
import numpy as np
from astropy.constants import e,eps0,h,m_e
from math import log, sqrt, exp
from astropy import units as u
from scipy import sparse

import matplotlib
import matplotlib.pyplot as plt

class GaAsSolver:
    def __init__(self):
        # Init Constants (SI units)
        self.q=1.6022e-19 * u.C # charge of electron 
        self.k_b=1.3806503e-23 * ((u.N*u.m)/u.K) #Boltzman constant
        self.eps_0=8.854187817e-12 * (u.C**2/(u.m**2*u.N))#permitvity constant for free space 
        self.T= 300.0 * u.K #room temprature 
        
        #GaAs is the materials to be used for this study and below are the values for different variables 
        self.eps_GaAs=12.93 #dielectric constant (static not at high frequency)
        self.epsGS = self.eps_GaAs*self.eps_0
        self.N_A=5e+23*(u.m**-3) #Charge density for acceptors
        self.N_D=2e+24*(u.m**-3) #Charge density for donors
        self.N= 2000
        self.Tol=1e-12
        
        # two of the three normalizing constants 
        self.ni= 1.8e12*(u.m**-3) #internsinic charge density 
        self.Vt= self.k_b*self.T/self.q # Thermal Voltage to be used for potential normilzation 
        
        print 'Vt='+str(self.Vt)
        # allocate arrays
        self.Vo=np.zeros(self.N)
        self.n=np.zeros(self.N)
        self.p=np.zeros(self.N)
        self.bt=np.zeros(self.N)
        self.alpha=np.zeros(self.N)
        self.a=np.zeros(self.N)
        self.b=np.zeros(self.N)
        self.c=np.zeros(self.N)
        self.f=np.zeros(self.N)
        self.y=np.zeros(self.N)
        
    def compute_potentials(self):
        # compute Build up potentials  
        self.Vbi = self.k_b*self.T*log((self.N_A*self.N_D)/(self.ni*self.ni))/self.q# Barrier potential, an intially established potential difference due to the electric field formed 
        self.VbiT= log((self.N_A*self.N_D)/(self.ni*self.ni)) # after normalization of Vbi with Vt
        
        print self.VbiT
        print self.Vbi
        
    def compute_depwdz(self):
        # compute the deplation width
        self.xno=sqrt((2*self.epsGS*self.N_A*self.VbiT*u.N/(u.m*u.C))/(self.q*self.N_D*(self.N_A+self.N_D)))  #deplation widith for n in m  
        self.xpo=sqrt((2*self.epsGS*self.N_D*self.VbiT*u.N/(u.m*u.C))/(self.q*self.N_A*(self.N_A+self.N_D)))  #deplation widith for p in m
        
        print "xno: " + str(self.xno)
        print "xpo: " + str(self.xpo)
        
        self.W=self.xno+self.xpo
        print "W: " + str(self.W)
                            
        #Debye length to be used for normalizing mesh size 
        self.Ldn=sqrt(self.epsGS*self.k_b*self.T*u.m**-2/(self.q*self.q*self.N_D))*u.m                     #Deybe length for donors
        self.Ldp=sqrt(self.epsGS*self.k_b*self.T*u.m**-2/(self.q*self.q*self.N_A))*u.m                    #Deybe length for acceptors
        self.Ldi=sqrt(self.epsGS*self.Vt*u.m**-2/(self.q*self.ni))*u.m                          #Deybe length for internsinic carrier concentration (which means when it is undoped)
        
        #Normalized width of deplation 
        self.WL=self.W*u.m/self.Ldn                 # after normalization of W with Ldi
        
        #outside -xpo<x<0 it is p region and 0<xno is the n region. Outside this region the is netural due to diffusion 
 
        self.x=self.WL #Since meshe length is limited by Debye length; it should be normalized 
        
        dx = self.x/(self.N-1);
        self.h=dx/10 # mesh size 
        self.h2=self.h*self.h
        
        print self.h2
        
    def get_dopingcc(self):
        self.C=np.zeros(self.N) # Doping concentration is different for p and  n region,
        #charges are normlized by N_D        
        for i in range (0,self.N):
            if i<= (self.N/2):
                self.C[i] = - self.N_A/self.N_D        
            else: 
                self.C[i] = self.N_D/self.N_D
        
    def init_bcnds(self):
        self.Vo[self.N-1]=log((1/2)+sqrt(1+(1/4)))
        self.Vo[0]=log(-(self.N_A/(2*self.N_D))+sqrt((((self.N_A/(2*self.N_D))**2))+((self.N_D/self.N_D)**2)))# Potential at ohmic contact between depltion and p-region, after normailzed by Vt
        
        self.a[0] = 0
        self.c[0] = 0
        self.b[0] = 1
        self.f[0] = self.Vo[0]
        self.a[self.N-1] = 0
        self.c[self.N-1]=0
        self.b[self.N-1] = 1
        self.f[self.N-1] = self.Vo[self.N-1]
        
        # set the rest of the initial values
        for i in range (1,self.N-1):
            self.a[i]=1/self.h2
            self.b[i]=-(2/self.h2+(exp(self.Vo[i])+exp(-self.Vo[i])))
            self.c[i]=1/self.h2
            self.f[i]=exp(self.Vo[i])-exp(-self.Vo[i])-self.C[i]-self.Vo[i]*(exp(self.Vo[i])+exp(-self.Vo[i]))
        
    def solve(self):
        #LU decompostion and iteration for possion
        #step 1, page 31,manuel-equality of L and U        
        taw=0  
        while not taw==1:
            self.alpha[0]=self.b[0]
            self.bt[0]=0 
            for k in range (1,self.N):
                self.bt[k]=self.a[k]/self.alpha[k-1]
                self.alpha[k]=self.b[k]-self.bt[k]*self.c[k-1]
            
            print self.bt          
            
            #step 2.1,page 31,solving for Lg=f
            g=np.zeros(self.N)
            g[0]=self.f[0]
            g[self.N-1]=self.f[self.N-1]-self.bt[self.N-1]*g[self.N-1]
            #g[N-1]=f[N-1]
            #g[N-1]=Vo[N-1]*alpha[N-1] 
            
            for k in range (1,self.N-1):
                g[k]=self.f[k]-self.bt[k]*g[k-1] 
        
            print 'f'
            print self.f
            print g
        
            
            #step 2.2,page 31,solving for U*Vo=g; from n-1,n-2...2,1 
            delta=np.zeros(self.N)            
            #last=Vo[N-1]
            #Vo[N-1]=0.5 #g[N-1]/alpha[N-1]            
            last=g[self.N-1]/self.alpha[self.N-1]    
            delta[self.N-1] = last- self.Vo[self.N-1] # differnce between
            self.Vo[self.N-1]=last 
            
            for i in range (self.N-2,-1,-1):
                last=(g[i]-self.c[i]*self.Vo[i+1])/self.alpha[i]
                delta[i]=last-self.Vo[i]
                self.Vo[i]=last
                
            #Finding the maximum delta             
            delta_max = 0
            delta_max = np.abs(delta).max()   
                
            if delta_max < self.Tol:
                taw = 1        
            else:
                for i in range (1,self.N-1):
                    self.b[i]=-(2/self.h2+(exp(self.Vo[i])+exp(-self.Vo[i])))
                
                    self.f[i]=exp(self.Vo[i])-exp(-self.Vo[i])-self.C[i]-self.Vo[i]*(exp(self.Vo[i])+exp(-self.Vo[i]))
            
    def plot_res(self):
        n = np.exp(self.Vo) # note: Vo is an array no need for a for loop
        p = np.exp(-self.Vo)   
        
        xn=self.xno*u.m/self.Ldn
        xp=self.xpo*u.m/self.Ldn
        print 'xn='+str(xn)
        print 'xp='+str(xp)
        print 'h='+str(self.h)
        print 'Np='+str(xp/self.h)
        print 'Nn='+str(xn/self.h)
        self.y[0]=0
        for i in range (1,self.N):
            self.y[i]=self.y[i-1]+self.h
                 
        fig = plt.figure()
        
        plt.semilogy(self.y,p*self.N_D,label='p')
        
        plt.semilogy(self.y,n*self.N_D,label='n')
        plt.xlabel('Deplation Width')
        plt.ylabel('Carrier Density')        
        
        '''
        plt.plot(y,Vo*Vt)
        plt.xlabel('Deplation Width')
        plt.ylabel('Potential')
        '''
        '''
        plt.plot(y,C,label='C')
        plt.xlabel('Deplation Width')
        plt.ylabel('Doping Concentration')
        '''        
        legend()        
        fig.suptitle('2000')
        plt.show()  
        
        print self.Vo  
        
if __name__ == "__main__":
    gas = GaAsSolver()
    gas.compute_potentials()    
    gas.compute_depwdz()
    gas.get_dopingcc()
    gas.init_bcnds()
    gas.solve()
    gas.plot_res()
    
