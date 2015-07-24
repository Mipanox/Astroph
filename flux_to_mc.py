import numpy as np
import cv2
from astropy import units as u
from astropy.constants import c, k_B, h, M_sun, m_p
from astropy.utils.data import get_readable_fileobj
from astropy.io import fits

class ds_flux(object):
    def __init__(self,line,T_ex,area,\
                 dis=None,beam=None,cell=None,linewidth=None, \
                 vrange=None,dataset=None,flux=None):
        self.Tx  = T_ex
        self.dis = dis
        self.vr  = vrange
        self.ll  = line
        self.pg  = area # polygon path if ds=True ;
                        # area (arcsec^2) if flux_T given
        
        ###--dictionary for line information--###
        # [Einstein A coeff., rotational constant B_e,
        #  rest freq., upper state J, X factor] 
        self.ln  = {"c18o(2-1)": [6.011*10**-7, 54.891*10**9, \
                                  220.*10**9, 2., 3.*10**-7],
                    "13co(3-2)": [2.181*10**-6, 55.101*10**9, \
                                  331.*10**9, 3., 2.19*10**-6],
                    "12co(3-2)": [2.492*10**-6, 57.636*10**9, \
                                  346.*10**9, 3., 1.68*10**-4]}
        if dataset:
            with get_readable_fileobj(dataset, cache=True) as f:
                self.fitsfile = fits.open(f)
                self.data     = self.fitsfile[0].data
                self.header   = self.fitsfile[0].header
            
            self.bmaj = self.header["bmaj"]*3600.
            self.bmin = self.header["bmin"]*3600.
            self.cx   = self.header["cdelt1"]*3600.
            self.cy   = self.header["cdelt1"]*3600.
            self.lw   = self.header["cdelt3"]/1000. # in km/s
        else:
            self.bmaj = beam[0]
            self.bmin = beam[1]
            self.cx   = cell[0]
            self.cy   = cell[1]
            self.lw   = linewidth

        if flux_T: self.fx = np.array(flux)
        else:      self.fx = self.flux()

    def polyarea(self):
    ###--return area of the polygon region in arcsec^2--###
        if type(self.pg) == int: return self.pg
        else:
        #mask of the polygon
            mk = np.zeros((self.data.shape[-1],self.data.shape[-2]))
            km = np.ones ((self.data.shape[-1],self.data.shape[-2]))
            cv2.fillConvexPoly(mk, np.array(self.pg), 1)

            mk  = mk.astype(np.bool)
            out = np.zeros_like(self.data[0,0])

            #how many " per pixel
            cv = np.abs(self.header["cdelt1"]*3600)
            return np.sum(km[mk])*cv**2
            
    def flux(self):
    ###--return flux in Jy/beam in the specified polygon--###
        #mask of the polygon
        mk = np.zeros((self.data.shape[-1],self.data.shape[-2]))
        cv2.fillConvexPoly(mk, np.array(self.pg), 1)

        mk  = mk.astype(np.bool)
        out = np.zeros_like(self.data[0,0])
        
        fx = []
        if self.vr:
            vii = int((self.vr[0] - self.header["crval3"]/1000.) \
                      /(self.header["cdelt3"]/1000.))-1
            vff = int((self.vr[1] - self.header["crval3"]/1000.) \
                      /(self.header["cdelt3"]/1000.))-1
            rg = range(vii,vff)
        else: rg = range(self.data.shape[-3])
        for zz in rg:
            out[mk] = self.data[0,zz][mk]
            fx.append(np.sum(out))
        return np.array(fx)

    def flux_density(self):
    ###--return flux density (Jy)---###
        #beam area = beam size / cell size
        bm = 2*np.pi*self.bmaj*self.bmin/ \
             (2*np.sqrt(2*np.log(2)))**2/(self.cx*self.cy)
        return self.fx / bm

    def rj_tb(self):
    ###--return temperature (Rayleigh-Jeans limit)--###
        #radians squared to arcsec squared
        ca = ((u.rad).to(u.arcsec))**2

        return self.flux_density()*c.value**2*10**-26/ \
               2/k_B.value/self.ln[self.ll][2]**2/ \
               self.polyarea()*ca
               
    ###--a simple function for convenience---###
    def f(self,x): return (np.exp(x)-1)**-1
    def tau(self):
    ###--return optical depth--###
        to = h.value/k_B.value*self.ln[self.ll][2]
        ex = to/self.Tx
        bg = to/2.725 #CMB temperature
        bt = (self.f(ex)-self.f(bg))**-1

        return -np.log(1-self.rj_tb()/to*bt)

    def tau_tot(self): return np.sum(self.tau())

    def col_d_tot(self):
    ###--return total column density of the line species (cm^-2)--###
        #channel width in Hz
        ftov = u.doppler_radio(self.ln[self.ll][2]*u.Hz)
        lw   = self.ln[self.ll][2]- \
               (self.lw*(u.km/u.s)).to(u.Hz,equivalencies=ftov).value
        
        #upper column density
        to = h.value/k_B.value*self.ln[self.ll][2]
        ex = to/self.Tx
        
        n_u = 8*np.pi*self.ln[self.ll][2]**2/(c.value*100.)**2* \
              self.f(ex)/self.ln[self.ll][0]* \
              np.sum(self.rj_tb())*self.lw

        #partition function, approximated by integral
        eu = k_B.value*self.Tx/h.value/self.ln[self.ll][1]
        zz = eu/(self.ln[self.ll][3]*2+1)* \
            np.exp(self.ln[self.ll][3]*(self.ln[self.ll][3]+1)/eu)

        return n_u*zz*10000.

    def mass(self):
    ###--return total mass (assuming mean molecular weight = 2.7 per H_2--###
        #conversion factor to M_sun
        cv = ((u.au)**2*(u.cm)**-2*(u.kg)).to(u.M_sun) \
             *2.7*2*m_p.value/self.ln[self.ll][4]

        return self.dis**2*self.col_d_tot()*self.polyarea()*cv
