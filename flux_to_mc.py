import numpy as np
import cv2
from astropy import units as u
from astropy.constants import c, k_B, h, M_sun, m_p
from astropy.utils.data import get_readable_fileobj
from astropy.io import fits

class ds_flux(object):
    def __init__(self,line,T_ex,area=None,sd=False,eta=None, \
                 dis=None,beam=None,cell=None,linewidth=None, \
                 vrange=None,dataset=None,flux=None):
        self.Tx  = T_ex
        self.dis = dis
        self.vr  = vrange
        self.ll  = line
        self.sd  = sd   # check if input is single-dish
        self.eta = eta
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
                self.header   = self.fitsfile[0].header
                self.data     = self.fitsfile[0].data
            
            self.bmaj = self.header["bmaj"]*3600.
            self.bmin = self.header["bmin"]*3600.
            self.cx   = np.abs(self.header["cdelt1"]*3600.)
            self.cy   = np.abs(self.header["cdelt1"]*3600.)
            self.lw   = np.abs(self.header["cdelt3"]/1000.) # in km/s
        else:
            self.bmaj = beam[0]
            self.bmin = beam[1]
            self.cx   = cell[0]
            self.cy   = cell[1]
            self.lw   = linewidth

        if flux: self.fx = np.array(flux)
        else:    self.fx = self.flux()

    def _poaa(self):
    ###--return polygon area in pixel number--###
        if self.header["naxis"] == 3: da = self.data
        else                        : da = self.data[0]
            
        #how many arcsec squared in one pixel
        ap = self.cx * self.cy
        if self.pg == None:     return da.shape[-1]*da.shape[-2]
        elif len(self.pg) == 1: return self.pg / ap
        else:
        #mask of the polygon
            mk = np.zeros((da.shape[-1],da.shape[-2]))
            km = np.ones ((da.shape[-1],da.shape[-2]))
            cv2.fillConvexPoly(mk, np.array(self.pg), 1)

            mk  = mk.astype(np.bool)
            out = np.zeros_like(da[0])
            
            return np.sum(km[mk])

    def polyarea(self):
    ###--return area of the polygon region in arcsec^2--###
        ap = self.cx * self.cy
        return self._poaa() * ap

    def _sm(self):
    ###--sum of pixel values--###
        if self.header["naxis"] == 3: da = self.data
        else                        : da = self.data[0]

        if self.vr:
            vii = int(self.header["crpix3"]+ \
                     (self.vr[0]-self.header["crval3"]/1000.) \
                      /(self.header["cdelt3"]/1000.))-1
            vff = int(self.header["crpix3"]+ \
                     (self.vr[1]-self.header["crval3"]/1000.) \
                      /(self.header["cdelt3"]/1000.))-1
            rg = range(vii,vff)
        else: rg = range(da.shape[-3])

        fx = []
        if self.pg and len(self.pg) > 1:
        #mask of the polygon
            mk = np.zeros((da.shape[-2],da.shape[-1]))
            cv2.fillConvexPoly(mk, np.array(self.pg), 1)
            
            mk  = mk.astype(np.bool)
            out = np.zeros_like(da[0])

            for zz in rg:
                out[mk] = da[zz][mk]
                fx.append(np.sum(out))
        else:
            for zz in rg:
                fx.append(np.sum(da[zz]))
        return np.array(fx)
            
    def flux(self):
    ###--return flux in Jy/beam in the specified polygon--###
        #radians squared to degrees squared
        cv = ((u.rad).to(u.deg))**2
        bm = 2*np.pi*self.bmaj*self.bmin/cv
        
        if self.sd: return self.flux_density()*bm
        else:       return self._sm()
            
    def flux_density(self):
    ###--return flux density (Jy)---###
        #radians squared to arcsec squared
        ca = ((u.rad).to(u.arcsec))**2

        #beam area = beam size / cell size
        bm = 2*np.pi*self.bmaj*self.bmin/ \
             (2*np.sqrt(2*np.log(2)))**2/(self.cx*self.cy)
             
        if self.sd:
            cv = 2*k_B.value*self.ln[self.ll][2]**2/c.value**2*10**26
            return self.rj_tb() * cv * self.polyarea() / ca
 
        else: return self.fx / bm

    def rj_tb(self):
    ###--return temperature (Rayleigh-Jeans limit)--###
        #radians squared to arcsec squared
        ca = ((u.rad).to(u.arcsec))**2

        if self.sd:
            return self._sm() / self.eta / self._poaa()
        else:
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
        lwd  = np.abs(self.lw)
        #channel width in Hz
        ftov = u.doppler_radio(self.ln[self.ll][2]*u.Hz)
        lw   = self.ln[self.ll][2]- \
               (lwd*(u.km/u.s)).to(u.Hz,equivalencies=ftov).value
        
        #upper column density
        to = h.value/k_B.value*self.ln[self.ll][2]
        ex = to/self.Tx
        
        n_u = 8*np.pi*self.ln[self.ll][2]**2/(c.value*100.)**2* \
              self.f(ex)/self.ln[self.ll][0]* \
              np.sum(self.tau())*lw
        #alternatively, one may use d(nu)/nu_0 = dv/c in linewidth:
        #n_u = 8*np.pi*self.ln[self.ll][2]/(c.value*100.)*\
        #      self.f(ex)/self.ln[self.ll][0]* \
        #      np.sum(self.tau())*lwd

        #partition function, approximated by integral
        eu = k_B.value*self.Tx/h.value/self.ln[self.ll][1]
        zz = eu/(self.ln[self.ll][3]*2+1)* \
             np.exp(self.ln[self.ll][3]*(self.ln[self.ll][3]+1)/eu)
        
        return n_u*zz

    def mass(self):
    ###--return total mass (assuming mean molecular weight = 2.7 per H_2--###
        #conversion factor to M_sun
        cv = ((u.au)**2*(u.cm)**-2*(u.kg)).to(u.M_sun) \
             *2.7*2*m_p.value/self.ln[self.ll][4]

        return self.dis**2*self.col_d_tot()*self.polyarea()*cv
