"""
References:
- https://github.com/jdoepfert/roipoly.py
- https://goo.gl/fSpC1z
"""


from PyQt4.uic import loadUiType
from PyQt4.QtCore import *
from PyQt4.QtGui import *

from matplotlib.backend_bases import FigureManagerBase, key_press_handler

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)

Ui_MainWindow, QMainWindow = loadUiType('window.ui')

import matplotlib.pyplot as plt
import matplotlib.path as mplPath

import numpy as np
from SF_each_s import SF as sf
from SF_new import SF as sfn
from vec_corr import Vec_Corr as vc
        
class Main(QMainWindow, Ui_MainWindow):
    def __init__(self, ):
        super(Main, self).__init__()
        self.setupUi(self)
        self.fig_dict = {}
        self.fg_dict  = {}
        
        self.mplfigs.itemClicked.connect(self.changefig)
        self.datacube.clicked.connect(self.selectFile)
        self.polarization.clicked.connect(self.selectPol)
        self.fetch.clicked.connect(self._fetch)
        self.res.clicked.connect(self._res)
        self.calc.clicked.connect(self._calc)
        self.reset.clicked.connect(self._reset)

        self.xst.valueChanged[str].connect(self.xchg)
        self.yst.valueChanged[str].connect(self.ychg)
        self.x_st, self.y_st = 0,0

        fig = Figure()
        self.fig = fig
        self.addmpl(fig, np.ones((100,100)))

        self.previous_point = []
        self.tempxpt,self.tempypt = [],[]
        self.allxpoints = []
        self.allypoints = []
        self.start_point = []
        self.end_point = []
        self.line = None
        self.roicolor = 'r'
        self.ax = plt.gca()
        self.dcube_path = '/Users/Mipanox/Desktop/coding_temp/SF/wzError/L1455_rgd.fits'
        self.poldt_path = '/Users/Mipanox/Desktop/coding_temp/SF/wzError/scupollegacy_l1455_cube.fits'
        self.dc_nm.setText(self.dcube_path)
        self.po_nm.setText(self.poldt_path)

        self.dn = None
        self.ds = None
        
        self.__ID2 = self.fig.canvas.mpl_connect(
            'button_press_event', self.__button_press_callback)

    def selectFile(self):
        self.dc_nm.setText(QFileDialog.getOpenFileName())
        self.dcube_path = unicode(self.dc_nm.text())
        
    def selectPol(self):
        self.po_nm.setText(QFileDialog.getOpenFileName())
        self.poldt_path = unicode(self.po_nm.text())

    def xchg(self):
        self.x_st = self.xst.value()
    def ychg(self):
        self.y_st = self.yst.value()

    def _fetch(self):
        self.dn = sfn(ds=self.dcube_path,name='foo',od=2.,bn=1.,
                     pol=self.poldt_path,du=1.5e-5)
        self.ds = sf(ds=self.dcube_path,name='foo',od=2.,bn=1.,
                    pol=self.poldt_path,du=1.5e-5)
        m0 = self.dn.m0
        m1 = self.dn.m1
        
        i = 0
        for n in [m0,m1]:
            fig_ = Figure()
            axf_ = fig_.add_subplot(111)
            cax = axf_.imshow(n,origin='lower')
            fig_.colorbar(cax)
            name = 'moment %s' %i
            self.fig_dict[name] = fig_
            self.fg_dict[name] = n
            self.mplfigs.addItem(name)
            i += 1

    def __quiver(self,gd,po,plt,tx):
        lx,ly = gd.shape
        y,x = np.mgrid[0:(lx-1):(lx)*1j, 0:(ly-1):(ly)*1j]

        gx,gy = -np.sin(np.radians(gd)),np.cos(np.radians(gd))
        px,py = -np.sin(np.radians(po)+90.),np.cos(np.radians(po)+90.)

        quiveropts = dict(headlength=0, pivot='middle',
                          scale=5e1, headaxislength=0)
        plt.axis('equal');
        plt.quiver(x,y,gx,gy,color='r',**quiveropts)
        plt.quiver(x,y,px,py,color='b',alpha=0.5,**quiveropts)
    
    def _calc(self):
        def avg_adj(ar,n): # reshaping even-indexed arrays
            (M,N) = ar.shape
            tt = np.zeros((M-n+1,N-n+1))
            for (x,y),i in np.ndenumerate(ar):
                if x > ar.shape[0]-n or y > ar.shape[1]-n: continue
                else:
                    ap = ar[slice(x,x+n),slice(y,y+n)]
                    tt[x,y] = ap.mean()
            return tt

        pol = self.dn._grad(pol=1)[0]
        grd = self.ds._grad()

        for i in range(1,len(grd)):
            if i % 2:
                pt = avg_adj(pol,i+1)
            else:
                pt = pol[i/2:-i/2,i/2:-i/2]
            gt = grd[i]
            tx = '%s x %s' %(i+1,i+1)
            fig = Figure()
            self.fig_dict[tx] = fig
            self.fg_dict[tx] = [gt,pt]
            self.mplfigs.addItem(tx)
            axf = fig.add_subplot(111)
            self.__quiver(gt,pt,axf,tx)

    def _res(self, item):
        cg,cp = self.fg[0],self.fg[1]
        
        if self.x_st >= 0 and self.y_st >= 0:
            gd = np.pad(cg,((0,2*self.x_st),(0,2*self.y_st)),
                        mode='constant', constant_values=(np.nan))
            po = np.pad(cp,((self.x_st,self.x_st),(self.y_st,self.y_st)),
                        mode='constant', constant_values=(np.nan))
        elif self.x_st < 0 and self.y_st >= 0:
            gd = np.pad(cg,((-self.x_st,-self.x_st),(0,2*self.y_st)),
                        mode='constant', constant_values=(np.nan))
            po = np.pad(cp,((0,-2*self.x_st),(self.y_st,self.y_st)),
                        mode='constant', constant_values=(np.nan))
        elif self.x_st >=0 and self.y_st < 0:
            gd = np.pad(cg,((0,2*self.x_st),(-self.y_st,-self.y_st)),
                        mode='constant', constant_values=(np.nan))
            po = np.pad(cp,((self.x_st,self.x_st),(0,-2*self.y_st)),
                        mode='constant', constant_values=(np.nan))
        else:
            gd = np.pad(cg,((-self.x_st,-self.x_st),(-self.y_st,-self.y_st)),
                        mode='constant', constant_values=(np.nan))
            po = np.pad(cp,((0,-2*self.x_st),(0,-2*self.y_st)),
                        mode='constant', constant_values=(np.nan))

        if self.x_st == 0 and self.y_st == 0: pass
        else:
            tx = '%s - (%s x %s) shifted' %(str(self.mplfigs.currentItem().text()),
                                            self.x_st,self.y_st)
            fig = Figure()
            self.fig_dict[tx] = fig
            self.fg_dict[tx] = [gd,po]
            self.mplfigs.addItem(tx)
            axf = fig.add_subplot(111)
            self.__quiver(gd,po,axf,tx)
            
        
        if len(self.allxpoints) > 0: # if roi selected
            self.tempxpt,self.tempypt = self.allxpoints,self.allypoints
            old_xd,old_yd = self.dn.m0.shape
            new_xd,new_yd = gd.shape
            
            for i in range(len(self.allxpoints)):
                self.allxpoints[i] *= float(new_xd) / float(old_xd)
                self.allypoints[i] *= float(new_yd) / float(old_yd)
        
            tp = self.getMask(gd)
        
            gd[tp==False] = np.nan
            po[tp==False] = np.nan

        gx,gy = -np.sin(np.radians(gd)),np.cos(np.radians(gd))
        px,py = -np.sin(np.radians(po)),np.cos(np.radians(po))
        # +/- 90 doesn't matter
        
        v_c = vc(v1=np.array([gx,gy]),v2=np.array([px,py]))
        self.rho_c.setText('%3e' %(v_c.corr_c()) )
        self.rho_h.setText('%3e' %(v_c.corr_h()) )
        
        self.allxpoints = self.tempxpt
        self.allypoints = self.tempypt

    def _reset(self):
        self.previous_point = []
        self.tempxpt,self.tempypt = [],[]
        self.allxpoints = []
        self.allypoints = []
        self.start_point = []
        self.end_point = []
        self.line = None

        ## remember to right-click before reset if changed frame
        self.ax.lines = []
        self.changefig(self.mplfigs.currentItem())

        self.xst.setValue(0)
        self.yst.setValue(0)
            
    def changefig(self, item):
        text = str(item.text())
        self.rmmpl()
        self.addmpl(self.fig_dict[text], self.fg_dict[text])

    def addmpl(self, fig, fg):
        self.canvas = FigureCanvas(fig)
        self.mplvl.addWidget(self.canvas)
        self.fg = fg
        
        self.ax = self.fig.add_subplot(111)
        self.canvas.draw()
        self.canvas.setFocusPolicy( Qt.ClickFocus )
        self.canvas.setFocus()
        self.canvas.mpl_connect('button_press_event', self.__button_press_callback)
        
    def rmmpl(self,):
        self.mplvl.removeWidget(self.canvas)
        self.canvas.close()

    def __button_press_callback(self, event):
        if event.inaxes:
            x, y = event.xdata, event.ydata
            self.ax = event.inaxes
            if event.button == 1 and event.dblclick == False:  # If you press the left button, single click
                if self.line == None: # if there is no line, create a line
                    self.line = plt.Line2D([x, x],
                                           [y, y],
                                           marker='o',
                                           color=self.roicolor)
                    self.start_point = [x,y]
                    self.previous_point =  self.start_point
                    self.allxpoints=[x]
                    self.allypoints=[y]
                                                
                    self.ax.add_line(self.line)
                    self.canvas.draw()
                    # add a segment
                else: # if there is a line, create a segment
                    self.line = plt.Line2D([self.previous_point[0], x],
                                           [self.previous_point[1], y],
                                           marker = 'o',color=self.roicolor)
                    self.previous_point = [x,y]
                    self.allxpoints.append(x)
                    self.allypoints.append(y)
                                                                                
                    event.inaxes.add_line(self.line)
                    self.canvas.draw()
            elif ((event.button == 1 and event.dblclick==True) or
                  (event.button == 3 and event.dblclick==False)) and self.line != None: # close the loop and disconnect
                self.canvas.mpl_disconnect(self.__ID2) #joerg
                        
                self.line.set_data([self.previous_point[0],
                                    self.start_point[0]],
                                   [self.previous_point[1],
                                    self.start_point[1]])
                self.ax.add_line(self.line)
                self.canvas.draw()
                self.line = None
                                    
    def getMask(self, ci):
        ny, nx = ci.shape        
        poly_verts = [(self.allxpoints[0], self.allypoints[0])]
        for i in range(len(self.allxpoints)-1, -1, -1):
            poly_verts.append((self.allxpoints[i], self.allypoints[i]))
        
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T

        ROIpath = mplPath.Path(poly_verts)
        grid = ROIpath.contains_points(points).reshape((ny,nx))
        
        return grid

if __name__ == '__main__':
    import sys
    from PyQt4 import QtGui

    app = QtGui.QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())

