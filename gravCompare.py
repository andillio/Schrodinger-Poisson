# a 1D version of 2D_updated.py

import numpy as np 
from matplotlib import animation
import matplotlib.pyplot as plt
import scipy.fftpack as sp
import husimi_sp as hu
import husimi as hu_
import time
import IC
import sys

# box parameters
N_x = 256# number of "pixels"
N_u = 256
L = 60. # kpc, length of side of box
dx = L/N_x # kpc, size of a single pixel
dxS = dx
dk = 2.*np.pi/L # kpc^-1
dt = 1e-1#4.0e-3#4.0e-4
periodic = False

n = 1024 # number of lagrangian particles
N = 100 # number of pixels in phase space diagram
N_ = 100 # number of pixels in coarse rho

# animation details
frames_ = 1000 # number of frames in video
framesteps_ = 100 # number of steps per frame
f = 5 # factor to multiply dx by in sig_x
sig_x = f*dx

#ofile_ = 'gaussian_cscale1D_{0}.mp4'.format(f) # name of output file
ofile_ = 'Compare_grav_slow.mp4'
animate_ = True

# all constants in the correct units
Mtot = 5e7#2.0e8#e7#1.0e12 # units of solar masses (Msun) 
G = 4.65e-12 # in units of kpc, Myr, Msun 
hbar = 1.76e-90 # in units of kpc, Myr, Msun 
mpart = (1e-22) # eV
mpart *= 9.0e-67 # convert to units of Msun
hbar_ = hbar/mpart
Npart = (Mtot/mpart)
m = Mtot/n

dX = L/N_
X = dX*(np.arange(-N_/2, N_/2)) # left edges of x cells

# create the line in x and k
x = dx*(.5+np.arange(-N_x/2, N_x/2))
x_ = dx*(.5+np.arange(-N_x/2, N_x/2))
kx = 2*np.pi*sp.fftfreq(N_x, d = dx)
#du = 5./N_u
#u = du*(.5+np.arange(-N_u/2, N_u/2))
u = hbar*sp.fftshift(kx)/mpart
du = u[1] - u[0]
kmax = np.max(kx)


class sim(object):
    def __init__(self):
        self.psi = None
        self.rho = None
        self.r = None
        self.v = None
        
        self.T = 0
        self.dt = dt
        self.start_ = None
        self.time_text = None
        self.rho_text = None
        
        self.K = None # Kernel for Michaels method
        self.K_ = None # Kernel for my method
        self.D2 = None
        
        self.imrhoCl = None
        self.axrhoCl = None
        
        self.imrhovCl = None
        self.axrhovCl = None
        
        self.imCl = None
        self.axCl = None
        
        self.imrhoH = None
        self.axrhoH = None
        
        self.imrhovH = None
        self.axrhovH = None
        
        self.imH = None
        self.axH = None
        
        self.imrho_ = None
        self.axrho_ = None
        
        self.im_ = None
        self.ax_ = None

        self.imrhov_ = None
        self.axrhov_ = None

                

s = sim()


# computes the potential from the wavefunction
def compute_phi(rho_r, fac = 500):
    # TODO: we know rho_r is real, so using rfft would be better
    if not(periodic):
        rho_r = np.concatenate((np.zeros(N_x*fac), rho_r, np.zeros(N_x*fac)),0)
    rho_k = sp.fft(rho_r)
    phi_k = None
    kx = 2*np.pi*sp.fftfreq(len(rho_r), d = dx)
    with np.errstate(divide ='ignore', invalid='ignore'):
        # np.errstate(...) suppresses the divide-by-zero warning for k = 0
        phi_k = -(4*np.pi*G*rho_k)/(kx**2)
    phi_k[0] = 0.0 # set k=0 component to 0
    phi_r = np.real(sp.ifft(phi_k))
    if periodic:
        return phi_r
    else:
        return phi_r[N_x*fac:N_x*(fac + 1)]


# returns y[i+1] - y[i]
# this means the last element will be the odd one out
def diff(y, axis = 0):
    dr = np.abs(np.roll(y, -1) - y)
    dr[dr > L/2.] -= L
    return np.abs(dr)

# returns av(y[i+1], y[i])
def av(y):
    ra = (np.roll(y,-1) + y)/2
    ra[-1] -= L/2
    if ra[-1] < -L/2:
        ra[-1] += L
    return ra

# returns rho defined at ra
def rho():
    dr = np.abs(diff(s.r))
    return m/dr


# returns acceleration defined at x_i+1/2
# G(r) is a linear interpolation of G(x) 
def a():
    G_ = -s.rho_.cumsum()*dX # this is now the gravity defined at x_i+1/2, the middle of each x cell
    G_ -= np.mean(G_)
    return G_*4*np.pi*G


# return value has units of y^2
def Delta2(P, y, dy):
    y_ = (P*y*dy).sum()
    return (P*dy*(y - y_)**2).sum()


def indetermin(psi):
    delx = sig_x**2
    delu = hbar_**2 / (4.*sig_x**2) 

    HX = np.sqrt(delx)/dxS#np.sqrt(delX + delx)/dxS
    HU = np.sqrt(delu)/du#np.sqrt(delU + delu)/du
    return HX, HU 


def acc():

    acc_ = np.zeros(len(s.r))

    i_R = np.argmax(s.r) # index of maximum r value

    rL = np.minimum(s.r, np.roll(s.r, -1)) # left and right boundary of cell for each unit of mpart
    rR = np.maximum(np.roll(s.r, -1), s.r) # does not include final cell

    dr = np.abs(rR - rL) # width of each cell not including final cell
    dr[i_R] = L - dr[i_R] # correct for reciprocal boundary conditions

    temp = rL[i_R]
    rL[i_R] = rR[i_R]
    rR[i_R] = temp

    acc_ += (np.min(s.r)/dr[i_R])*mpart # mass at left edge before first r value

    for i in range(len(s.r)):
        r_ = s.r[i]

        acc_[i] += np.minimum(np.maximum((r_ - rL)/dr,0),1).sum()*m


    if periodic:
        acc_ -= (Mtot/L)*(s.r+L/2)
    acc_ -= np.mean(acc_)

    return -acc_*4*np.pi*G



# needs to deal with boundary at dr[-1]
def rho_():
    Rho_ = np.zeros(N_)

    rL = np.minimum(s.r, np.roll(s.r, -1))[:-1] # left and right boundary of cell for each unit of mpart
    rR = np.maximum(np.roll(s.r, -1), s.r)[:-1] # does not include final cell
    dr = np.abs(rR - rL) # width of each cell not including final cell
    
    RL = s.r[-1] # bounds
    RR = s.r[0]
    DR = np.abs((RR - RL) - L)

    for i in range(N_):

        x_ = X[i]
        xL, xR = x_ - dX*.5, x_ + dX*.5
        
        dR = rR - xR
        
        bool_ = (dR > 0)
        Rho_[i] += ((m*np.minimum(np.maximum((xR - rL), 0), dX)/(dr*dX))[bool_]).sum()

        bool_ = (dR < 0)
        Rho_[i] += ((m*np.minimum(np.maximum((rR - xL), 0), dr)/(dr*dX))[bool_]).sum()

        Rho_[i] += np.minimum(np.maximum(RL - xL, 0),dX)*m/(DR*dX)
        Rho_[i] += np.minimum(np.maximum((xR - RR),0),dX)*m/(DR*dX)

    #print Rho_.sum()*dX/Mtot
    return Rho_


def quantum_phi(A):
    dA = np.roll(A,-1) - A
    ddA = dA - np.roll(dA,1)
    ddA = ddA/A
    VQ = -ddA*hbar*hbar_/(2*dxS*dxS)
    return VQ

 
def L_(rho):        
        rho_r = Mtot*(np.abs(np.diagonal(rho))) # mass density
        phi_r = compute_phi(rho_r)
        Vr = np.diag(phi_r*mpart)

        #H = D + Vr
        H = (s.D2 + Vr)*s.dt

        #print np.max(s.rho), np.max(H), (np.abs(np.diagonal(s.rho))).sum()*dxS
        comm = np.matmul(H,rho) - np.matmul(rho, H)
        return comm*1j/hbar

def update(i):

    for framestep in xrange(framesteps_):
        # density, graviational potential, total potential energy
        rho_r = Mtot*(np.abs(s.psi))**2
        phi_r = compute_phi(rho_r) #- Mtot*quantum_phi(np.abs(s.psi_r))*quantum_potential_correct
        Vr = phi_r*mpart # the PE field for an individual particle

        # update position half-step
        s.psi *= np.exp(-1j*dt*Vr/(2.*hbar))
        
        # update momentum full-step
        psi_k = sp.fft(s.psi)
        psi_k *= np.exp(-1j*dt*hbar*(kx**2)/(2*mpart))
        s.psi = sp.ifft(psi_k)

        # update position half-step
        s.psi *= np.exp(-1j*dt*Vr/(2*hbar))

        s.T += dt

        s.r += s.dt/2. * s.v
        
        s.v += acc()*s.dt

        s.r += s.dt/2. * s.v
  #####################
        U1 = s.rho + L_(s.rho)
    
        U2 = (3./4)*s.rho + (1./4)*U1 + (1./4)*L_(U1)
    
        s.rho = (1./3)*s.rho + (2./3)*U2 + (2./3)*L_(U2)


def remaining(done, total):
    dt = time.time() - s.start_
    return hms((dt*total)/float(done) - dt)

# given a time T in s
# returns (hours, mins, secs) remaining
def hms(T):
    r = T
    hrs = int(r)/(60*60)
    mins = int(r%(60*60))/(60)
    s = int(r%60)
    return (hrs, mins, s)


def repeat_print(string):
    sys.stdout.write('\r' +string)
    sys.stdout.flush()


def animate(i):
    s.time_text.set_text('$t=%.1f$ Myr' % s.T)
    r2 = np.diagonal(np.matmul(s.rho,s.rho)).sum()*dx*dx # gets r2
    s.rho_text.set_text(r'$\rho^2 = %.3f$' % r2)
    
    H = hu.f_H(s.psi, s.K, dx)
    H_ = hu_.f_H(s.rho, s.K_, x, u, hbar_, L,1)
    
    heatmap, xedges, yedges = np.histogram2d(s.r, s.v, bins = N, range=[[-L/2,L/2],[np.min(u), np.max(u)]])
    s.imCl.set_array(heatmap.T)
    if np.min(heatmap.T) != np.max(heatmap.T):
        s.imCl.set_clim(vmin=np.min(heatmap.T), vmax=np.max(heatmap.T))
    
    s.imH.set_array(H)
    if np.min(H) != np.max(H):
        s.imCl.set_clim(vmin=np.min(H), vmax=np.max(H))

    s.im_.set_array(H_)
    if np.min(H_) != np.max(H_):
        s.im_.set_clim(vmin=np.min(H_), vmax=np.max(H_))
        
    rho = rho_()
    s.imrhoCl.set_data(X, rho)
    s.axrhoCl.set_xlim(np.min(X), np.max(X))
    s.axrhoCl.set_ylim(np.min(rho), np.max(rho))

    nv, binsv = np.histogram(s.v, bins = N, range=(np.min(u), np.max(u)))
    dv = (np.max(u) - np.min(u))/N
    nv = nv*Mtot/float(n*dv)
    v_ = binsv[:-1]
    s.imrhovCl.set_data(v_, nv)
    s.axrhovCl.set_xlim(np.min(v_), np.max(v_))
    s.axrhovCl.set_ylim(np.min(nv), np.max(nv))
    
    #rho = Mtot*H.sum(axis = 0)*du
    rho = Mtot*(np.abs(s.psi))**2
    s.imrhoH.set_data(x, rho)
    s.axrhoH.set_xlim(np.min(x), np.max(x))
    s.axrhoH.set_ylim(np.min(rho), np.max(rho))
    
    rho = Mtot*H.sum(axis = 1)*dx
    s.imrhovH.set_data(u, rho)
    s.axrhovH.set_xlim(np.min(u), np.max(u))
    s.axrhovH.set_ylim(np.min(rho), np.max(rho))
    
    #rho = Mtot*H_.sum(axis = 0)
    rho = Mtot*(np.abs(np.diagonal(s.rho)))
    s.imrho_.set_data(x, rho)
    s.axrho_.set_xlim(np.min(x), np.max(x))
    s.axrho_.set_ylim(np.min(rho), np.max(rho))
    
    rho = Mtot*H_.sum(axis = 1)
    s.imrhov_.set_data(u, rho)
    s.axrhov_.set_xlim(np.min(u), np.max(u))
    s.axrhov_.set_ylim(np.min(rho), np.max(rho))
    
    update(i)
    repeat_print(('%i hrs, %i mins, %i s remaining.' %remaining(i + 1, frames_)))

    return s.imCl, s.imrhoCl, s.imrhovCl, s.imH, s.imrhoH, s.imrhovH, s.im_, s.imrho_, s.imrhov_, s.time_text, s.rho_text
    

def makeFig():
    fig, axs = plt.subplots(3,3, figsize = (18,18))
    s.axCl, s.axrhoCl, s.axrhovCl = axs[0]
    s.axH, s.axrhoH, s.axrhovH = axs[1]
    s.ax_, s.axrho_, s.axrhov_ = axs[2]

    s.time_text = s.axrhoCl.text(.7,.9,'', ha='center', va='center', transform=s.axrhoCl.transAxes, bbox = {'facecolor': 'white', 'pad': 5})
    s.rho_text = s.axrho_.text(.7,.9,'', ha='center', va='center', transform=s.axrho_.transAxes, bbox = {'facecolor': 'white', 'pad': 5})

    H = hu.f_H(s.psi, s.K, dx)
    H_ = hu_.f_H(s.rho, s.K_, x, u, hbar_, L)
    
    s.axCl.set_title(r"$f_{Cl}$")
    s.axH.set_title(r"$f_{H}$")
    s.ax_.set_title(r"$f_{new}$")
    s.axCl.set_ylabel("u [kpc/Myr]")
    s.axH.set_ylabel("u [kpc/Myr]")
    s.ax_.set_ylabel("u [kpc/Myr]")
    s.ax_.set_xlabel("x [kpc]")
    s.axrhoCl.set_title(r"$\rho_x$")
    s.axrho_.set_xlabel("x [kpc]")
    s.axrhovCl.set_title(r"$\rho_u$")
    s.axrhov_.set_xlabel("u [kpc/Myr]")
    
    heatmap, xedges, yedges = np.histogram2d(s.r, s.v, bins = N, range=[[-L/2,L/2],[np.min(u), np.max(u)]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    s.imCl = s.axCl.imshow(heatmap.T, extent = extent, origin = 'lower', aspect = 'auto',cmap = 'viridis')
    
    s.imH = s.axH.imshow(H, interpolation = 'none', extent=[-L/2, L/2, np.min(u),np.max(u)], aspect = 'auto', origin = 'lower',cmap = 'viridis')
    
    s.im_ = s.ax_.imshow(H_, interpolation = 'none', extent=[-L/2, L/2, np.min(u),np.max(u)], aspect = 'auto', origin = 'lower',cmap = 'viridis')
    
    s.imrhoCl, = s.axrhoCl.plot(X, rho_())
    
    nv, binsv = np.histogram(s.v, bins = N, range=(np.min(u), np.max(u)))
    dv = (np.max(u) - np.min(u))/N
    nv = nv*Mtot/float(n*dv)
    v_ = binsv[:-1]
    s.imrhovCl, = s.axrhovCl.plot(v_, nv)
    
#    rho = Mtot*H.sum(axis = 0)*du
    rho = Mtot*(np.abs(s.psi))**2
    s.imrhoH, = s.axrhoH.plot(x, rho)
    
    rho = Mtot*H.sum(axis = 1)*dx
    s.imrhovH, = s.axrhovH.plot(u, rho)
    
#    rho = Mtot*H_.sum(axis = 0)
    rho = Mtot*(np.abs(np.diagonal(s.rho)))
    s.imrho_, = s.axrho_.plot(x, rho)
    
    rho = Mtot*H_.sum(axis = 1)
    s.imrhov_, = s.axrhov_.plot(u, rho)
    
    return fig
    


def main(sig_x, ofile):

    time0 = time.time()
    # make the ICs
    s.psi = IC.gauss_r(x, 0, 10, dx)
    s.K = hu.K_H(x, x_, u, hbar, mpart, sig_x, L)
    
    s.rho = np.outer(s.psi,np.conj(s.psi))
    #s.rho /= np.diagonal(s.rho).sum()
    s.K_ = hu_.getKernel(kx, sig_x, hbar_)
    I = np.identity(N_x)
    #D2 = -2*I + np.roll(I,-1,1) + np.roll(I,1,1)
    #D2 = (-30*I + 16*np.roll(I,-1,1) + 16*np.roll(I,1,1)- 1*np.roll(I,2,1)- 1*np.roll(I,-2,1))/12. 
    #D2 = (-(205./72)*I + (8./5)*np.roll(I,-1,1) + (8./5)*np.roll(I,1,1) - (1./5)*np.roll(I,2,1) - (1./5)*np.roll(I,-2,1) + (8./315)*np.roll(I,3,1) + (8./315)*np.roll(I,-3,1) - (1./560)*np.roll(I,4,1) - (1./560)*np.roll(I,-4,1)) 
    D2 = (-(1077749./352800)*I + (16./9)*np.roll(I,-1,1) + (16./9)*np.roll(I,1,1) - (14./45)*np.roll(I,2,1) - (14./45)*np.roll(I,-2,1) + (112./1485)*np.roll(I,3,1) + (112./1485)*np.roll(I,-3,1) - (7./396)*np.roll(I,4,1) - (7./396)*np.roll(I,-4,1))
    D2 += ((112./32175)*np.roll(I,-5,1) + (112./32175)*np.roll(I,5,1) - (2./3861)*np.roll(I,6,1) - (2./3861)*np.roll(I,-6,1) + (16./315315)*np.roll(I,7,1) + (16./315315)*np.roll(I,-7,1) - (1./411840)*np.roll(I,8,1) - (1./411840)*np.roll(I,-8,1)) 
    D2 *= -hbar_*hbar/(2.*dxS**2) 

    s.D2 = D2

    s.r = IC.cl_gauss(0, 10, -L/2., L/2., n)
    s.v = np.zeros(n)

    # loop until done
    s.T = 0.0
    s.start_ = time.time()

    if animate_:

        fig = makeFig()

        ani = animation.FuncAnimation(fig, animate, frames=frames_, interval=200, blit=True)

        # for writing the video
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)
        ani.save(ofile, writer=writer)

    else:

        for i in range(frames_):
            update(i)

    print '\ncompleted in %i hrs, %i mins, %i s' %hms(time.time()-time0)
    print "outputed: ", ofile

if __name__ == "__main__":
    sig_x = dx*f
    main(sig_x, ofile_)