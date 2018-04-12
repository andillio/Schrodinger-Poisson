import numpy as np 
import pylab as pyl 
import matplotlib.pyplot as plt
from matplotlib import animation
import time
import sys
import math
import scipy.fftpack as sp
import husimi_sp as hu
import husimi as hu_


# animation details
frames_ = 1000 # number of frames in video
framesteps_ = 100 # min number of steps per frame

ofile_ = "LandauSpectral.mp4"

CIC = True
max_ =True

order = 5
dt = 1.e-4 # fraction of T_scale (plasma periods)

L = 1.

n = 4*4096 # number of particles
N = 128 # number of grid cells
Mtot = 1.
mpart = 1./n

xg = (0.5+np.arange(N))/N # center of x grid cells
dx = L/N

lam =  1.#1./4.
v_th = .5 # units of u_c

########## CIC stuff ###################
C = -1.
omega0 = np.sqrt(-C/2.)

u_c = lam*omega0/(np.sqrt(2)*np.pi)

width = .08
v0 = .4

T_scale = np.pi*2./omega0
#######################################

########## Schr #######################
N_x = 256 # number of "pixels"
N_u = 256

dxS = L/N_x
dk = 2.*np.pi/L # kpc^-1

f = 5 # factor to multiply dx by in sig_x
sig_x = dxS*f

CS = -.000002#-.000004#-2.0e-4

hbar_ = 1.0e-6#
hbar = hbar_*mpart
eps0 = -4.*np.pi/CS * Mtot/L

omega0S = np.sqrt(-CS/2.)
u_cS = lam*omega0S/(np.sqrt(2)*np.pi)
T_scaleS = np.pi*2./omega0S

# create the line in x and k
x = dxS*(.5+np.arange(0, N_x))
#x = dx*(.5+np.arange(-N_x/2, N_x/2))
x_ = x.copy()
kx = 2*np.pi*sp.fftfreq(N_x, d = dxS)

u = hbar_*sp.fftshift(kx)
du = u[1] - u[0]

#Sa = (50/7)*np.array([-5,-4,-3,-2,-1,0, 1,2, 3,4,5])
#Sa = (50/4)*np.array([-7,-5,-3,-1,0, 1, 3,5,7])
Sa = np.arange(-200,201)#
v0S = v0
noise = 0


class sim(object):
	def __init__(self):
		self.r = None
		self.w = None
		self.n = None # an array containing the number of particles per stream
		self.v = None
		self.weight = None

		self.K = None
		self.K_ = None

		self.HK = None # spectral representation of kinetic term in hamiltonian

		self.psi_r = None
		self.rho = None
		self.rho_r = None
		self.unc = None

		self.T = 0
		self.t = []
		self.ECl = []
		self.EH = []
		self.EH_ = []
		self.start_ = 0
		self.time_text = None

		self.imCl = None
		self.imH = None
		self.imH_ = None

		self.imECl = None
		self.imEH = None
		self.imEH_ = None

		self.axCl = None
		self.axH = None
		self.axH_ = None

		self.axECl = None
		self.axEH = None
		self.axEH_ = None


s = sim()


def remaining(done, total):
	Dt = time.time() - s.start_
	return hms((Dt*total)/float(done) - Dt)

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


def fast_CIC_deposit(x,mi,Ngrid=N,periodic=1):
    """cloud in cell density estimator
    """
    if ((np.size(mi)) < (np.size(x))):
        m=x.copy()
        m[:]=mi
    else:
        m=mi

    dx = 1./Ngrid
    rho = np.zeros(Ngrid)
 
    left = x-0.5*dx
    right = left+dx
    xi = np.int32(left/dx)
    frac = (1.+xi-left/dx)
    ind = pyl.where(left<0.)
    frac[ind] = (-(left[ind]/dx))
    xi[ind] = Ngrid-1
    xir = xi.copy()+1
    xir[xir==Ngrid] = 0
    rho  = pyl.bincount(xi,  weights=frac*m, minlength=Ngrid)
    rho2 = pyl.bincount(xir, weights=(1.-frac)*m, minlength=Ngrid)

    rho += rho2
    
    return rho*Ngrid


def Phi_with_FFT_1D(density, N=128, C=1):
    """Calculate Phi (array with N elements) from nabla^2 Phi = C * rho
    Assuming a periodic domain 0..1 in one Dimension"""
    fd = density
    dx = 1./N
    lphi = np.zeros(N,dtype=complex)

    delta = np.fft.fft(fd)      # forward transform of density
    k = np.fft.fftfreq(len(fd)) # returns the wave numbers 
#    lphi[1:] = ((- C/(2.*math.pi)**2 *delta /((k/dx)**2)))[1:]
    lphi[1:] = ((- C/(2.*math.pi)**2 *delta /(pyl.sin(k)**2/dx**2)))[1:]   # including CIC shape
    lphi[0] = 0.  # zero out k=0 mode
    fPhi = (np.fft.ifft(lphi)).real 

    return fPhi


def a_from_Phi(Phi):
    """Calculate  - grad Phi  from Phi assuming a periodic domain
    domain the is 0..1 and dx=1./len(Phi)
    """
    N = len(Phi)
    dx = 1./N
    a = - central_difference(Phi)/dx
    return a


def CIC_acc(x,m,Ngrid,C):
    dx = 1./Ngrid
    xg = (0.5+np.arange(Ngrid))/Ngrid
    rho = fast_CIC_deposit(x,m,Ngrid)
#    rho = CIC_deposit(x,m,Ngrid)
    rho = rho - rho.mean()
    Phi = Phi_with_FFT_1D(rho, Ngrid, C)
    a = a_from_Phi( Phi)
    left = x-0.50000*dx
    xi = np.int64(left/dx)
    frac = (1.+xi-left/dx)
    ap = (frac)*(np.roll(a,0))[xi] + (1.-frac) * (np.roll(a,-1))[xi]
    return ap

def array_make_periodic(x,w):
    w[x>=1] += 1
    x[x>=1.] -= 1.

    w[x<0] -= 1 
    x[x<0.]  +=1.


def central_difference(y):
    """ Central difference:  (y[i+1]-y[i-1])/2 
    """
    return (np.roll(y,-1)-np.roll(y,1))/2


def acc(r, w, M, n_):

	acc_ = np.zeros(len(r))

	rL = np.minimum(r + w*L, np.roll(r, -1) + L*np.roll(w, -1))
	rR = np.maximum(r + w*L, np.roll(r, -1) + L*np.roll(w, -1))

	left = 0
	right = 0
	for i in range(len(n_)):
		n_i = n_[i]
		right += n_i - 1
		rL[right] = np.minimum(r[right] + w[left]*L, r[left] + (w[left] + 1)*L)
		rR[right] = np.maximum(r[right] + w[left]*L, r[left] + (w[left] + 1)*L)
		left += n_i

	dr = np.abs(rR - rL)

	rL = rL%L
	rR = rR%L

	in_ = rL < rR
	out_ = rL > rR

	for i in range(len(r)):

		r_ = r[i]

		acc_[i] += (np.minimum(np.maximum((r_ - rL)/dr,0), 1)[in_]).sum()*mpart # add all the inside
		
		acc_[i] += (np.minimum(np.maximum((r_ - rL)/dr,0), 1)[out_]).sum()*mpart # add all outside contributions (left sides only)
		acc_[i] += ((np.minimum(r_, rR)/dr)[out_]).sum()*mpart # contribution from right sides

		r_ += (dr/L).astype(int)*mpart/dr

	acc_ -= (M/L)*(r+L/2) # if adding this line doesnt work try again with average mean somewhere in loop
	acc_ -= np.mean(acc_)

	return -acc_*C


def indetermin(psi):

	delx = sig_x**2
	delu = hbar_**2 / (4.*sig_x**2) 

	HX = np.sqrt(delx)/dxS#np.sqrt(delX + delx)/dxS
	HU = np.sqrt(delu)/du#np.sqrt(delU + delu)/du

	return HX, HU 


# computes the potential from the wavefunction
def compute_phi(rho_r, fac = 500):
	rho_k = sp.fft(rho_r)
	phi_k = None
	kx = 2*np.pi*sp.fftfreq(len(rho_r), d = dxS)
	with np.errstate(divide ='ignore', invalid='ignore'):
		# np.errstate(...) suppresses the divide-by-zero warning for k = 0
		phi_k = -(CS*rho_k)/(kx**2)
	phi_k[0] = 0.0 # set k=0 component to 0
	phi_r = np.real(sp.ifft(phi_k))
	return phi_r


def update(i):
    iters_ = framesteps_#
    #iters_ = np.min([i+1, framesteps_])
    for i in range(iters_):
    ############### classical
        s.r += s.v*dt*T_scale/2.
        array_make_periodic(s.r, s.w)

        a = np.zeros(n)
        if CIC:	
            a = CIC_acc(s.r,mpart,N,C)
        else:
            a = acc(s.r, s.w, Mtot, s.n)
        s.v += a*dt*T_scale
    
        s.r += s.v*dt*T_scale/2.
        array_make_periodic(s.r, s.w)
    ###############
    ############### old husimi way
        # density, graviational potential, total potential energy
        rho_r = Mtot*(np.abs(s.psi_r))**2
        phi_r = compute_phi(rho_r)
        Vr = phi_r*mpart # the PE field for an individual particle
    
        # update momentum half-step
        s.psi_r *= np.exp(-1j*dt*T_scaleS*Vr/(2.*hbar))
    
        # update position full-step
        psi_k = sp.fft(s.psi_r)
        psi_k *= np.exp(-1j*dt*T_scaleS*hbar*(kx**2)/(2*mpart))
        s.psi_r = sp.ifft(psi_k)
    
        # update momentum half-step
        s.psi_r *= np.exp(-1j*dt*T_scaleS*Vr/(2*hbar))
    ################
    ################ new husimi
        
        rho_r = Mtot*(np.abs(np.diagonal(s.rho)))
        phi_r = compute_phi(rho_r) 
        Vi, Vj = np.meshgrid(phi_r, phi_r)
        dV = Vi - Vj
        s.rho *= np.exp(-1j*dt*T_scaleS*dV/(2.*hbar_))
        s.rho = sp.fft2(s.rho)
        s.rho *= s.HK
        s.rho = sp.ifft2(s.rho)
        s.rho *= np.exp(-1j*dt*T_scaleS*dV/(2.*hbar_))  
      
    ################
        s.T += dt
	s.unc = indetermin(s.psi_r)



def animate(i):

	s.time_text.set_text('$t=%.1f$' % s.T)

	H = hu.f_H(s.psi_r, s.K, dxS)
	H_ = hu_.f_H(s.rho, s.K_, x, u, hbar_, L)

	s.imCl.set_data(s.r, s.v/u_c)
	s.axCl.set_xlim(np.min(s.r), np.max(s.r))
	s.axCl.set_ylim(np.min(u/u_cS), np.max(u/u_cS))

	s.imH.set_array(H)
	if np.min(H) != np.max(H):
		s.imH.set_clim(vmin=np.min(H), vmax=np.max(H))

	s.imH_.set_array(H_)
	if np.min(H_) != np.max(H_):
		s.imH_.set_clim(vmin=np.min(H_), vmax=np.max(H_))

	s.t.append(s.T)

	ECl = (CIC_acc(s.r,mpart,N,C)/(-C))
	s.ECl.append(np.max(ECl))

	rho_r = Mtot*(np.abs(s.psi_r))**2
	phi_r = compute_phi(rho_r)
	E = (a_from_Phi(phi_r)/(-CS))
	s.EH.append(np.max(E))

	rho_r = Mtot*(np.abs(np.diagonal(s.rho)))
	phi_r = compute_phi(rho_r)
	E_ = (a_from_Phi(phi_r)/(-CS))
	s.EH_.append(np.max(E_))

	if max_:
		s.imECl.set_data(s.t,s.ECl)
		s.axECl.set_xlim(0, np.max(s.t))
		s.axECl.set_ylim(np.min(s.ECl), np.max(s.ECl))

		s.imEH.set_data(s.t,s.EH)
		s.axEH.set_xlim(0, np.max(s.t))
		s.axEH.set_ylim(np.min(s.EH), np.max(s.EH))

		s.imEH_.set_data(s.t,s.EH_)
		s.axEH_.set_xlim(0, np.max(s.t))
		s.axEH_.set_ylim(np.min(s.EH_), np.max(s.EH_))

	else:
		s.imECl.set_data(s.r, ECl)
		s.axECl.set_ylim(-.02, .02)

		s.imEH.set_data(x,E)
		s.axEH.set_ylim(-.02, .02)

		s.imEH_.set_data(x,E_)
		s.axEH_.set_ylim(-.02, .02)

	update(i)

	repeat_print(('%i hrs, %i mins, %i s remaining.' %remaining(i + 1, frames_)))

	return s.imCl, s.imH, s.imH_, s.imECl, s.imEH, s.imEH_, s.time_text


def makeFig():

	fig, axs = plt.subplots(2,3, figsize = (18,7))
	s.axCl, s.axH, s.axH_ = axs[0]
	s.axECl, s.axEH, s.axEH_ = axs[1]

	H = hu.f_H(s.psi_r, s.K, dxS)
	H_ = hu_.f_H(s.rho, s.K_, x, u, hbar_, L)

	s.imCl, = s.axCl.plot(s.r, s.v/u_c, '.', markersize=.5, color = 'k')
	s.axCl.set_ylim(np.min(u/u_cS), np.max(u/u_cS))
	s.axCl.set_ylabel(r'$u$ [$u_c$]')
	s.axCl.set_title(r"$f_{cl}(t,x,u_c)$")	

	s.imH = s.axH.imshow(H, interpolation = 'none', extent=[0, L, np.min(u)/u_cS,np.max(u)/u_cS], aspect = 'auto', origin = 'lower',cmap = 'viridis')
	s.axH.set_xlabel('$x$ [kpc]')
	s.axH.set_title(r'$f_{H}(t, x, u_c)$')

	s.imH_ = s.axH_.imshow(H_, interpolation = 'none', extent=[0, L, np.min(u)/u_cS,np.max(u)/u_cS], aspect = 'auto', origin = 'lower',cmap = 'viridis')
	s.axH_.set_xlabel('$x$ [kpc]')
	s.axH_.set_title(r'$f_{new}(t, x, u_c)$')

	s.t.append(0)

	ECl = CIC_acc(s.r,mpart,N,C)/(-C)
	s.ECl.append(np.max(ECl))

	rho_r = Mtot*(np.abs(s.psi_r))**2
	phi_r = compute_phi(rho_r)
	E = a_from_Phi(phi_r)/(-CS)
	s.EH.append(np.max(E))

	rho_r = Mtot*(np.abs(np.diagonal(s.rho)))
	phi_r = compute_phi(rho_r)
	E_ = a_from_Phi(phi_r)/(-CS)
	s.EH_.append(np.max(E_))

	if max_:
		s.imECl, = s.axECl.plot(s.t, s.ECl)
		s.imEH, = s.axEH.plot(s.t, s.EH)
		s.imEH_, = s.axEH_.plot(s.t, s.EH_)

		s.axECl.set_xlabel(r"$t [\omega_0^{-1}]$")
		s.axEH.set_xlabel(r"$t [\omega_0^{-1}]$")
		s.axEH_.set_xlabel(r"$t [\omega_0^{-1}]$")
		s.axECl.set_ylabel(r"$E_{max}/C$")
	else:
		s.imECl, = s.axECl.plot(s.r, ECl, '.')
		s.imEH, = s.axEH.plot(x, E)
		s.imEH_, = s.axEH_.plot(x, E_)

		s.axECl.set_xlabel(r"$x [L]$")
		s.axEH.set_xlabel(r"$x [L]$")
		s.axEH_.set_xlabel(r"$x [L]$")
		s.axECl.set_ylabel(r"$E/C$")

	s.time_text = s.axCl.text(.7,.9,'', ha='center', va='center', transform=s.axCl.transAxes, bbox = {'facecolor': 'white', 'pad': 5})

	#plt.show()
	return fig


def getPsiIC():

	psi = np.zeros(N_x) + 0j

	for i in range(len(s.weight)):
		w_ = s.weight[i]
		Sa_ = Sa[i]
		phi = -v0S*u_cS*L**2 / np.pi**2 *np.cos(2*np.pi*x/lam) + x*hbar_*np.pi*2.*Sa_/L
		psi_ = np.exp(phi/hbar_*1.j)
		psi_ /= (np.abs(psi_)**2).sum()*dxS
		psi += np.sqrt(w_)*psi_

	v = hbar_*4*np.pi*np.pi*Sa/(lam*L*np.sqrt(-CS))
	w = np.exp(-(v/v_th)**2 + 0j)
	w /= w.sum()

	s.rho = np.zeros((N_x, N_x)) + 0j

	for i in range(len(Sa)):
		w_ = w[i] 
#		w_ += np.random.uniform(0,w_*noise)
		Sa_ = Sa[i]
		phi = -v0S*u_cS*L**2 / np.pi**2 *np.cos(2*np.pi*x/lam) - x*hbar_*np.pi*2.*Sa_/L
		psi_ = np.exp(phi/hbar_*1.j)
		psi_ /= (np.abs(psi_)**2).sum()*dxS
		s.rho += w_*np.outer(psi_,np.conj(psi_))

	#s.rho = np.load("testFile.npy")
#	s.rho = s.rho*np.random.uniform(1-noise,1+noise,(N_x,N_x))

	return psi


def setICs(lam_):
    n_streams = len(Sa)
    y = np.array([])
    v = np.array([])
    s.w = np.zeros(n)
    s.weight = np.zeros(n_streams)
    s.n = np.zeros(n_streams)

    Kj,Ki = np.meshgrid(-kx,kx)
    dK2 = Ki**2 - Kj**2
    s.HK = np.exp(hbar_*dK2*dt*T_scaleS/2. *1.0j) #kinetic term for Hamiltonian

    for i in range(len(Sa)):
        S_ = Sa[i]
        if S_ != 0:
            v_ = hbar_*4*np.pi*np.pi*S_/(lam_*L*np.sqrt(-CS))
            w_ = np.exp(-(v_/v_th)**2)
            s.weight[i] = w_
        else:
            w_ = 1.
            s.weight[i] = w_

    s.weight /= s.weight.sum()
    s.n[Sa != 0] = (s.weight[Sa != 0]*n/s.weight.sum()).astype(int)
    s.n[Sa == 0] = n - s.n.sum()
    s.n = (s.n).astype(int)

    for i in range(len(Sa)):
        n_i = s.n[i]
        y_ = (i/float(n_streams) + np.arange(n_i))*L/float(n_i)
        S_ = Sa[i]
        v_ = np.ones(n_i)*hbar_*4*np.pi*np.pi*S_/(lam*L*np.sqrt(-CS))*u_c
        v = np.concatenate((v, v_),0)
        y = np.concatenate((y,y_),0)

    s.r = y
    s.v = v
    s.v += v0*u_c*pyl.sin(2.*np.pi*s.r/lam_)

    s.psi_r = getPsiIC()
    s.rho_r = Mtot*(np.abs(s.psi_r))**2

    s.K = hu.K_H(x, x_, u, hbar, mpart, sig_x, L)
    s.K_ = hu_.getKernel(kx, sig_x, hbar_) 

    s.T = 0
    s.start_ = time.time()

def main(ofile, lam_):
    time0 = time.time()
    setICs(lam_)
    fig = makeFig()

    ani = animation.FuncAnimation(fig, animate, frames = frames_, interval = 200, blit = True)

    # for writing the video
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(ofile, writer=writer)

    print 'completed in %i hrs, %i mins, %i s' %hms(time.time()-time0)
    print "output: ", ofile



if __name__ == "__main__":
	main(ofile_, lam)