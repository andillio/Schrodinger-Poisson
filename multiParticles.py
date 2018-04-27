# lets compare solutions evolving
#	a multi hilbert space wavefunction
#	the wavefunction
# 	the density operator
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import animation
import time 
import sys
import math
import scipy.fftpack as sp 


# animation details
frames_ = 500 # number of frames in video
framesteps_ = 100 # min number of steps per frame

ofile_ = "Few_High_h_high_w.mp4"

n = 3 # number of particles

N = 256 # number of grid cells

dt = 1e-4

Mtot = 1.
mpart = 1./n
m = mpart

C = 1. # constant in Poisson's eqn

L = 1. # length of box
dx = L/N
width = dx*30

hbar_ = 1.0e-2 # 1e-2
hbar = hbar_*mpart

x = dx*(.5+np.arange(0, N))
kx = 2*np.pi*sp.fftfreq(N, d = dx)
u = hbar_*sp.fftshift(kx)

du = u[1] - u[0]


class sim(object):
	def __init__(self):
		self.Psi = None # a vector containing n vectors corresponding to psi_n
		self.psi = None
		self.rho = None

		self.HK = None # spectral representation for kinetic term in hamiltonian
		self.D2 = None # position space representation for kinetic term in hamiltonian
	
		self.T = None # total elapsed in simulation time

		self.start_ = 0
		self.time_text = None

		self.imM = None # multiparticle case
		self.imM_sp = None # lines for specific particles
		self.imS = None # Schroedinger case
		self.imN = None # von Neumann case

		self.axM = None 
		self.axS = None
		self.axN = None

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


def compute_phi(rho_r, fac = 500):
	rho_k = sp.fft(rho_r)
	phi_k = None
	kx = 2*np.pi*sp.fftfreq(len(rho_r), d = dx)
	with np.errstate(divide ='ignore', invalid='ignore'):
		# np.errstate(...) suppresses the divide-by-zero warning for k = 0
		phi_k = -(C*rho_k)/(kx**2)
	phi_k[0] = 0.0 # set k=0 component to 0
	phi_r = np.real(sp.ifft(phi_k))
	return phi_r


def update(i):

	for i in range(framesteps_):

		rhoM = mpart*(np.abs(s.Psi)**2).sum(axis = 0)
		for j in range(n):
			rhoM_ = rhoM - mpart*(np.abs(s.Psi[j,:])**2)
			V_ = mpart*compute_phi(rhoM_)

			s.Psi[j,:] *= np.exp(-1j*dt*V_/(2.*hbar))

			psi_k_ = sp.fft(s.Psi[j,:])
			psi_k_ *= np.exp(-1j*dt*hbar*(kx**2)/(2*mpart))
			s.Psi[j,:] = sp.ifft(psi_k_)

			s.Psi[j,:] *= np.exp(-1j*dt*V_/(2.*hbar))

		rho_r = Mtot*(np.abs(s.psi))**2
		phi_r = compute_phi(rho_r)
		Vr = phi_r*mpart # the PE field for an individual particle

		# update momentum half-step
		s.psi *= np.exp(-1j*dt*Vr/(2.*hbar))

		# update position full-step
		psi_k = sp.fft(s.psi)
		psi_k *= np.exp(-1j*dt*hbar*(kx**2)/(2*mpart))
		s.psi = sp.ifft(psi_k)

		# update momentum half-step
		s.psi *= np.exp(-1j*dt*Vr/(2*hbar))
		###### Neumann-Landau method
		rho_r = Mtot*(np.abs(np.diagonal(s.rho)))
		phi_r = compute_phi(rho_r)
		Vi, Vj = np.meshgrid(phi_r, phi_r)
		dV = Vi - Vj
		s.rho *= np.exp(-1j*dt*dV/(2.*hbar_))
		s.rho = sp.fft2(s.rho)
		s.rho *= s.HK
		s.rho = sp.ifft2(s.rho)
		s.rho *= np.exp(-1j*dt*dV/(2.*hbar_)) 
    ####################

        s.T += dt




def animate(i):

	s.time_text.set_text('$t=%.1f$' % s.T)

	rhoM = mpart*(np.abs(s.Psi)**2).sum(axis = 0)
	rhoS = Mtot*(np.abs(s.psi)**2)
	rhoN = Mtot*(np.abs(np.diagonal(s.rho)))

	s.imM.set_data(x,rhoM)
	if np.max(rhoM) != np.min(rhoM):
		s.axM.set_ylim(0, np.max(rhoM))
	s.imS.set_data(x, rhoS)
	if np.max(rhoS) != np.min(rhoS):
		s.axS.set_ylim(0, np.max(rhoS))
	s.imN.set_data(x, rhoN)
	if np.max(rhoN) != np.min(rhoN):
		s.axN.set_ylim(0, np.max(rhoN))

	for j in range(n):
		rhoM_ = mpart*np.abs(s.Psi[j])**2
		s.imM_sp[j].set_data(x, rhoM_)

	update(i)

	repeat_print(('%i hrs, %i mins, %i s remaining.' %remaining(i + 1, frames_)))

	return s.imM, s.imS, s.imN, s.time_text


def makeFig():

	fig, axs = plt.subplots(1,3, figsize = (18,4))
	s.axM, s.axS, s.axN = axs

	rhoM = mpart*(np.abs(s.Psi)**2).sum(axis = 0)
	rhoS = Mtot*(np.abs(s.psi)**2)
	rhoN = Mtot*(np.abs(np.diagonal(s.rho)))
#	print rhoM.sum()
#	print rhoS.sum()
#	print rhoN.sum()

	s.imM, = s.axM.plot(x, rhoM, 'k')
	s.axM.set_xlabel("x [L]")
	s.axM.set_ylabel(r"$\rho_x$")
	s.axM.set_title("Multi-particle Schrodinger")

	s.imM_sp = []
	for i in range(n):
		rhoM_ = mpart*np.abs(s.Psi[i])**2
		imM_sp_, = s.axM.plot(x, rhoM_, '--')
		s.imM_sp.append(imM_sp_)

	s.imS, = s.axS.plot(x, rhoS)
	s.axS.set_xlabel("x [L]")
	s.axS.set_title("Schrodinger")

	s.imN, = s.axN.plot(x, rhoN)
	s.axN.set_xlabel("x [L]")
	s.axN.set_title("Von Neumann")

	s.time_text = s.axM.text(.7,.9,'', ha='center', va='center', transform=s.axM.transAxes, bbox = {'facecolor': 'white', 'pad': 5})
	return fig


def getGauss(mu, s):
	arg = -.5*((x-mu)/float(s))**2 + 0j
	rval = np.exp(arg)
	rval /= np.sqrt(np.sum((np.abs(rval)**2))*dx)
	return rval


def setPsiM():
	s.Psi = np.zeros((n, len(x))) + 0j
	for i in range(n):
		psi = getGauss(L*(i+1)/float(n+1)/2. + L/4., width)
		s.Psi[i,:] = psi

def setKterms():
    Kj,Ki = np.meshgrid(-kx,kx)
    dK2 = Ki**2 - Kj**2
    s.HK = np.exp(hbar_*dK2*dt/2. *1.0j) #kinetic term for Hamiltonian

    I = np.identity(N)

    D2 = (-(1077749./352800)*I + (16./9)*np.roll(I,-1,1) + (16./9)*np.roll(I,1,1) - (14./45)*np.roll(I,2,1) - (14./45)*np.roll(I,-2,1) + (112./1485)*np.roll(I,3,1) + (112./1485)*np.roll(I,-3,1) - (7./396)*np.roll(I,4,1) - (7./396)*np.roll(I,-4,1))
    D2 += ((112./32175)*np.roll(I,-5,1) + (112./32175)*np.roll(I,5,1) - (2./3861)*np.roll(I,6,1) - (2./3861)*np.roll(I,-6,1) + (16./315315)*np.roll(I,7,1) + (16./315315)*np.roll(I,-7,1) - (1./411840)*np.roll(I,8,1) - (1./411840)*np.roll(I,-8,1)) 
    D2 *= -hbar_*hbar/(2.*dx**2)
    s.D2 = D2


def setPsiRho():
	s.psi = np.zeros(len(x)) + 0j
	s.rho = np.zeros((len(x),len(x))) + 0j
	for i in range(n):
		psi = getGauss(L*(i+1)/float(n+1)/2. + L/4., width)
		s.psi += psi
		s.rho += (1./n)*np.outer(psi, np.conj(psi))
#	psi_ = np.sqrt(np.abs(np.diagonal(s.rho))) + 0j
#	psi_ /= (np.abs(psi_)**2).sum()*dx
	s.psi /= np.sqrt((np.abs(s.psi)**2).sum()*dx)


def setICs():
	s.T = 0
	s.start_ = time.time()
	setKterms()
	setPsiM()
	setPsiRho()


def main(ofile):
    print "starting simulation: ", ofile
    time0 = time.time()
    setICs()
    fig = makeFig()

    ani = animation.FuncAnimation(fig, animate, frames = frames_, interval = 200, blit = True)

    # for writing the video
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=24, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(ofile, writer=writer)

    print 'completed in %i hrs, %i mins, %i s' %hms(time.time()-time0)
    print "output: ", ofile	


if __name__ == "__main__":
	main(ofile_)