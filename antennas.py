from scipy.constants import pi
from scipy.constants import speed_of_light
import numpy as np

class Linear_Antenna:
	def __init__(self, frequency=300e6, radius=1e-4, lenght_factor=1/2, source_voltage=1) -> None:
		
		# antenna characteristics
		self.lbd = speed_of_light/frequency
		self.w = 2 * pi * frequency
		self.k = (2 * pi)/self.lbd
		self.a = radius * self.lbd
		self.L = self.lbd * lenght_factor
		self.volt = source_voltage
		
		# constants
		self.j = 0 + 1j
		self.e = 8.854187817e-12

	# triangular base function to aproximate the current's distribution in the antenna
	def __T(self, z_prime):
		if abs(z_prime) <= self.delta:
			return (1 - abs(z_prime)/self.delta)
		else:
			return 0
	
	# returns the z position (like L/2) for a given n
	def __Zn(self, n):
		return (-self.L/2 + n * self.delta)

	# calculate the current value for a given z position
	def __Iz(self, z_prime):
		iz = 0.0 + 0.0j
		for n in range(self.N):
			iz = iz + self.I[n,0] * self.__T(z_prime - self.__Zn(n+1))
		return iz
	
	def __G(self, z):
		return (np.exp(-self.j*self.k*np.sqrt(z**2 + self.a**2))/(4*pi*np.sqrt(z**2 + self.a**2)))

	def __psi(self, m, n):
		if m == n:
			return ((1/(2*pi*self.delta))*np.log(self.delta/self.a) - self.j*self.k/(4*pi))
		else:
			return (self.__G(self.__Zn(m) - self.__Zn(n)))

	def __A(self, m, n):
		return (self.delta**2 * self.__psi(m,n))

	def __phi(self, m, n):
		return ( self.__psi(m-0.5,n-0.5) - self.__psi(m-0.5,n+0.5) - self.__psi(m+0.5,n-0.5) + self.__psi(m+0.5,n+0.5) )
		
	def compute_mom(self, N):
		self.N = N
		self.delta = self.L/(self.N+1)
		excitation_point = round((self.N+1)/2 - 1)

		Z = np.zeros( (self.N, self.N), dtype=complex )
		V = np.zeros( (self.N, 1), dtype=complex )

		# populate voltage matrix
		V[excitation_point, 0] = -self.j * self.w * self.e * self.volt

		# populate impedance matrix
		for m in range(self.N):
			for n in range(self.N):
				Z[m,n] = self.k**2 * self.__A(m,n) - self.__phi(m,n)

		# solve the linear system
		self.I = np.linalg.solve(Z, V)

		# current distribution in the antenna
		current_distribution = list()
		for n in range(self.N + 2):
			current = self.__Iz(self.__Zn(n))
			current_distribution.append(current)

		# input impedance: Z = V/I
		input_impedance = self.volt/self.I[excitation_point, 0]

		# Zn list
		zn = list()
		for n in range(self.N + 2):
			zn.append(self.__Zn(n))

		return [current_distribution, input_impedance, zn]
		
def demo():
	import matplotlib.pylab as plt

	Max_N = 19
	inter = (Max_N - 1)/2 + 1
	antenna = Linear_Antenna()

	# plotting values of the last term (N = 19)
	current, impedance, zn = antenna.compute_mom(Max_N)
	print("input impedance: ")
	print(impedance, end="\n\n")
	print("max current:")
	print(current[round((Max_N+1)/2 - 1)], end=" A\n")

	abs_current = list()
	for element in current:
		abs_current.append(abs(element)*1000) # convert to mA
	print(abs_current[round((Max_N+1)/2 - 1)], end=" mA\n\n")

	_, ax = plt.subplots()
	ax.plot(zn, abs_current, label='N=19')
	ax.set(ylabel='|I| mA', xlabel='Z/λ')
	ax.grid()
	plt.legend()
	plt.show()

	# plotting values for all Ns
	abs_currents = list()
	impedances = list()
	zns = list()
	print("impedances:")
	for n in range(round(inter)):
		current, impedance, zn = antenna.compute_mom(2*n+1)
		zns.append(zn)
		print(impedance, end=" N = ")
		print(2*n+1)
		impedances.append(impedance)
		abs_current = list()
		for element in current:
			abs_current.append(abs(element)*1000) # convert to mA
		abs_currents.append(abs_current)
	
	_, ax = plt.subplots()
	for idx in range(len(abs_currents)):
		legend = "N = " + str(2*idx+1)
		ax.plot(zns[idx], abs_currents[idx], label=legend)
	ax.set(ylabel='|I| mA', xlabel='Z/λ')
	ax.grid()
	plt.legend()
	plt.show()

	# plotting impedance change per N
	_, ax = plt.subplots()
	resistances = list()
	reactances = list()
	Ns = list()
	idx = 0
	for number in impedances:
		resistances.append(np.real(number))
		reactances.append(np.imag(number))
		Ns.append(2*idx+1)
		idx = idx + 1
	ax.plot(Ns, resistances, label='resistance')
	ax.plot(Ns, reactances, label='reactance')
	ax.set(xlabel='N')
	ax.grid()
	plt.legend()
	plt.show()

if __name__ == "__main__":
	demo()

