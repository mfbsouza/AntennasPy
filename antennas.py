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
		self.delta = None
		self.N = None
		
		# constants
		self.j = 0 + 1j
		self.e = 8.854187817e-12

	def __Triangular_Base(self, z):
		if abs(z) <= self.delta:
			return (1 - abs(z)/self.delta)
		else:
			return 0
	
	def __z(self, n):
		return (-self.L/2 + n * self.delta)

	def __Iz(self, I_array, z, base_function=None):
		i = 0.0 + 0.0j
		for n in range(self.N):
			i = i + I_array[n,0] * base_function(z - self.__z(n+1))
		return i
	
	def __G(self, z):
		return (np.exp(-self.j*self.k*np.sqrt(z**2 + self.a**2))/(4*pi*np.sqrt(z**2 + self.a**2)))

	def __psi(self, m, n):
		if m == n:
			return ((1/(2*pi*self.delta))*np.log(self.delta/self.a) - self.j*self.k/(4*pi))
		else:
			return (self.__G(self.__z(m) - self.__z(n)))

	def __A(self, m, n):
		return (self.delta**2 * self.__psi(m,n))

	def __phi(self, m, n):
		return ( self.__psi(m-0.5,n-0.5) - self.__psi(m-0.5,n+0.5) - self.__psi(m+0.5,n-0.5) + self.__psi(m+0.5,n+0.5) )
		
	def compute_mom(self, N_list, max_interactions=50):
		currents_list = []
		impedances_list = []
		z_list = []

		for inter in range(max_interactions):
			self.N = 2*inter + 1
			self.delta = self.L/(self.N+1)
			Z = np.zeros( (self.N, self.N), dtype=complex )
			V = np.zeros( (self.N, 1), dtype=complex )

			# populate voltage matrix
			V[round(((self.N+1)/2)-1), 0] = -self.j * self.w * self.e * self.volt

			# populate impedance matrix
			for m in range(self.N):
				for n in range(self.N):
					Z[m,n] = self.k**2 * self.__A(m,n) - self.__phi(m,n)

			# solve the linear system
			I = np.linalg.solve(Z, V)

			# store currents, impedances and z values by a given N
			if self.N in N_list:

				# build current list
				currents = []
				for i in range(self.N + 2):
					current = self.__Iz(I, self.__z(i), self.__Triangular_Base)
					currents.append(abs(current)*1000)
				currents_list.append(currents)

				# build z/lambda list
				z = []
				for i in range(self.N + 2):
					z.append(self.__z(i))
				z_list.append(z)

				#build impedance list
				input_impedance = 1/I[round(((self.N+1)/2)-1),0]
				impedances_list.append(input_impedance)

		return [currents_list, impedances_list, z_list]

def demo():
	import matplotlib.pylab as plt
	max_inter = 10
	Ns_to_save = [1,3,5,7,9,11,13,15,19]
	antena = Linear_Antenna()
	current, impedance, z = antena.compute_mom(Ns_to_save, max_interactions=max_inter)

	# plotting values of the last term (N = 19)
	_, ax = plt.subplots()
	print(impedance[-1])
	lbl = "N = " + str(Ns_to_save[-1])
	ax.plot(z[-1],current[-1],label=lbl)
	ax.set(ylabel='|I| mA', xlabel='Z/λ')
	ax.grid()
	plt.legend()
	plt.show()

	# plotting values for all Ns
	_, ax = plt.subplots()
	print("")
	for idx in range(len(Ns_to_save)):
		print(impedance[idx])
		lbl = "N = " + str(Ns_to_save[idx])
		ax.plot(z[idx],current[idx], label=lbl)
	ax.set(ylabel='|I| mA', xlabel='Z/λ')
	ax.grid()
	plt.legend()
	plt.show()

	# plotting impedance change per N
	_, ax = plt.subplots()
	resistances = []
	reactances = []
	for number in impedance:
		resistances.append(np.real(number))
		reactances.append(np.imag(number))
	ax.plot(Ns_to_save, resistances, label='resistance')
	ax.plot(Ns_to_save, reactances, label='reactance')
	ax.set(xlabel='N')
	ax.grid()
	plt.legend()
	plt.show()

if __name__ == "__main__":
	demo()

