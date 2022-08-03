from scipy.signal import square, sawtooth
import numpy as np

def batman_scaled(sample_points = 10000, amplitude = 2.0):
	def H(x):
		return 0.5 * (1.0 + np.sign(x))

	def w(x):
		return 3.0*np.sqrt(1.0 - (x/7.0)**2)

	def l(x):
		return (6.0/7.0)*np.sqrt(10) + (3.0 + x)/2.0 - (3.0/7.0)*np.sqrt(10)*np.sqrt(4.0 - (x + 1.0)**2)

	def h(x):
		return 0.5*(3.0*(abs(x - 0.5) + abs(x + 0.5) + 6.0) - 11.0*(abs(x - 0.75) + abs(x + 0.75)))

	def r(x):
		return (6.0/7.0)*np.sqrt(10) + (3.0 - x)/2.0 - (3.0/7.0)*np.sqrt(10)*np.sqrt(4.0 - (x - 1.0)**2)

	def batman(x):
		return w(x) + (l(x) - w(x))*H(x + 3.0) + (h(x) - l(x))*H(x + 1.0) + (r(x) - h(x))*H(x - 1.0) + (w(x) - r(x))*H(x - 3.0)

	def scaled(x):
		x = x.astype(complex)
		return 1.0 - batman(x * 16.0 - 8.0).real / 1.5

	time = np.linspace(0, 1, sample_points, endpoint=True)
	bat = scaled(time)*amplitude*(-1)

	return time, bat

def sine(start= 0, stop = 2*np.pi, sample_points = 10000, amplitude = 2.0):
	time = np.linspace(start, stop, sample_points, endpoint=True)
	sine = np.sin(time)*amplitude

	return time, sine

def capacitor_charging(start = 0, stop = 10, sample_points = 10000, voltage = 1.0, resistor = 1000, capacity = 0.001):
	time = np.linspace(start, stop, sample_points, endpoint=True)
	voltage = voltage*(1 - np.exp(-time/(resistor*capacity)))

	return time, voltage

def capacitor_discharging(start = 0, stop = 10, sample_points = 10000, voltage = 1.0, resistor = 1000, capacity = 0.001):
	time = np.linspace(start, stop, sample_points, endpoint=True)
	voltage = voltage*np.exp(-time/(resistor*capacity))

	return time, voltage

def rectangular(start = 0, stop = 5, sample_points = 10000, frequency = 1.0):
	time = np.linspace(start, stop, sample_points, endpoint=True)

	return time, square(2*np.pi*time*frequency)

def sawtooth(start = 0, stop = 5, sample_points = 10000, frequency = 1.0):
	time = np.linspace(start, stop, sample_points, endpoint=True)

	return time, sawtooth(2*np.pi*time*frequency)