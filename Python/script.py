from scipy.interpolate import CubicSpline
import numpy as np

from more_itertools import windowed
from tqdm.contrib import tzip

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = ["Latin Modern Roman"]
plt.rcParams['text.latex.preamble'] = r'\usepackage[locale=US,per-mode=symbol,separate-uncertainty,sticky-per]{siunitx}\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{microtype}'

def find_nearest(array, value):
    return (np.abs(array - value)).argmin()

def MinMaxScaler(x, min, max):
	return ((x - np.amin(x)) / (np.amax(x) -  np.amin(x))) * (max - min) + min

def split(data_x, data_y, interval_size, step_size, period_length, pp_amplitude, iterate=False):
	x_split = [[x for x in val if x is not None] for val in list(windowed(data_x, interval_size, step=step_size))]
	y_split = [[y for y in val if y is not None] for val in list(windowed(data_y, interval_size, step=step_size))]

	if len(x_split[-1]) < len(x_split[0]) and not iterate:
		print(f'Warning! Chosen interval size of {interval_size} and step size of {step_size} result in last interval being smaller than the rest.')

	result = np.vstack([np.array([np.average(val_x), np.average(val_y)]) for (val_x, val_y) in zip(x_split, y_split)])

	spline = CubicSpline(result[:,0], result[:,1])
	spline_dev = [np.abs(val_data - val_spline) for (val_data, val_spline) in zip(data_y[period_length:period_length*2], spline(data_x[period_length:period_length*2]))]

	return x_split, y_split, result, spline, spline(data_x), np.mean(spline_dev)/pp_amplitude

def find_best(data_x, data_y, interval_start, interval_stop, interval_step, step_start, step_stop, step_step, period_length, pp_amplitude):
	combinations = np.zeros(((step_stop - step_start)//step_step + 1, (interval_stop - interval_start)//interval_step + 1))

	for i, a in tzip(range(step_start, step_stop + 1, step_step), range(step_stop - step_start + 1), desc='Steps'):
		for j, b in tzip(range(interval_start, interval_stop + 1, interval_step), range(interval_stop - interval_start + 1), desc='Intervals', leave=False):
			combinations[a,b] = split(data_x, data_y, j, i, period_length, pp_amplitude, iterate=True)[5]

	return combinations

def plot_best(data_x, data_y, interval_start, interval_stop, interval_step, step_start, step_stop, step_step, period_length, v_min, v_max, v_step, zoom_x, zoom_y):
	comb = find_best(data_x, data_y, interval_start, interval_stop, interval_step, step_start, step_stop, step_step, period_length)

	X = np.arange(interval_start, interval_stop + 1, interval_step)/period_length
	Y = np.arange(step_start, step_stop + 1, step_step)/period_length

	zoom_x_arr = int(np.ceil(period_length*zoom_x/interval_step))
	zoom_y_arr = int(np.ceil(period_length*zoom_y/interval_step))

	extent = np.array([interval_start, interval_stop, step_start, step_stop])/period_length
	levels = np.arange(v_min, v_max + v_step, v_step)

	cbar = plt.colorbar(plt.imshow(comb, origin='lower', aspect='equal', extent=extent, vmin=v_min, vmax=v_max), ax=plt.gca())
	plt.contour(X, Y, comb, levels=levels[1:6], origin='lower', colors='white', linewidths=0.5, linestyles='dashed')

	ins = plt.gca().inset_axes([0.5, 0.5, 0.45, 0.45])
	ins.imshow(comb[0:zoom_y_arr, 0:zoom_x_arr], origin='lower', aspect='equal', extent=[extent[0], zoom_x, extent[2], zoom_y], vmin=v_min, vmax=v_max)
	lab = ins.clabel(ins.contour(X[0:zoom_x_arr], Y[0:zoom_y_arr], comb[0:zoom_y_arr, 0:zoom_x_arr], levels=levels[1:6], origin='lower', colors='white', linewidths=0.5, linestyles='dashed'), colors='white', fontsize='small', inline_spacing=-5)

	for l in lab:
		l.set_rotation(0)

	plt.xlabel(r'Gate Length $\tau$')
	plt.ylabel(r'Sampling Resolution $t_0$')
	cbar.ax.set_xlabel(r'$\overline{d}_{pp}$')

	plt.gca().yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.1f$T$'))
	plt.gca().xaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.1f$T$'))
	plt.gca().minorticks_on()

	ins.xaxis.set_major_locator(plt.MaxNLocator(3))
	ins.yaxis.set_major_locator(plt.MaxNLocator(3))
	ins.minorticks_on()

	ins.yaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$T$'))
	ins.xaxis.set_major_formatter(mticker.FormatStrFormatter(r'%.2f$T$'))

	plt.gca().indicate_inset_zoom(ins, edgecolor='red', linewidth=2.0, alpha=1.0)
	ins.set_aspect(aspect='equal')

	plt.tight_layout()
	plt.show()

def plot_single(data_x, data_y, interval_size, step_size, period_length, pp_amplitude):
	plt.figure(figsize=(10,6))

	x_split, _, result, _, spline_y, spline_dev = split(data_x, data_y, interval_size, step_size, period_length)
	print(f'average deviation of ≈{np.around(spline_dev,4)} relative to the peak-to-peak amplitude')

	plt.plot(data_x[period_length:period_length*2], data_y[period_length:period_length*2], color='red', label='Oscilloscope Signal', zorder=2)
	plt.scatter(result[:,0], result[:,1], s=20, color='black', zorder=4, label='Time-Discrete\nData Point')
	plt.bar(result[:,0], result[:,1], alpha=0.1, width=[np.abs(val[0] - val[-1]) for val in x_split], color='blue', edgecolor='black', zorder=1, label='Averaged Interval')
	plt.plot(data_x, spline_y, color='green', zorder=3, label='Interpolated\nTime-Discrete\nSignal')

	plt.gca().tick_params(axis='both', which='both', direction='in', bottom=True, left=True, top=True, right=True, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
	plt.gca().minorticks_on()

	handles, labels = plt.gca().get_legend_handles_labels()
	handles = [handles[0], handles[3], handles[2], handles[1]]
	labels = [labels[0], labels[3], labels[2], labels[1]]

	plt.xlim((data_x[period_length], data_x[period_length*2]))
	plt.ylim((-2*pp_amplitude/3, 2*pp_amplitude/3))

	plt.xlabel(r'Time $t$ [$\si{\milli\second}$]')
	plt.ylabel(r'Voltage $U$ [$\si{\volt}$]')

	plt.legend(handles, labels, framealpha=1.0, fontsize='small')
	plt.tight_layout()
	plt.show()

def plot_phase(data_x, data_y, phase, slope, slope_var, interval_size, step_size, period_length, pp_amplitude):
	fig, ax1 = plt.subplots(figsize=(10,6))
	ax2 = ax1.twinx()

	phase = np.append(phase, 1)
	slope = np.append(slope, slope[0])
	slope_var = np.append(slope_var, slope_var[0])

	phase_scaled = MinMaxScaler(phase, np.amin(data_x[period_length:period_length*2]), np.amax(data_x[period_length:period_length*2]))
	result = split(data_x, data_y, interval_size, step_size, period_length)[2]

	avg_err = np.zeros((2, result.shape[0]))

	for i in range(avg_err.shape[1]):
		idx = find_nearest(data_x, result[i,0])
		avg_err[:,i] = (0, np.abs(result[i,1] - data_y[idx])) if (result[i,1] < data_y[idx]) else (np.abs(result[i,1] - data_y[idx]), 0)

	ax1.errorbar(result[:,0], result[:,1], avg_err, marker='o', linestyle='', color='black', ms=5, capsize=2.0, elinewidth=1.0, capthick=1.0, zorder=3, label='Time-Discrete\nData Point')
	ax2.errorbar(phase_scaled, slope, slope_var, marker='^', color='blue', mfc='white', mec='blue', linestyle='', capsize=2.0, elinewidth=1.0, capthick=1.0, label='Phase Slope', zorder=2)
	ax1.plot(data_x, data_y, color='red', zorder=1, label='Oscilloscope Signal')

	ax1.tick_params(axis='both', which='both', direction='in', bottom=True, left=True, top=True, right=False, labelbottom=True, labeltop=False, labelleft=True, labelright=False)
	ax2.tick_params(axis='y', which='both', direction='in')

	handles = [*ax1.get_legend_handles_labels()[0], *ax2.get_legend_handles_labels()[0]]
	labels = [*ax1.get_legend_handles_labels()[1], *ax2.get_legend_handles_labels()[1]]

	plt.xlim((data_x[period_length], data_x[period_length*2]))
	ax1.set_ylim((-2*pp_amplitude/3, 2*pp_amplitude/3))
	ax2.set_ylim((-pp_amplitude*14/3, pp_amplitude*14/3))

	ax2.yaxis.set_major_locator(plt.MultipleLocator(0.5))
	ax1.minorticks_on()
	ax2.minorticks_on()

	ax1.set_xlabel(r'Time $t$ [$\si{\milli\second}$]')
	ax1.set_ylabel(r'Voltage $U$ [$\si{\volt}$]')
	ax2.set_ylabel(r'Phase Slope d/d$x$ $\varphi\left(x\right)$ [$\si{\radian\per\micro\metre}$]')

	plt.legend(handles, labels, framealpha=1.0, fontsize='small')
	plt.tight_layout()
	plt.show()