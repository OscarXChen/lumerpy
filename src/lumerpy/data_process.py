import os
import sys
import lumerpy as lupy
from .fdtd_manager import get_fdtd_instance
import numpy as np
import matplotlib.pyplot as plt

u = 1e-6


def plot_initialize(paper_font=False):
	"""避免GUI交互问题和中文不显示的问题"""
	import matplotlib
	matplotlib.use('TkAgg')  # 避免 GUI 交互问题
	# 设置支持中文的字体，并根据是否论文需要修改中文为宋体，英文为times new roman
	if paper_font is False:
		plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体
	else:
		plt.rcParams['font.family'] = ['SimSun', 'Times New Roman']
	plt.rcParams['axes.unicode_minus'] = False  # 解决负号 "-" 显示为方块的问题


def select_E_component_by_range_from_dataset(
		Edatas,
		axis_name,
		component='Ey',
		min_val=None,
		max_val=None,
		fixed_axis_name=None,
		fixed_axis_value=None,
		plot=False,
		Energyshow=True,
		selected_range=None,plot_energy=False
):
	axis_map = {'x': 0, 'y': 1, 'z': 2}
	comp_map = {'Ex': 0, 'Ey': 1, 'Ez': 2}

	if axis_name not in axis_map:
		raise ValueError("axis_name 必须是 'x', 'y' 或 'z'")
	if component not in comp_map:
		raise ValueError("component 必须是 'Ex', 'Ey' 或 'Ez'")

	axis_idx = axis_map[axis_name]
	comp_idx = comp_map[component]

	coord_values = np.array(Edatas[axis_name])
	E_data = Edatas["E"]

	# 如果需要固定 z/x/y
	fixed_coord_value = None
	if fixed_axis_name and fixed_axis_value is not None:
		if fixed_axis_name not in axis_map:
			raise ValueError("fixed_axis_name 必须是 'x', 'y' 或 'z'")
		fixed_axis_idx = axis_map[fixed_axis_name]
		fixed_coord_array = np.array(Edatas[fixed_axis_name])
		closest_index = np.argmin(np.abs(fixed_coord_array - fixed_axis_value))
		fixed_coord_value = fixed_coord_array[closest_index]
		slicer = [slice(None)] * E_data.ndim
		slicer[fixed_axis_idx] = slice(closest_index, closest_index + 1)
		E_data = E_data[tuple(slicer)]
		if fixed_axis_name == axis_name:
			coord_values = fixed_coord_array[closest_index:closest_index + 1]

	# 准备多个区域的结果
	E_all, coord_all, energy_all = [], [], []

	# 多区域处理
	region_list = []
	if selected_range is not None:
		region_list = selected_range
	else:
		region_list = [[min_val, max_val]]

	for r in region_list:
		r_min, r_max = r
		mask = (coord_values >= r_min) & (coord_values <= r_max)
		range_indices = np.where(mask)[0]
		coord_selected = coord_values[range_indices]

		# 选出电场分量
		slicer = [slice(None)] * E_data.ndim
		slicer[axis_idx] = range_indices
		slicer[-1] = comp_idx
		E_selected = E_data[tuple(slicer)]
		E_all.append(np.squeeze(E_selected))
		coord_all.append(coord_selected)

		if Energyshow:
			energy = np.sum(np.abs(E_selected) ** 2)
			energy_all.append(energy)

	# -------------------------
	# 🎨 统一纵坐标画图：电场分布
	# -------------------------
	if plot:
		n = len(region_list)
		vmin = min([np.min(e) for e in E_all])
		vmax = max([np.max(e) for e in E_all])
		fig, axs = plt.subplots(1, n, figsize=(6 * n, 4))
		if n == 1:
			axs = [axs]
		for i in range(n):
			coord_um = coord_all[i] * 1e6
			ax = axs[i]
			e = E_all[i]
			if e.ndim == 1:
				ax.plot(coord_um, e)
				ax.set_ylim(vmin, vmax)
				ax.set_title(f"{component} in region {i + 1}")
				ax.set_xlabel(f"{axis_name} (μm)")
				ax.set_ylabel(component)
				ax.grid(True)
			elif e.ndim == 2:
				extent = [coord_um[0], coord_um[-1], 0, e.shape[1]]
				im = ax.imshow(e.T, aspect='auto', origin='lower', extent=extent, vmin=vmin, vmax=vmax)
				ax.set_title(f"{component} in region {i + 1}")
				ax.set_xlabel(f"{axis_name} (μm)")
				ax.set_ylabel("Other axis index")
				plt.colorbar(im, ax=ax, label=component)
		plt.tight_layout()
		plt.show()

	# -------------------------
	# 🎨 能量图 + 输出 + 能量标注
	# -------------------------
	if Energyshow:
		fig, axs = plt.subplots(1, len(E_all), figsize=(6 * len(E_all), 4))
		if len(E_all) == 1:
			axs = [axs]
		for i, e in enumerate(E_all):
			Ey2 = np.abs(e) ** 2
			coord_um = coord_all[i] * 1e6
			energy = energy_all[i]
			ax = axs[i]

			if Ey2.ndim == 1:
				ax.plot(coord_um, Ey2)
				ax.set_title(f"|{component}|² in region {i + 1}")
				ax.set_xlabel(f"{axis_name} (μm)")
				ax.set_ylabel(f"|{component}|²")
				ax.grid(True)
				# ✅ 添加能量标注
				ax.text(0.98, 0.95, f"Energy = {energy:.2e}",
						transform=ax.transAxes,
						fontsize=10, color='red',
						horizontalalignment='right',
						verticalalignment='top')

			elif Ey2.ndim == 2:
				extent = [coord_um[0], coord_um[-1], 0, Ey2.shape[1]]
				im = ax.imshow(Ey2.T, aspect='auto', origin='lower', extent=extent)
				ax.set_title(f"|{component}|² in region {i + 1}")
				ax.set_xlabel(f"{axis_name} (μm)")
				ax.set_ylabel("Other axis index")
				plt.colorbar(im, ax=ax, label=f"|{component}|²")
				# ✅ 添加能量标注
				ax.text(0.98, 0.95, f"Energy = {energy:.2e}",
						transform=ax.transAxes,
						fontsize=10, color='red',
						horizontalalignment='right',
						verticalalignment='top')

		plt.tight_layout()
		if plot_energy:
			plt.show()

		for i, e in enumerate(energy_all):
			print(f"区域 {i + 1} 累积 {component}² 能量为: {e:.4e}")

	return E_all, coord_all, fixed_coord_value, energy_all if Energyshow else None


def get_simple_out(selected_range, power_name="local_outputs", z_fixed=0.11e-6):
	FD = get_fdtd_instance()
	Edatas = FD.getresult(power_name, "E")

	E_list, coord_list, z_used, energy_list = select_E_component_by_range_from_dataset(
		Edatas, axis_name='y', component='Ey', fixed_axis_name='z',
		fixed_axis_value=z_fixed, selected_range=selected_range, plot=False, Energyshow=True)

	# print(energy_list)
	idx = int(np.argmax(energy_list))

	return idx,energy_list
# def cal_result(power_name):
# 	FD = get_fdtd_instance()
# 	Edatas = FD.getresult(power_name, "E")
#
# 	select_E_component_by_range(E_data=Edatas,coord_values=)
#
#
# 	Ez_index = int(len(Edatas["E"][0, 0, :, 0, 0]) / 2)  # 选取中间的那个值
# 	Eys = Edatas["E"][0, :, Ez_index, 0, 1]
# 	# Edatas["E"].shape = (1, 338, 10, 1, 3) # 应该分别是：x,y,z,f,(Ex,Ey,Ez)
# 	# 我有一个高维度数据组Edatas["E"]，其中Edatas["E"].shape=(1, 338, 10, 1, 3)，分别对应
# 	# x，y，z，f，(Ex,Ey,Ez)
# 	# 我现在希望：
# 	# 选取所有x在我指定的范围（例如：index=[3,5]）中的Ey数据，如何做？
