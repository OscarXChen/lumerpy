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
		selected_range=None, plot_energy=False,save_path=None
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
		vmax=vmax*1.1
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
				ax.set_title(f"区域 {i} 的{component}")
				ax.set_xlabel(f"{axis_name} (μm)")
				ax.set_ylabel(component)
				ax.grid(True)
			elif e.ndim == 2:
				extent = [coord_um[0], coord_um[-1], 0, e.shape[1]]
				im = ax.imshow(e.T, aspect='auto', origin='lower', extent=extent, vmin=vmin, vmax=vmax)
				ax.set_title(f"区域 {i} 的 {component}")
				ax.set_xlabel(f"{axis_name} (μm)")
				ax.set_ylabel("Other axis index")
				plt.colorbar(im, ax=ax, label=component)
		plt.tight_layout()
		plt.show()

	# -------------------------
	# 🎨 能量图 + 输出 + 能量标注
	# -------------------------
	if Energyshow:

		# ✅ 获取所有 Ey² 的全局最小/最大值
		all_Ey2 = [np.abs(e) ** 2 for e in E_all]
		ymin = min(np.min(e) for e in all_Ey2)
		ymax = max(np.max(e) for e in all_Ey2)
		ymax=ymax*1.1

		fig, axs = plt.subplots(1, len(E_all), figsize=(6 * len(E_all), 4))
		if len(E_all) == 1:
			axs = [axs]

		for i, Ey2 in enumerate(all_Ey2):
			coord_um = coord_all[i] * 1e6
			energy = energy_all[i]
			ax = axs[i]

			if Ey2.ndim == 1:
				ax.plot(coord_um, Ey2)
				ax.set_ylim(ymin, ymax)  # ✅ 统一 y 轴范围
				ax.set_title(f"区域 {i} 的 |{component}|²")
				ax.set_xlabel(f"{axis_name} (μm)")
				ax.set_ylabel(f"|{component}|²")
				ax.grid(True)
				ax.text(0.98, 0.95, f"累计能量 = {energy:.2e}",
						transform=ax.transAxes,
						fontsize=10, color='red',
						horizontalalignment='right',
						verticalalignment='top')

			elif Ey2.ndim == 2:
				extent = [coord_um[0], coord_um[-1], 0, Ey2.shape[1]]
				im = ax.imshow(Ey2.T, aspect='auto', origin='lower', extent=extent,
							   vmin=ymin, vmax=ymax)  # ✅ 统一色标范围
				ax.set_title(f"区域 {i} 的 |{component}|²")
				ax.set_xlabel(f"{axis_name} (μm)")
				ax.set_ylabel("Other axis index")
				plt.colorbar(im, ax=ax, label=f"|{component}|²")
				ax.text(0.98, 0.95, f"累计能量 = {energy:.2e}",
						transform=ax.transAxes,
						fontsize=10, color='red',
						horizontalalignment='right',
						verticalalignment='top')

		plt.tight_layout()
		if plot_energy:
			plt.show()
			if save_path:
				import os
				os.makedirs(save_path, exist_ok=True)
				import time
				current_time = time.strftime("%m%d-%H%M")
				fig.savefig(f"{save_path}{current_time}_{component}.png", dpi=300)
				print(f"✅ 所有能量图已保存至 {save_path}_{component}.png")
		for i, e in enumerate(energy_all):
			print(f"区域 {i} 累计 {component}² 能量为: {e:.4e}")

	return E_all, coord_all, fixed_coord_value, energy_all if Energyshow else None

def get_simple_out(selected_range, power_name="local_outputs", z_fixed=0.11e-6,
				   plot=False, Energyshow=True, plot_energy=False,
				   axis_name='y', component='Ey', fixed_axis_name='z',save_path=False):
	FD = get_fdtd_instance()
	Edatas = FD.getresult(power_name, "E")

	E_list, coord_list, z_used, energy_list = select_E_component_by_range_from_dataset(
		Edatas, axis_name=axis_name, component=component, fixed_axis_name=fixed_axis_name,
		fixed_axis_value=z_fixed, selected_range=selected_range,
		plot=plot, Energyshow=Energyshow, plot_energy=plot_energy,save_path=save_path)

	# print(energy_list)
	idx = int(np.argmax(energy_list))

	return idx, energy_list
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

def get_results(size=(1, 50), channals_output=2, duty_cycle=0.5, margins_cycle=(0, 0, 0, 0), power_name="local_outputs",
				period=0.5e-6, width=0.2e-6, z_fixed=0.11e-6,
				file_path=r"E:\0_Work_Documents\Simulation\lumerpy\03_cat",
				file_name=r"m00_temp.fsp",save_path=False):
	import sys
	import os

	# 用户在这里设置 API 和文件路径
	api_path = r"C:/Program Files/Lumerical/v241/api/python"
	# file_path = r"E:\0_Work_Documents\Simulation\lumerpy\03_cat"
	# file_name = r"m01_wg.fsp"
	# file_name = r"m00_temp.fsp"
	sys.path.append(os.path.normpath(api_path))  # 添加 API 路径以确保可以成功导入 lumapi
	import lumerpy as lupy

	lupy.tools.check_path_and_file(file_path=file_path, file_name=file_name)
	# import lumapi		# lupy库中已经包含了lumapi的导入，不需要额外导入lumapi
	lupy.setup_paths(api_path, file_path, file_name)  # 设置路径到库

	# --------------------基本设置结束--------------------
	fdtd_instance = lupy.get_fdtd_instance(hide=True, solution_type="FDTD")  # 创建fdtd实例，这应该是第一个实例，hide=True时，隐藏窗口
	# lupy.version()  # 测试一下是否成功
	FD = lupy.get_existing_fdtd_instance()  # 返回创建的实例，以便使用lumapi
	if not FD:
		print("未正确创建实例，请检查")
	u = 1e-6

	# --------------------现在既可以调用lumapi，也可以调用lupy库--------------------
	import numpy as np

	lupy.plot_initialize()
	Edatas = FD.getresult(power_name, "E")
	out_y_ls_temp, out_y_start_ls_temp, out_y_pixel_scale = lupy.tools.get_single_inputs_center_x(
		channels=channals_output,
		data_single_scale=size,
		duty_cycle=duty_cycle,
		margins_cycle=margins_cycle)

	fdtd_y_span = FD.getnamed("FDTD", "y span")
	scale_ratio = (fdtd_y_span / size[1])
	extra_gap_y = (period - width) / 2  # 额外抬高半个槽和槽之间的间距
	extra_gap_y = extra_gap_y + width  # 场发射位置本来就在槽和槽中间
	out_y_ls = []
	starts_ls = []
	out_y_start_ls = []
	out_y_range = np.zeros((len(out_y_ls_temp), 2))
	out_y_span = out_y_pixel_scale * scale_ratio
	for i in range(len(out_y_ls_temp)):
		out_y_ls.append(out_y_ls_temp[i] * scale_ratio + extra_gap_y)
		out_y_start_ls.append(out_y_start_ls_temp[i] * scale_ratio + extra_gap_y)
		out_y_range[i, :] = out_y_start_ls[i], out_y_start_ls[i] + out_y_span
	# print(f"输出位置[{i}]：{out_y_start_ls[i]},{out_y_start_ls[i] + out_y_span}")
	# print(out_y_range)
	# 选择好输出范围即可
	# selected_ranges = np.array([
	# 	[0e-6, 6e-6],
	# 	[12e-6, 18e-6]
	# ])

	idx, energy_list = lupy.get_simple_out(selected_range=out_y_range, power_name=power_name, z_fixed=z_fixed,
										   plot=True, plot_energy=True,save_path=save_path)
	output_energy_ls = [round(float(x), 4) for x in energy_list]
	# print(f"输出区域是：{idx}，并且各输出值为：{output_energy_ls}")


	for i in range(out_y_range.shape[0]):
		area_start,area_end=out_y_range[i,:]
		print(f"区域 {i} 范围：{area_start*1e6:.2f},\t{area_end*1e6:.2f}")
	# print(f"可能输出区域为：{out_y_range}")
	print(f"输出区域是：区域 {idx}，并且各区域输出值为：{output_energy_ls}")
	return idx,output_energy_ls


import numpy as np


def read_unique_csv(path, delimiter=",", dtype=float, has_header=True):
	"""
	用 np.loadtxt 读取 CSV 文件并返回唯一记录数和唯一记录

	参数:
		path: str, CSV 文件路径
		delimiter: str, 分隔符，默认逗号 ","
		dtype: 数据类型，默认 float

	返回:
		unique_count: int, 不重复记录数
		unique_records: ndarray, shape=(n_unique, n_cols)
	"""
	# 读取整个 CSV 文件
	if has_header:
		data = np.loadtxt(path, delimiter=delimiter, dtype=dtype, skiprows=1)
	else:
		data = np.loadtxt(path, delimiter=delimiter, dtype=dtype)

	# 找到唯一行
	unique_records, idx = np.unique(data, axis=0, return_index=True)
	unique_records = unique_records[np.argsort(idx)]	# 保持原本的顺序
	unique_count = unique_records.shape[0]

	return unique_count, unique_records
