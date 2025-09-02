# import os
# import sys
# import lumerpy as lupy
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
		Edatas, axis_name, component='Ey', min_val=None, max_val=None, fixed_axis_name=None, fixed_axis_value=None,
		plot_Ey_flag=False, Energyshow=True, selected_range=None, plot_energy_flag=False, save_path=None
):
	# 这里的Energyshow好像只是为了返回能量分布列表
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
	if plot_Ey_flag:
		n = len(region_list)
		vmin = min([np.min(e) for e in E_all])
		vmax = max([np.max(e) for e in E_all])
		vmax = vmax * 1.1
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
		ymax = ymax * 1.1

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
		if plot_energy_flag:
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
				   plot_Ey_flag=False, Energyshow=True, plot_energy_flag=False,
				   axis_name='y', component='Ey', fixed_axis_name='z', save_path=False):
	FD = get_fdtd_instance()
	Edatas = FD.getresult(power_name, "E")

	E_list, coord_list, z_used, energy_list = select_E_component_by_range_from_dataset(
		Edatas, axis_name=axis_name, component=component, fixed_axis_name=fixed_axis_name,
		fixed_axis_value=z_fixed, selected_range=selected_range,
		plot_Ey_flag=plot_Ey_flag, Energyshow=Energyshow, plot_energy_flag=plot_energy_flag, save_path=save_path)

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

def get_simulation_results(size=(1, 50), channals_output=2, duty_cycle=0.5, margins_cycle=(0, 0, 0, 0),
						   power_name="local_outputs",
						   period=0.5e-6, width=0.2e-6, z_fixed=0.11e-6,
						   file_path=r"E:\0_Work_Documents\Simulation\lumerpy\03_cat",
						   file_name=r"m00_temp.fsp", save_path=False, plot_Ey_flag=True, plot_energy_flag=True,
						   save_flag=False):
	'''
	返回输出的区域编码和能量；
	此外，save_flag若为True，则将能量图保存到save_path
	'''
	import sys
	import os

	# 用户在这里设置 API 和文件路径
	api_path = r"C:/Program Files/Lumerical/v241/api/python"
	sys.path.append(os.path.normpath(api_path))  # 添加 API 路径以确保可以成功导入 lumapi
	import lumerpy as lupy
	lupy.tools.check_path_and_file(file_path=file_path, file_name=file_name, auto_newfile=False)
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
	# Edatas = FD.getresult(power_name, "E")
	out_y_pixel_center_ls, out_y_pixel_start_ls, out_y_pixel_span, _ = lupy.tools.get_single_inputs_center_x(
		channels=channals_output,
		data_single_scale=size,
		duty_cycle=duty_cycle,
		margins_cycle=margins_cycle)

	# fdtd_y_span = FD.getnamed("FDTD", "y span")  # 这里要改一下，不应该通过FDTD的区域范围获取有效宽度，这部分工作挺麻烦的
	fdtd_y_span = FD.getnamed("effective_y_span", "y min")  # 通过仿真对象直接传递/px，先这样吧
	scale_ratio = (fdtd_y_span / size[1])
	# extra_gap_y = (period - width) / 2  # 额外抬高半个槽和槽之间的间距
	# extra_gap_y = extra_gap_y + width  # 场发射位置本来就在槽和槽中间，这两行代码下来，这个额外抬高的y值就对应着槽和槽中间的硅板的y方向中心
	extra_gap_y = 0  # 新的设计思路转变为，不在输入和输出处讨论应当抬高多少位置，转变为在设置metaline的时候抬高多少位置
	out_y_metric_center_ls = []
	starts_ls = []
	out_y_metric_start_ls = []
	out_y_metric_total = np.zeros((channals_output, 2))
	out_y_span = out_y_pixel_span * scale_ratio
	for i in range(channals_output):  # 对每个输入/出通道操作
		# out_y_metric_center_ls.append(out_y_pixel_center_ls[i] * scale_ratio + extra_gap_y)		# 这里应该有点问题，涉及到extra_gap_y，先不管他
		out_y_metric_start_ls.append(out_y_pixel_start_ls[i] * scale_ratio + extra_gap_y)
		out_y_metric_total[i, :] = out_y_metric_start_ls[i], out_y_metric_start_ls[i] + out_y_span
	# print(f"输出位置[{i}]：{out_y_metric_start_ls[i]},{out_y_metric_start_ls[i] + out_y_span}")
	# print(out_y_metric_total)
	# 选择好输出范围即可
	# selected_ranges = np.array([
	# 	[0e-6, 6e-6],
	# 	[12e-6, 18e-6]
	# ])

	if save_flag:
		output_area_code, energy_list = lupy.get_simple_out(selected_range=out_y_metric_total, power_name=power_name,
															z_fixed=z_fixed, plot_Ey_flag=plot_Ey_flag,
															plot_energy_flag=plot_energy_flag, save_path=save_path)
	else:
		output_area_code, energy_list = lupy.get_simple_out(selected_range=out_y_metric_total, power_name=power_name,
															z_fixed=z_fixed, plot_Ey_flag=plot_Ey_flag,
															plot_energy_flag=plot_energy_flag,
															save_path=False)  # 我知道这里逻辑很古怪，先这样吧
	output_energy_ls = [round(float(x), 4) for x in energy_list]
	# print(f"输出区域是：{output_area_code}，并且各输出值为：{output_energy_ls}")

	for i in range(channals_output):
		area_start, area_end = out_y_metric_total[i, :]
		print(f"区域 {i} 范围：{area_start * 1e6:.2f},\t{area_end * 1e6:.2f}")
	# print(f"可能输出区域为：{out_y_metric_total}")
	print(f"输出区域是：区域 {output_area_code}，并且各区域输出值为：{output_energy_ls}")
	return output_area_code, output_energy_ls


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
	# txt = "\n\t本函数已弃用，请调用difrannpy库里datas.py的同名函数。\n\t如果必然需要本函数，请手动进入源代码，删去注释使用"
	# raise NotImplementedError(txt)
	# 读取整个 CSV 文件
	if has_header:
		data = np.loadtxt(path, delimiter=delimiter, dtype=dtype, skiprows=1)
	else:
		data = np.loadtxt(path, delimiter=delimiter, dtype=dtype)

	# 找到唯一行
	unique_records, idx = np.unique(data, axis=0, return_index=True)
	unique_records = unique_records[np.argsort(idx)]  # 保持原本的顺序
	unique_count = unique_records.shape[0]
	return unique_count, unique_records

def save_csv_results(save_path, save_name, int_to_record, list_to_append="", save_index=-1):
	'''以每行记录形如：【0,0.1,0.2】的形式保存仿真结果为csv格式'''
	if save_index == -1:
		file_csv_path = save_path + save_name.removesuffix(".fsp") + ".csv"
	else:
		file_csv_path = save_path + save_name.removesuffix(".fsp") + "-" + str(save_index) + ".csv"
	save_temp = int_to_record + list(list_to_append)
	with open(file_csv_path, "a+") as fp:
		np.savetxt(fp, [save_temp], delimiter=",")
	print(f"csv文件已保存至：{file_csv_path}")


def get_channels_in_out(path_data, path_pd, show_flag=False, return_data_decode_flag=False):
	data_count, data_raw = read_unique_csv(path_data)

	data_y = data_raw[:, 0]
	data_X = data_raw[:, 1:]

	data_X_decode = np.apply_along_axis(recover_original, axis=1, arr=data_X)
	# print(f"展示前16条经过译码的输入数据为：\n{data_X_decode[0:16]}")
	pd_count, pd_raw = read_unique_csv(path_pd)

	pd_overview = pd_raw[0]
	pd_pds = pd_raw[1:]
	pd_decode = np.apply_along_axis(recover_original, axis=1, arr=pd_pds)

	channels_in = len(data_X_decode[0])
	channels_out = len(pd_decode)
	if show_flag:
		print(f"不重复训练数据共有：{data_count}条")
		print(f"展示第0条输入数据为：\n{data_X[0]},展示前16条输出数据为：\n{data_y[0:16]}")
		print(f"不重复pd数据共有：{pd_count}条")
		print(f"展示前8条经过译码的输出pd为：\n{pd_decode[0:8]}")
	if not return_data_decode_flag:
		return channels_in, channels_out
	else:
		return channels_in, channels_out, data_X_decode

def recover_original(arr, repeat=3):
	"""
	从扩展数组恢复原始数组

	参数:
		arr: numpy 一维数组 (扩展结果)
		repeat: 每个元素重复次数 (默认 3)

	返回:
		原始数组 (numpy 一维数组)
	"""
	arr = np.asarray(arr)

	# 第一步：解开重复
	if arr.size % repeat != 0:
		raise ValueError("数组长度不能被 repeat 整除")
	reduced = arr.reshape(-1, repeat)[:, 0]  # 取每组的第一个

	# 第二步：去掉中间插的 0（取偶数位置）
	original = reduced[::2]

	return original.astype(int)

def get_data_single_scale(channels_in, each_pix=3, data_single_scale_row=1):
	data_single_scale_col = channels_in * 2 * each_pix  # 默认占空比为50%，所以搞出2倍
	# 这里还有一个事必须提一下，如果bit_expand_flag=True，那么由于扩展组合编码的关系，实际的col数会是2倍
	data_single_scale = (data_single_scale_row, data_single_scale_col)
	# 下面这个位扩展标志位相关代码已弃用，改成在调用函数的外面直接翻倍输入通道
	# if bit_expand_flag:  # 如果采用扩展组合编码
	# 	# 这里插一句，这里有点屎山的感觉了，因为data_single_scale这个元组需要给generate_data_total()函数
	# 	# 但是如果使用扩展组合编码的话，实际上的data_single_scale会变为两倍，所以搞出了一个data_single_scale_temp变量去存这个结果
	# 	# 但是实际上后面的程序，哪哪都要这个data_singel_scale_temp，包括后面提到的size也是
	# 	# 也就是说，变量size才是真正的“数据尺寸”
	# 	data_single_scale_temp = (data_single_scale[0], data_single_scale[1] * 2)
	# else:
	# 	data_single_scale_temp = data_single_scale
	return data_single_scale