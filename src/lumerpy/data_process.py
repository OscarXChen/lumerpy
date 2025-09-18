# import os
# import sys
# import lumerpy as lupy
from .fdtd_manager import get_fdtd_instance
import numpy as np
import matplotlib.pyplot as plt
import os

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
	# 这里的Energyshow是为了是否计算能量分布，如果Energyshow为False，那么不会有能量分布的计算，也不会正确保存图像结果
	# 坐标轴与电场分量的名称到索引的映射
	axis_map = {'x': 0, 'y': 1, 'z': 2}
	comp_map = {'Ex': 0, 'Ey': 1, 'Ez': 2}

	# 参数检查：axis_name 与 component 必须在上面的映射中
	if axis_name not in axis_map:
		raise ValueError("axis_name 必须是 'x', 'y' 或 'z'")
	if component not in comp_map:
		raise ValueError("component 必须是 'Ex', 'Ey' 或 'Ez'")

	axis_idx = axis_map[axis_name]  # 要做区间筛选的“坐标轴”对应到 E_data 的哪个维度
	comp_idx = comp_map[component]  # 要选取的电场分量（最后一维的索引）

	coord_values = np.array(Edatas[axis_name])
	E_data = Edatas["E"]  # 完整的电场数据

	# 如果需要固定 z/x/y
	fixed_coord_value = None
	if fixed_axis_name and fixed_axis_value is not None:
		if fixed_axis_name not in axis_map:
			raise ValueError("fixed_axis_name 必须是 'x', 'y' 或 'z'")
		fixed_axis_idx = axis_map[fixed_axis_name]
		fixed_coord_array = np.array(Edatas[fixed_axis_name])
		# 找到与 fixed_axis_value 最接近的坐标点索引
		closest_index = np.argmin(np.abs(fixed_coord_array - fixed_axis_value))
		fixed_coord_value = fixed_coord_array[closest_index]

		# 构造切片列表 slicer，长度 = E_data.ndim（每个维度给一个索引器）
		# 先全部置为 slice(None) 表示“取该维的所有元素”
		slicer = [slice(None)] * E_data.ndim
		# 在固定的轴维度上仅保留 [closest_index : closest_index+1] 这一段（长度为1，维度不丢）
		slicer[fixed_axis_idx] = slice(closest_index, closest_index + 1)
		# 应用切片（tuple(...) 是 NumPy 索引约定）
		E_data = E_data[tuple(slicer)]
		# 若固定的轴刚好就是我们要做区间筛选的轴，那么相应 coord_values 也只剩下一个坐标点
		if fixed_axis_name == axis_name:
			coord_values = fixed_coord_array[closest_index:closest_index + 1]

	# 用于收集每个区间的结果（支持多区间）
	E_all, coord_all, energy_all = [], [], []

	# 多区域处理
	# 构造区间列表：
	# - 若提供了 selected_range（形如 [[min1,max1], [min2,max2]]），逐个区间处理；
	# - 否则退化为单一区间 [min_val, max_val]
	region_list = []
	if selected_range is not None:
		region_list = selected_range
	else:
		region_list = [[min_val, max_val]]

	# —— 逐区间进行筛选与取分量 ——
	for r in region_list:
		r_min, r_max = r
		# 1) 先用布尔掩码选出坐标落在 [r_min, r_max] 范围内的位置
		#    mask 的形状与 coord_values 相同（通常是一维），True 表示该索引落在区间内
		mask = (coord_values >= r_min) & (coord_values <= r_max)
		# 2) 把 True 的位置拿出来做索引数组（range_indices 是一维整型数组）
		range_indices = np.where(mask)[0]
		# 3) 取出这些位置对应的坐标值，作为该区间的坐标数组
		coord_selected = coord_values[range_indices]
		# 4) 构造对 E_data 的高维切片：
		#    - 我们要在“筛选轴”（axis_idx）上使用一个“整型索引数组”（range_indices）
		#    - 在“最后一维”（分量维）上使用“单个整型索引”（comp_idx）取出 Ex/Ey/Ez
		#
		# ★ 索引规则要点（NumPy）：
		#   a) 基本索引（basic indexing）：切片 slice(start, stop, step)、单个 int、... —— 这些不会触发“高级索引”规则；
		#   b) 高级索引（advanced indexing）：用“整型数组”或“布尔数组”当索引器会触发高级索引；
		#   c) 当混合使用基本索引与高级索引时：
		#      - 所有“高级索引的轴”会被提到结果的“前面”，其形状是各高级索引器广播后的形状；
		#      - 其余采用基本索引的轴，按原顺序跟在后面；
		#      - 若在某个维度上用的是“单个 int”（属于基本索引），该维会被移除（减少一个维度）。
		#
		#   在本例中：
		#     - 在 axis_idx 维，我们用的是 “整型索引数组 range_indices” → 这是高级索引；
		#     - 在最后一维（-1），我们用的是 “单个整型 comp_idx” → 这是基本索引，且会移除“分量维”；
		#     - 其它维度用 slice(None) → 基本索引，维度保留。
		#
		#   因为出现了高级索引（range_indices），返回结果的形状会把该高级轴（len(range_indices)）放到最前面，
		#   然后拼上其余保留下来的各轴（不含被 int 取走的最后一维）。
		# 选出电场分量
		slicer = [slice(None)] * E_data.ndim
		# 在“筛选轴”上放入“整型索引数组”（高级索引），只取区间内的那几层
		slicer[axis_idx] = range_indices
		# 在“最后一维”（分量维）上放入“单个整型”（基本索引），从而只取一个分量（该维度被移除）
		slicer[-1] = comp_idx

		# 实际取数：
		# E_selected 的形状规则（举例）：若 E_data 原形状是 (Nx, Ny, Nz, 3)
		# - 假设 axis_idx=0（即沿 x 轴筛选，range_indices 长度为 K）
		# - 则 E_selected 的形状通常为 (K, Ny, Nz) —— 注意 K 这个高级索引维会被“提到最前面”
		E_selected = E_data[tuple(slicer)]
		# 为了后续处理方便，去掉长度为 1 的维度（例如前面固定轴但保留了长度为1的维度）
		# 小提示：np.squeeze 只会移除 size=1 的轴，不会改变轴顺序；若想“固定轴也完全消失”，就靠这里的 squeeze。
		E_all.append(np.squeeze(E_selected))
		coord_all.append(coord_selected)

		# 可选的能量计算：对该区间的选中分量做 |E|^2 求和（对所有元素求和，跟轴顺序无关）
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
	# print(f"✅ 所有能量图已保存至 {save_path}_{component}.png")
	# for i, e in enumerate(energy_all):
	# 	print(f"区域 {i} 累计 {component}² 能量为: {e:.4e}")

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
						   save_flag=False, show_area_flag=True, effective_y_span_flag=False,
						   double_output_record_flag=False, effective_y_span=0):
	'''
	返回输出的区域编码和能量；
	此外，save_flag若为True，则将能量图保存到save_path
	'''
	import sys
	import os

	# # 用户在这里设置 API 和文件路径
	# api_path = r"C:/Program Files/Lumerical/v241/api/python"
	# sys.path.append(os.path.normpath(api_path))  # 添加 API 路径以确保可以成功导入 lumapi
	# import lumerpy as lupy
	# lupy.tools.check_path_and_file(file_path=file_path, file_name=file_name, auto_newfile=False)
	# # import lumapi		# lupy库中已经包含了lumapi的导入，不需要额外导入lumapi
	# lupy.setup_paths(api_path, file_path, file_name)  # 设置路径到库
	#
	# # --------------------基本设置结束--------------------
	# fdtd_instance = lupy.get_fdtd_instance(hide=True, solution_type="FDTD")  # 创建fdtd实例，这应该是第一个实例，hide=True时，隐藏窗口
	# # lupy.version()  # 测试一下是否成功
	# FD = lupy.get_existing_fdtd_instance()  # 返回创建的实例，以便使用lumapi
	import lumerpy as lupy
	FD = lupy.initialize(file_path=file_path,file_name=file_name)
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
	if effective_y_span_flag:
		# fdtd_y_span = FD.getnamed("effective_y_span", "y min")  # 通过仿真对象直接传递/px，先这样吧
		fdtd_y_span = effective_y_span
	else:
		fdtd_y_span = FD.getnamed("FDTD", "y span")  # 这里要改一下，不应该通过FDTD的区域范围获取有效宽度，这部分工作挺麻烦的

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
	if show_area_flag:
		for i in range(channals_output):
			area_start, area_end = out_y_metric_total[i, :]
			print(f"区域 {i} 范围：{area_start * 1e6:.2f},\t{area_end * 1e6:.2f}")
		# print(f"可能输出区域为：{out_y_metric_total}")
		print(f"输出区域是：区域 {output_area_code}，并且各区域输出值为：{output_energy_ls}")

	# 多存一次关于之前的输出区域的记录
	if double_output_record_flag:
		extra_gap_y = (period - width) / 2  # 额外抬高半个槽和槽之间的间距
		extra_gap_y = extra_gap_y + width  # 场发射位置本来就在槽和槽中间，这两行代码下来，这个额外抬高的y值就对应着槽和槽中间的硅板的y方向中心
		# extra_gap_y = 0  # 新的设计思路转变为，不在输入和输出处讨论应当抬高多少位置，转变为在设置metaline的时候抬高多少位置
		out_y_metric_center_ls_2 = []
		starts_ls = []
		out_y_metric_start_ls_2 = []
		out_y_metric_total_2 = np.zeros((channals_output, 2))
		out_y_span = out_y_pixel_span * scale_ratio
		for i in range(channals_output):  # 对每个输入/出通道操作
			# out_y_metric_center_ls.append(out_y_pixel_center_ls[i] * scale_ratio + extra_gap_y)		# 这里应该有点问题，涉及到extra_gap_y，先不管他
			out_y_metric_start_ls_2.append(out_y_pixel_start_ls[i] * scale_ratio + extra_gap_y)
			out_y_metric_total_2[i, :] = out_y_metric_start_ls_2[i], out_y_metric_start_ls_2[i] + out_y_span
		# print(f"输出位置[{i}]：{out_y_metric_start_ls[i]},{out_y_metric_start_ls[i] + out_y_span}")
		# print(out_y_metric_total)
		# 选择好输出范围即可
		# selected_ranges = np.array([
		# 	[0e-6, 6e-6],
		# 	[12e-6, 18e-6]
		# ])
		# save_path=os.path.join(save_path,"record-2")
		if save_flag:
			output_area_code_2, energy_list_2 = lupy.get_simple_out(selected_range=out_y_metric_total_2,
																	power_name=power_name,
																	z_fixed=z_fixed, plot_Ey_flag=plot_Ey_flag,
																	plot_energy_flag=plot_energy_flag,
																	save_path=save_path)
		else:
			output_area_code_2, energy_list_2 = lupy.get_simple_out(selected_range=out_y_metric_total_2,
																	power_name=power_name,
																	z_fixed=z_fixed, plot_Ey_flag=plot_Ey_flag,
																	plot_energy_flag=plot_energy_flag,
																	save_path=False)  # 我知道这里逻辑很古怪，先这样吧
		output_energy_ls_2 = [round(float(x), 4) for x in energy_list_2]
		return output_area_code, output_energy_ls, output_area_code_2, output_energy_ls_2
	else:
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
		file_csv_path = os.path.join(save_path, save_name.removesuffix(".fsp")) + ".csv"
	else:
		file_csv_path = os.path.join(save_path, save_name.removesuffix(".fsp")) + "-" + str(save_index) + ".csv"
	save_temp = [int_to_record] + list(list_to_append)
	os.makedirs(os.path.dirname(file_csv_path), exist_ok=True)
	with open(file_csv_path, "a+") as fp:
		np.savetxt(fp, [save_temp], delimiter=",")
	# print(f"csv文件已保存至：{file_csv_path}")
	return file_csv_path


def get_channels_in_out(path_data, path_pd, show_flag=False, return_data_decode_flag=False, each_pix=3):
	data_count, data_raw = read_unique_csv(path_data)

	data_y = data_raw[:, 0]
	data_X = data_raw[:, 1:]

	data_X_decode = np.apply_along_axis(recover_original, axis=1, arr=data_X, repeat=each_pix)
	# print(f"展示前16条经过译码的输入数据为：\n{data_X_decode[0:16]}")
	pd_count, pd_raw = read_unique_csv(path_pd)

	pd_overview = pd_raw[0]
	pd_pds = pd_raw[1:]
	pd_decode = np.apply_along_axis(recover_original, axis=1, arr=pd_pds, repeat=each_pix)

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


def get_data_single_scale(channels_in, each_pix=3, data_single_scale_row=1, duty_cycle=0.5):
	data_single_scale_col = channels_in / duty_cycle * each_pix  # 默认占空比为50%，所以搞出2倍
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
