# 本页用于直接复制到命令行终端窗口运行
import sys
import os

# 用户在这里设置 API 和文件路径
api_path = r"C:/Program Files/Lumerical/v241/api/python"
# file_path = r"E:\0_Work_Documents\Simulation\lumerpy\03_cat"
file_path = r"E:\0_Work_Documents\Simulation\lumerpy\01_equal-plane"
# file_name = r"m01_wg.fsp"
file_name = r"m1.10_gaussian.fsp"
# file_name = r"m00_temp.fsp"


file_path=r"E:\0_Work_Documents\Simulation\lumerpy\01_equal-plane\records"
file_name=r"m1.4_gaussian_20250114_16-51-26.fsp"
sys.path.append(os.path.normpath(api_path))  # 添加 API 路径以确保可以成功导入 lumapi
import lumerpy as lupy

lupy.tools.check_path_and_file(file_path=file_path, file_name=file_name)
# import lumapi		# lupy库中已经包含了lumapi的导入，不需要额外导入lumapi
lupy.setup_paths(api_path, file_path, file_name)  # 设置路径到库

# --------------------基本设置结束--------------------
fdtd_instance = lupy.get_fdtd_instance(hide=False, solution_type="FDTD")  # 创建fdtd实例，这应该是第一个实例，hide=True时，隐藏窗口
# lupy.version()  # 测试一下是否成功
FD = lupy.get_existing_fdtd_instance()  # 返回创建的实例，以便使用lumapi
if not FD:
	print("未正确创建实例，请检查")
u = 1e-6

# --------------------现在既可以调用lumapi，也可以调用lupy库--------------------
import numpy as np

input("输入回车继续")
lupy.plot_initialize()
lupy.cal_eff_delta("eri00")
lupy.cal_eff_delta("eri01")
lupy.cal_eff_delta("eri02")
lupy.cal_eff_delta("eri03")

# input("输入回车结束")

# Edatas = FD.getresult("local_outputs", "E")
# z_fixed = 0.11e-6

# out_y_span = out_y_pixel_scale * scale_ratio
# for i in range(len(out_y_ls_temp)):
# 	out_y_ls.append(out_y_ls_temp[i] * scale_ratio + extra_gap_y)
# 	out_y_start_ls.append(out_y_start_ls_temp[i] * scale_ratio + extra_gap_y)
# 	print(f"输出位置[{i}]：{out_y_start_ls[i]},{out_y_start_ls[i] + out_y_span}")

# # 选择好输出范围即可
# selected_ranges = np.array([
# 	[0e-6, 6e-6],
# 	[12e-6, 18e-6]
# ])
#
# idx, energy_list = lupy.get_simple_out(selected_range=selected_ranges, power_name="local_outputs", z_fixed=z_fixed)
# output_energy_ls = [round(float(x), 4) for x in energy_list]
# print(f"输出区域是：{idx}，并且各输出值为：{output_energy_ls}")

# E_list, coord_list, z_used, energy_list = lupy.select_E_component_by_range_from_dataset(
# 	Edatas, axis_name='y', component='Ey', fixed_axis_name='z',
# 	fixed_axis_value=z_fixed, selected_range=ranges, plot=False, Energyshow=True)
# print(energy_list)
# idx = int(np.argmax(energy_list))
# print("能量最大的区域是 Region", idx + 1)

# input("输入回车结束")

# power_name="local_outputs"
# Edatas = FD.getresult(power_name, "E")
# E_list, coord_list, z_used, energy_list = lupy.select_E_component_by_range_from_dataset(
# 	Edatas, axis_name="y", component="Ey",min_val=FD.getnamed("FDTD","y min"),max_val=FD.getnamed("FDTD","y max"),
# 	plot=True, Energyshow=True, plot_energy=True)