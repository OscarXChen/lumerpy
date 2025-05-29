from .fdtd_manager import get_fdtd_instance
from collections import OrderedDict  # 这种写法要手动引入，也很烦


def add_simulation_fdtd(x=0, y=0, z=0, x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0,
						background_material="SiO2 (Glass) - Palik",
						x_min_bc="metal", x_max_bc="metal",
						y_min_bc="metal", y_max_bc="metal",
						z_min_bc="metal", z_max_bc="metal",
						mesh_accuracy=2
						):
	FD = get_fdtd_instance()
	props = OrderedDict([
		("x min", x_min),
		("x max", x_max),
		("y min", y_min),
		("y max", y_max),
		("z min", z_min),
		("z max", z_max),
		("x min bc", x_min_bc),
		("x max bc", x_max_bc),
		("y min bc", y_min_bc),
		("y max bc", y_max_bc),
		("z min bc", z_min_bc),
		("z max bc", z_max_bc),
	])
	ob_fdtd = FD.addfdtd(properties=props)
	FD.set("mesh accuracy", mesh_accuracy)
	FD.set("background material", background_material)
	return ob_fdtd


def add_simulation_fde(x=0, y=0, z=0, x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0,
					   solver_type="2D Z normal", background_material="SiO2 (Glass) - Palik",
					   x_min_bc="metal", x_max_bc="metal",
					   y_min_bc="metal", y_max_bc="metal",
					   z_min_bc="metal", z_max_bc="metal", ):
	FD = get_fdtd_instance()

	if solver_type == "2D X normal":
		if x_min != x_max:
			print("对待放置的2D X normal仿真，输入的x_min和x_max不相等，这将是其x坐标，请检查！")
		props = OrderedDict([
			("solver type", solver_type),
			("y min", y_min),
			("y max", y_max),
			("z min", z_min),
			("z max", z_max),
			("x", x_min),
			("background material", background_material)])
		ob_mode = FD.addfde(properties=props)
		FD.set("y min bc", y_min_bc)  # 吐槽：奇葩逻辑，设置了y min bc为periodic以后，再设置y max bc就会出错，明明bloch条件都不会
		if y_min_bc != "periodic":
			FD.set("y max bc", y_max_bc)
		FD.set("z min bc", z_min_bc)
		if z_min_bc != "periodic":
			FD.set("z max bc", z_max_bc)
	elif solver_type == "2D Y normal":
		if y_min != y_max:
			print("对待放置的2D Y normal仿真，输入的y_min和y_max不相等，这将是其y坐标，请检查！")
		props = OrderedDict([
			("solver type", solver_type),
			("x min", x_min),
			("x max", x_max),
			("z min", z_min),
			("z max", z_max),
			("y", y_min),
			("background material", background_material),
			("x min bc", x_min_bc),
			("x max bc", x_max_bc),
			("z min bc", z_min_bc),
			("z max bc", z_max_bc)
		])
		ob_mode = FD.addfde(properties=props)
		FD.set("x min bc", x_min_bc)  # 吐槽：奇葩逻辑，设置了y min bc为periodic以后，再设置y max bc就会出错，明明bloch条件都不会
		if x_min_bc != "periodic":
			FD.set("x max bc", x_max_bc)
		FD.set("z min bc", z_min_bc)
		if z_min_bc != "periodic":
			FD.set("z max bc", z_max_bc)
	elif solver_type == "2D Z normal":
		if z_min != z_max:
			print("对待放置的2D Z normal仿真，输入的z_min和z_max不相等，这将是其z坐标，请检查！")
		props = OrderedDict([
			("solver type", solver_type),
			("x min", x_min),
			("x max", x_max),
			("y min", y_min),
			("y max", y_max),
			("z", z_min),
			("background material", background_material),
			("x min bc", x_min_bc),
			("x max bc", x_max_bc),
			("y min bc", y_min_bc),
			("y max bc", y_max_bc)
		])
		ob_mode = FD.addfde(properties=props)
		FD.set("x min bc", x_min_bc)  # 吐槽：奇葩逻辑，设置了y min bc为periodic以后，再设置y max bc就会出错，明明bloch条件都不会
		if x_min_bc != "periodic":
			FD.set("x max bc", x_max_bc)
		FD.set("y min bc", y_min_bc)
		if y_min_bc != "periodic":
			FD.set("y max bc", y_max_bc)
	else:
		print("传入参数simulation_type设置错误，必须为"
			  "\n\t【2D X normal】【2D Y normal】【2D Z normal】"
			  "\n中的一个")  # 剩下的类型还没开始写
		props = OrderedDict()  # 这个要是不写，下面老是报警报，写一下去警报
		ob_mode = None
	return ob_mode


def GPU_on():
	"""对于3D FDTD仿真，使用GPU加速，函数默认仅提供1个GPU进行加速。此外，GPU加速仅支持3D FDTD仿真。"""
	FD = get_fdtd_instance()

	bc_ls = [FD.getnamed("FDTD", "x min bc"),
			 FD.getnamed("FDTD", "x max bc"),
			 FD.getnamed("FDTD", "y min bc"),
			 FD.getnamed("FDTD", "y max bc"),
			 FD.getnamed("FDTD", "z min bc"),
			 FD.getnamed("FDTD", "z max bc")
			 ]
	valid_bm = {"PML", "Metal", "Anti-Symmetric", "PMC", "Symmetric"}
	if not all(item in valid_bm for item in bc_ls):  # 列表推导式，检查边界条件是否都在有效边界条件列表中，not取反
		print("\t警告！程序尝试使用GPU加速计算，但开启失败，程序已自动设置为CPU模式继续运行!\n"
			  "\tGPU加速模式，边界条件仅支持下面的情形：\n"
			  "\t「PML」「Metal」「Anti-Symmetric」「PMC」「Symmetric」\n"
			  "\t不支持：「Periodic」或「Bloch」边界条件！\n")
		GPU_off()
		return False
	else:
		import os
		file_name = os.path.basename(FD.currentfilename())
		if file_name[0].isdigit() or (not file_name.isascii()):
			print("\t警告！GPU加速模式，文件名必须满足以下要求：\n"
				  "\t1：必须为全ASCII编码，不可以有中文字符\n"
				  "\t2：不可以以数字作为文件名开头\n"
				  "\t程序已自动设置为CPU模式继续运行")
			GPU_off()
			return False
		else:
			FD.setresource("FDTD", "GPU", 1)
			FD.setresource("FDTD", 1, "GPU Device", "Auto")
			FD.setnamed("FDTD", "express mode", 1)
			return True


def GPU_off():
	"""关闭GPU加速模式（回到CPU模式），一般用于开启GPU加速失败时的异常分支处理"""
	FD = get_fdtd_instance()
	FD.setresource("FDTD", "GPU", 0)
	# FD.setresource("FDTD", 1, "GPU Device", "GPU 0")
	FD.setnamed("FDTD", "express mode", 0)
	return True


def simulation_time_check():
	FD = get_fdtd_instance()
	FDTD_x_span = FD.getnamed("FDTD", "x max") - FD.getnamed("FDTD", "x min")
	FDTD_y_span = FD.getnamed("FDTD", "y max") - FD.getnamed("FDTD", "y min")
	FDTD_z_span = FD.getnamed("FDTD", "z max") - FD.getnamed("FDTD", "z min")
	simulation_time = FD.getnamed("FDTD", "simulation time")
	broadcast_distance = simulation_time / (1000 * 1e-15) * 100 * 1e-6  # 1000fs 对应的传播距离大约是88μm，取个整算100μm
	if broadcast_distance < FDTD_x_span or broadcast_distance < FDTD_y_span or broadcast_distance < FDTD_z_span:
		print("\t警告!仿真时间很可能小于光完全传播所需时间！")
		if simulation_time == 1e-12:
			print("\t仿真时长为默认的1000fs，请检查是否需要调节仿真时长！")
		return False
	else:
		return True
