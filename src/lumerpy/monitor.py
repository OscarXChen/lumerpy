from .fdtd_manager import get_fdtd_instance

u = 1e-6


def add_power_monitor(name="phase", x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0,
					  monitor_type="2D Z-normal"):
	FD = get_fdtd_instance()
	ob_power_monitor = FD.addpower()
	FD.set("name", name)

	if monitor_type == "2D X-normal":
		if x_min != x_max:
			print("对待放置的2D X-normal监视器，输入的x_min和x_max不相等，这将是其x坐标，请检查！")
		else:
			FD.set("monitor type", monitor_type)
			FD.set("x", x_min)
			FD.set("y min", y_min)
			FD.set("y max", y_max)
			FD.set("z min", z_min)
			FD.set("z max", z_max)
	elif monitor_type == "2D Y-normal":
		if y_min != y_max:
			print("对待放置的2D Y-normal监视器，输入的y_min和y_max不相等，这将是其y坐标，请检查！")
		else:
			FD.set("monitor type", monitor_type)
			FD.set("y", y_min)
			FD.set("x min", x_min)
			FD.set("x max", x_max)
			FD.set("z min", z_min)
			FD.set("z max", z_max)
	elif monitor_type == "2D Z-normal":
		if z_min != z_max:
			print("对待放置的2D Z-normal监视器，输入的z_min和z_max不相等，这将是其z坐标，请检查！")
		else:
			FD.set("monitor type", monitor_type)
			FD.set("x min", x_min)
			FD.set("x max", x_max)
			FD.set("y min", y_min)
			FD.set("y max", y_max)
			FD.set("z", z_min)
	elif monitor_type == "Linear X":
		if y_min != y_max or z_min != z_max:
			print("对待放置的Linear X监视器，输入的y_min和y_max，和（或）z_min和z_max不相等，这将是其y，z坐标，请检查！")
		else:
			FD.set("monitor type", monitor_type)
			FD.set("x min", x_min)
			FD.set("x max", x_max)
			FD.set("y", y_min)
			FD.set("z", z_min)
	elif monitor_type == "Linear Y":
		if x_min != x_max or z_min != z_max:
			print("对待放置的Linear Y监视器，输入的x_min和x_max，和（或）z_min和z_max不相等，这将是其x，z坐标，请检查！")
		else:
			FD.set("monitor type", monitor_type)
			FD.set("y min", y_min)
			FD.set("y max", y_max)
			FD.set("x", x_min)
			FD.set("z", z_min)
	elif monitor_type == "Linear Z":
		if x_min != x_max or y_min != y_max:
			print("对待放置的Linear Z监视器，输入的x_min和x_max，和（或）y_min和y_max不相等，这将是其x，y坐标，请检查！")
		else:
			FD.set("monitor type", monitor_type)
			FD.set("z min", z_min)
			FD.set("z max", z_max)
			FD.set("x", x_min)
			FD.set("y", y_min)
	else:
		print("传入参数monitor_type设置错误，必须为"
			  "\n\t【2D X-normal】【2D Y-normal】【2D Z-normal】"
			  "\n\t【Linear X】【Linear Y】【Linear Z】\n中的一个")
	return ob_power_monitor


def add_global_monitor(name="global",
					   monitor_type="2D Z-normal",dipole_avoid=False,delta_x=0.1*u):
	# 添加全局监视器，看场俯视图
	FD = get_fdtd_instance()
	if FD.getnamednumber("FDTD"):
		sim_name = "FDTD"
	elif FD.getnamednumber("FDE"):
		sim_name = "FDE"
	elif FD.getnamednumber("varFDTD"):
		sim_name = "varFDTD"
	elif FD.getnamednumber("EME"):
		sim_name = "EME"
	else:
		print("警告！未找到FDTD Solution或MODE Solution对应的仿真区域，无法创建全局监视器")
		return 0
	FD.select(sim_name)
	x_min = FD.getnamed("FDTD", "x min")
	x_max = FD.getnamed("FDTD", "x max")
	y_min = FD.getnamed("FDTD", "y min")
	y_max = FD.getnamed("FDTD", "y max")
	z_min = FD.getnamed("FDTD", "z min")
	z_max = FD.getnamed("FDTD", "z max")

	# dipole附近的场经常是非物理的，如果dipole_avoid非0，那么就往x正方向挪delta_x的距离，避免非物理波源的影响
	if dipole_avoid:
		if FD.getnamednumber("dipole"):
			if monitor_type == "2D Z-normal":
				x_min = FD.getnamed("dipole", "x")
				x_min = x_min + delta_x
		else:
			print("本函数还没写完，没找到名为dipole的源，先这样吧")
			return False

	ob_power_monitor = None

	if monitor_type == "2D X-normal":
		ob_power_monitor = add_power_monitor(name=name, x_min=(x_min + x_max) / 2, x_max=(x_min + x_max) / 2,
											 y_min=y_min, y_max=y_max,
											 z_min=z_min, z_max=z_max, monitor_type="2D X-normal")
	elif monitor_type == "2D Y-normal":
		ob_power_monitor = add_power_monitor(name=name, x_min=x_min, x_max=x_max,
											 y_min=(y_min + y_max) / 2, y_max=(y_min + y_max) / 2,
											 z_min=z_min, z_max=z_max, monitor_type="2D Y-normal")
	elif monitor_type == "2D Z-normal":
		ob_power_monitor = add_power_monitor(name=name, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
											 z_min=(z_min + z_max) / 2, z_max=(z_min + z_max) / 2,
											 monitor_type="2D Z-normal")
	elif monitor_type == "Linear X":
		ob_power_monitor = add_power_monitor(name=name, x_min=x_min, x_max=x_max, y_min=(y_min + y_max) / 2,
											 y_max=(y_min + y_max) / 2,
											 z_min=(z_min + z_max) / 2, z_max=(z_min + z_max) / 2,
											 monitor_type="Linear X")
	elif monitor_type == "Linear Y":
		ob_power_monitor = add_power_monitor(name=name, x_min=(x_min + x_max) / 2, x_max=(x_min + x_max) / 2,
											 y_min=y_min, y_max=y_max,
											 z_min=(z_min + z_max) / 2, z_max=(z_min + z_max) / 2,
											 monitor_type="Linear Y")
	elif monitor_type == "Linear Z":
		ob_power_monitor = add_power_monitor(name=name, x_min=(x_min + x_max) / 2, x_max=(x_min + x_max) / 2,
											 y_min=(y_min + y_max) / 2,
											 y_max=(y_min + y_max) / 2, z_min=z_min, z_max=z_max,
											 monitor_type="Linear Z")
	else:
		print("传入参数monitor_type设置错误，必须为"
			  "\n\t【2D X-normal】【2D Y-normal】【2D Z-normal】"
			  "\n\t【Linear X】【Linear Y】【Linear Z】\n中的一个")
		return None
	return ob_power_monitor


def add_power_monitor_metaline(monitor_name="", metaline_name=""):
	"""给指定名字的衍射线添加一条线类型功率监视器，几何位置位于衍射线中心"""
	FD = get_fdtd_instance()
	x_min = FD.getnamed(metaline_name, "x min")
	x_max = FD.getnamed(metaline_name, "x max")
	y = FD.getnamed(metaline_name, "y")
	z = FD.getnamed(metaline_name, "z")
	ob_power_monitor = add_power_monitor(name=monitor_name, x_min=x_min, x_max=x_max, y_min=y, y_max=y, z_min=z,
										 z_max=z, monitor_type="Linear X")
	return ob_power_monitor
