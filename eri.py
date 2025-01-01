# eri=Effective refractive index
from .fdtd_manager import get_fdtd_instance
import numpy as np


def set_neff_monitor(pre_name="", direction="x", x1_min=0, x1_max=0, y1_min=0, y1_max=0, z1_min=0, z1_max=0, x2_min=0,
					 x2_max=0, y2_min=0, y2_max=0, z2_min=0, z2_max=0):
	"""这个函数是为了计算有效折射率而自动化放置监视器,函数接收基本参数，返回2个相位监视器对象。计算逻辑为放置2个2D监视器看相位计算相位差。当监视器x轴垂直时，x1_min为监视器1的x轴位置"""
	FD = get_fdtd_instance()
	if direction == "x":
		ob_moni_1 = FD.addpower()
		FD.set("name", pre_name + "_front")
		FD.set("monitor type", "2D X-normal")
		FD.set("x", x1_min)
		FD.set("y min", y1_min)
		FD.set("y max", y1_max)
		FD.set("z min", z1_min)
		FD.set("z max", z1_max)

		ob_moni_2 = FD.addpower()
		FD.set("name", pre_name + "_back")
		FD.set("monitor type", "2D X-normal")
		FD.set("x", x2_min)
		FD.set("y min", y2_min)
		FD.set("y max", y2_max)
		FD.set("z min", z2_min)
		FD.set("z max", z2_max)
	elif direction == "y":
		ob_moni_1 = FD.addpower()
		FD.set("name", pre_name + "_front")
		FD.set("monitor type", "2D Y-normal")
		FD.set("y", y1_min)
		FD.set("x min", x1_min)
		FD.set("x max", x1_max)
		FD.set("z min", z1_min)
		FD.set("z max", z1_max)

		ob_moni_2 = FD.addpower()
		FD.set("name", pre_name + "_back")
		FD.set("monitor type", "2D Y-normal")
		FD.set("y", y2_min)
		FD.set("x min", x2_min)
		FD.set("x max", x2_max)
		FD.set("z min", z2_min)
		FD.set("z max", z2_max)
	elif direction == "z":
		ob_moni_1 = FD.addpower()
		FD.set("name", pre_name + "_front")
		FD.set("monitor type", "2D Z-normal")
		FD.set("z", z1_min)
		FD.set("x min", x1_min)
		FD.set("x max", x1_max)
		FD.set("y min", y1_min)
		FD.set("y max", y1_max)

		ob_moni_2 = FD.addpower()
		FD.set("name", pre_name + "_back")
		FD.set("monitor type", "2D Z-normal")
		FD.set("z", z2_min)
		FD.set("x min", x2_min)
		FD.set("x max", x2_max)
		FD.set("y min", y2_min)
		FD.set("y max", y2_max)
	else:
		print("输入的计算方向错误")
		ob_moni_1, ob_moni_2 = False, False
	return ob_moni_1, ob_moni_2


def cal_eff_reg(eri_monitor, axis="x", direction="Ez", wavelength=1.55e-6):
	'''通过线监视器获得电场数据，线性拟合后获得斜率，继而计算折射率'''
	if axis != "x" and axis != "y" and axis != "z":
		print("\t【错误】\n\t输入的计算轴axis必须为「x」「y」「z」中的一个")
		return 0
	if not direction in ["Ex", "Ey", "Ez"]:
		print("\t【错误】\n\t计算的偏振方向direction必须为「Ex」「Ey」「Ez」中的一个")
		return 0

	FD = get_fdtd_instance()

	x = FD.getresult(eri_monitor, axis).ravel()
	E_polarization = FD.getresult(eri_monitor, direction).ravel()
	phase = []
	for i in E_polarization:
		phase.append(np.angle(i))
	phase = np.array(phase)
	phase = np.unwrap(phase)
	# cal_eff_reg(x,phase)
	slope, intercept = np.polyfit(x, phase, 1)  # 线性拟合获得斜率和截距
	eff = slope * wavelength / 2 / 3.1415927  # 计算有效折射率
	return eff
