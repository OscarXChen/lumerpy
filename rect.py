from .fdtd_manager import get_fdtd_instance
from collections import OrderedDict


def add_rect(name="", x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0, material="SiO2 (Glass) - Palik"):
	FD = get_fdtd_instance()
	props = OrderedDict([
		("name", name),
		("x min", x_min),
		("x max", x_max),
		("y min", y_min),
		("y max", y_max),
		("z min", z_min),
		("z max", z_max),
	])
	ob_rect = FD.addrect(properties=props)
	FD.set("material", material)  # 吐槽：addrect()函数居然没有材料属性
	return ob_rect


def add_slab(name="", x_min=0, x_max=0, y_min=0, y_max=0, z_min=0, z_max=0, material="Si (Silicon) - Palik"):
	FD = get_fdtd_instance()
	FD.addstructuregroup(name="slab")
	FD.set("x", 0)
	FD.set("y", 0)
	FD.set("z", 0)
	props = OrderedDict([
		("name", "slab layer"),
		("x min", x_min),
		("x max", x_max),
		("y min", y_min),
		("y max", y_max),
		("z min", z_min),
		("z max", z_max),
	])
	ob_slab = FD.addrect(properties=props)
	FD.set("material", "Si (Silicon) - Palik")  # 吐槽：addrect()函数居然没有材料属性
	FD.set("override mesh order from material database", 1)
	FD.set("mesh order", 3)
	# FD.addtogroup("slab")
	return ob_slab
