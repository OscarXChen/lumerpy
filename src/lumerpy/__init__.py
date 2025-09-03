# 创建库函数流程
# 在lupy目录下创建文件如：function.py
# 在function.py中添加：
# from .fdtd_manager import get_fdtd_instance
# 在function中写函数，例如：
# def test_function():
#	FD = get_fdtd_instance()	# 获取实例
# 	print("test")
# 在__init__.py中写添加：
# from .function.py import test_function
# 在主函数中直接调用lupy.test_function()即可

from .fdtd_manager import get_fdtd_instance
from .fdtd_manager import setup_paths
from .fdtd_manager import setup_api_path
from .fdtd_manager import get_existing_fdtd_instance
from .fdtd_manager import open_fdtd
from .fdtd_manager import close_fdtd_instance

from .rect import add_rect
from .rect import add_slab

from .source import add_source_plane
from .source import add_source_dipole
from .source import add_source_gaussian
from .source import add_source_mode

from .eri import set_neff_monitor
from .eri import cal_eff_reg
from .eri import cal_eff_delta

from .monitor import add_power_monitor
from .monitor import add_global_monitor
from .monitor import add_power_monitor_metaline

from .simulation import add_simulation_fdtd
from .simulation import add_simulation_fde

from .donn import add_metalines

from .tools import u_print

from .console import console

from .data_process import *
def help():
	hello()


def hello():
	print("\t欢迎使用Lumerpy库\n"
		  "\t这是一个为了方便调用而二次包装Lumerical Python API的库\n"
		  "\t可以在python console中输入「lumerpy.console(globals())」以快速开始\n"
		  "\t详情请参见Github页面：https://github.com/OscarXChen/lumerpy")


def version():
	FD = get_fdtd_instance()
	lupy_version = "1.4.0"
	version_date = "2025.09.03"
	print(f"Lumerpy库版本：{lupy_version}\n"
		  f"Lumerical版本：{FD.version()}\n"
		  f"发布时间：{version_date}")
	return FD.version()  # 调用 FDTD 实例的 version() 方法

def author():
	print(f"原始作者：陈泽煊，初始发布于2025.01.01")

def miao():
	print("阳猫宇飞喵~")

