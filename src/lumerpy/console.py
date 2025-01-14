# 本页用于直接复制到命令行终端窗口运行
import sys
import os

# 用户在这里设置 API 和文件路径
api_path = r"C:\Program Files\Lumerical\v241\api\python".replace("\\", "/")
file_path = r"E:\0_Work_Documents\Simulation\lumerpy\00_temp"
file_name = r"m00_temp.fsp"
sys.path.append(os.path.normpath(api_path))  # 添加 API 路径以确保可以成功导入 lumapi
import lumerpy as lupy
lupy.tools.check_path_and_file(file_path=file_path,file_name=file_name)
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
