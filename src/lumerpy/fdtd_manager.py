import os
import lumapi		# 此处会报一个错，提示没有lumapi库，不用管它
import time

_fdtd_instance = None
_api_path = None
_file_path = None
_file_name = None


def setup_paths(api_path, file_path, file_name):
	"""设置 API 路径、文件路径和文件名"""
	global _api_path, _file_path, _file_name
	_api_path = api_path
	_file_path = file_path
	_file_name = file_name


def get_fdtd_instance(hide=False, solution_type="FDTD"):
	global _fdtd_instance
	if solution_type == "FDTD":
		if _fdtd_instance == None:
			if _api_path == None:
				raise ValueError("错误！程序必须设置 API 路径！\n请先调用setup_paths()函数或检查API是否正确设置！")
			if _file_path == "" or _file_name == "":
				print("警告！未设置文件路径，程序结束时将不会保存文件！")
				time.sleep(0.5)
				_fdtd_instance = lumapi.FDTD(hide=False)  # 此时程序会打开一个新文件
			else:
				# 创建 FDTD 实例并传入 filename 和 hide 参数
				filename = os.path.join(_file_path, _file_name)
				_fdtd_instance = lumapi.FDTD(filename=filename, hide=hide)  # 此时程序会打开指定的文件
	elif solution_type == "MODE":
		if _fdtd_instance is None:
			if _api_path is None:
				raise ValueError("错误！程序必须设置 API 路径！\n请先调用setup_paths()函数或检查API是否正确设置！")
			if _file_path is None or _file_name is None:
				print("警告！未设置文件路径，程序结束时将不会保存文件！")
				time.sleep(0.5)
				_fdtd_instance = lumapi.MODE(hide=False)  # 此时程序会打开一个新文件
			else:
				# 创建 FDTD 实例并传入 filename 和 hide 参数
				filename = os.path.join(_file_path, _file_name)
				_fdtd_instance = lumapi.MODE(filename=filename, hide=hide)  # 此时程序会打开指定的文件
	else:
		print("设置的solution_type必须为【FDTD】或【MODE】，请检查输入！")
		time.sleep(3)
	return _fdtd_instance


def get_existing_fdtd_instance():
	"""获取已存在的 FDTD 实例，若不存在则返回 None"""
	return _fdtd_instance


def close_fdtd_instance():
	global _fdtd_instance
	if _fdtd_instance is not None:
		_fdtd_instance.close()  # 不知道为什么此处弹了一个警告
		_fdtd_instance = None
