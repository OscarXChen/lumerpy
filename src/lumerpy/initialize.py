def initialize(file_path=r"./temp/fsp", file_name=r"m1.0_test.fsp",
			   api_path=r"C:/Program Files/Lumerical/v241/api/python", GUI_hide_flag=False):
	import os
	import sys
	# 用户在这里设置 API 和文件路径
	if not os.path.exists(api_path):
		api_path = r"D:/Program Files/Lumerical/v241/api/python"
		if not os.path.exists(api_path):
			api_path = r"E:/Program Files/Lumerical/v241/api/python"
			if not os.path.exists(api_path):
				api_path = r"F:/Program Files/Lumerical/v241/api/python"
				if not os.path.exists(api_path):
					prompt_text = (f"API路径设置错误！\n"
								   f"\t未能从默认位置找到找到Lumerical API的位置\n"
								   f"\t请手动传入api_path,形如：\n"
								   f"\tinitialize(api_path=C:/Program Files/Lumerical/v241/api/python")
					raise FileNotFoundError(prompt_text)
	sys.path.append(os.path.normpath(api_path))  # 添加 API 路径以确保可以成功导入 lumapi
	import lumerpy as lupy
	lupy.tools.check_path_and_file(file_path=file_path, file_name=file_name)
	lupy.setup_paths(api_path, file_path, file_name)  # 设置路径到库
	# --------------------基本设置结束--------------------
	fdtd_instance = lupy.get_fdtd_instance(hide=GUI_hide_flag,
										   solution_type="FDTD")  # 创建fdtd实例，这应该是第一个实例，hide=True时，隐藏窗口
	FD = lupy.get_existing_fdtd_instance()  # 返回创建的实例，以便使用lumapi
	if not FD:
		print("未正确创建实例，请检查")
	u = 1e-6

	# --------------------现在既可以调用lumapi，也可以调用lupy库--------------------
	return FD
