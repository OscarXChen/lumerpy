# 真值表定义规范如下：
# A1 A0 B1 B0 L1 L2 L3 X1 X0
# A1为1，则输入A=1，A0为1，则输入A=0
# X1为1，则输出X为1，X0为1，则输出X为0
# 显然，这套规则下，A1·A0=0，不可能同时为1
def get_true_table_basics():
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 0, 0,
		0, 1, 0, 0,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1, ]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
	]).reshape(-1, 3)
	outputs = np.array([
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_only_NAND():
	# 重构屎山，仅有与非门
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
	]
	).reshape(-1, 4)
	logic_control = np.array([
	])
	outputs = np.array([
		0, 1,
		1, 0,
		1, 0,
		1, 0,
	]).reshape(-1, 2)
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs

def get_true_table_only_xor_origin():
	# 异或自救
	import numpy as np
	inputs = np.array([
		0, 0,
		0, 1,
		1, 0,
		1, 1,
	]
	).reshape(-1, 2)
	logic_control = np.array([
	])
	outputs = np.array([
		0, 1,
		1, 0,
		1, 0,
		0, 1,
	]).reshape(-1, 2)
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs

def get_true_table_only_xor_expand():
	# 异或自救
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
	]
	).reshape(-1, 4)
	logic_control = np.array([
	])
	outputs = np.array([
		0, 1,
		1, 0,
		1, 0,
		0, 1,
	]).reshape(-1, 2)
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs

def get_true_table_basics_2():
	# 论文4.1
	import numpy as np
	# 与非或非异或
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
	]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
	]).reshape(-1, 3)
	outputs = np.array([
		0, 1,
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
		1, 0,
		0, 1,
		1, 0,
		1, 0,
		0, 1,
	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_logic_func_complex_1():
	# 论文4.2
	import numpy as np
	# 与非或非异或
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
	]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
	]).reshape(-1, 2)

	outputs = np.array([
		0, 0, 0, 0, 0, 1,
		0, 0, 0, 0, 0, 1,
		0, 0, 1, 0, 0, 0,
		0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 1,
		0, 0, 0, 1, 0, 0,
		0, 1, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0,
	]).reshape(-1, 6)
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_logic_func_adder():
	# 论文4.3
	import numpy as np
	inputs = np.array([
		# 0, 1, 0, 1, 0, 1,
		# 0, 1, 0, 1, 1, 0,
		# 0, 1, 1, 0, 0, 1,
		# 0, 1, 1, 0, 1, 0,
		# 1, 0, 0, 1, 0, 1,
		# 1, 0, 0, 1, 1, 0,
		# 1, 0, 1, 0, 0, 1,
		# 1, 0, 1, 0, 1, 0,
		0, 1, 0, 1, 0, 1,
		0, 1, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 1,
		0, 1, 1, 0, 1, 0,
		1, 0, 0, 1, 0, 1,
		1, 0, 0, 1, 1, 0,
		1, 0, 1, 0, 0, 1,
		1, 0, 1, 0, 1, 0,
		0, 1, 0, 1, 0, 1,
		0, 1, 0, 1, 1, 0,
		0, 1, 1, 0, 1, 0,
		0, 1, 1, 0, 0, 1,
		1, 0, 0, 1, 1, 0,
		1, 0, 0, 1, 0, 1,
		1, 0, 1, 0, 0, 1,
		1, 0, 1, 0, 1, 0,

	]
	).reshape(-1, 6)
	logic_control = np.array([
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
	]).reshape(-1, 2)

	outputs = np.array([
		0, 1, 0, 0,
		1, 0, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 1,
		1, 0, 0, 0,
		0, 0, 0, 1,
		0, 0, 0, 1,
		0, 0, 1, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,

	]).reshape(-1, 4)
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_logic_func_adder_expand():
	# 论文4.3
	import numpy as np
	inputs = np.array([
		# 0, 1, 0, 1, 0, 1,
		# 0, 1, 0, 1, 1, 0,
		# 0, 1, 1, 0, 0, 1,
		# 0, 1, 1, 0, 1, 0,
		# 1, 0, 0, 1, 0, 1,
		# 1, 0, 0, 1, 1, 0,
		# 1, 0, 1, 0, 0, 1,
		# 1, 0, 1, 0, 1, 0,
		0, 1, 0, 1, 0, 1, 1,
		0, 1, 0, 1, 1, 0, 0,
		0, 1, 1, 0, 0, 1, 0,
		0, 1, 1, 0, 1, 0, 0,
		1, 0, 0, 1, 0, 1, 0,
		1, 0, 0, 1, 1, 0, 0,
		1, 0, 1, 0, 0, 1, 0,
		1, 0, 1, 0, 1, 0, 0,
		0, 1, 0, 1, 0, 1, 0,
		0, 1, 0, 1, 1, 0, 0,
		0, 1, 1, 0, 1, 0, 0,
		0, 1, 1, 0, 0, 1, 0,
		1, 0, 0, 1, 1, 0, 0,
		1, 0, 0, 1, 0, 1, 0,
		1, 0, 1, 0, 0, 1, 0,
		1, 0, 1, 0, 1, 0, 0,

	]
	).reshape(-1, 7)
	logic_control = np.array([
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
		0, 1,
	]).reshape(-1, 2)

	outputs = np.array([
		0, 1, 0, 0,
		1, 0, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 1,
		1, 0, 0, 0,
		0, 0, 0, 1,
		0, 0, 0, 1,
		0, 0, 1, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,

	]).reshape(-1, 4)
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs

def get_true_table_multiplier_2bit():
	# 毕业后仍然要做的新乘法器
	import numpy as np
	# 可以处理2个2 bit输入的乘法器
	inputs = np.array([
		1, 0, 1, 0, 1, 0, 1, 0,
		1, 0, 1, 0, 1, 0, 0, 1,
		1, 0, 1, 0, 0, 1, 1, 0,
		1, 0, 1, 0, 0, 1, 0, 1,
		1, 0, 0, 1, 1, 0, 1, 0,
		1, 0, 0, 1, 1, 0, 0, 1,
		1, 0, 0, 1, 0, 1, 1, 0,
		1, 0, 0, 1, 0, 1, 0, 1,
		0, 1, 1, 0, 1, 0, 1, 0,
		0, 1, 1, 0, 1, 0, 0, 1,
		0, 1, 1, 0, 0, 1, 1, 0,
		0, 1, 1, 0, 0, 1, 0, 1,
		0, 1, 0, 1, 1, 0, 1, 0,
		0, 1, 0, 1, 1, 0, 0, 1,
		0, 1, 0, 1, 0, 1, 1, 0,
		0, 1, 0, 1, 0, 1, 0, 1,
	]
	).reshape(-1, 8)
	logic_control = np.array([

	])



	outputs = np.array([
		1, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0,
		0, 0, 1, 0, 0, 0, 0,
		0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		1, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 1, 0,
		0, 0, 0, 0, 0, 0, 1,
	]).reshape(-1, 7)
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs
def get_true_table_NAND_NOR_XOR():
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1, ]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
	]).reshape(-1, 3)
	outputs = np.array([
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_xor_nxor_nand():
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
	]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
	]).reshape(-1, 3)
	outputs = np.array([
		0, 1,
		1, 0,
		1, 0,
		0, 1,
		1, 0,
		0, 1,
		0, 1,
		1, 0,
		0, 1,
		1, 0,
		1, 0,
		1, 0,
	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_only_xor():
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
	]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
	]).reshape(-1, 3)
	outputs = np.array([
		0, 1,
		1, 0,
		1, 0,
		0, 1,

	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_only_xor2():
	import numpy as np
	inputs = np.array([
		1, 0, 1,
		1, 0, 0,
		0, 1, 1,
		0, 1, 0,
	]
	).reshape(-1, 3)
	logic_control = np.array([
		0,
		1,
		0,
		1,
	]).reshape(-1, 1)
	outputs = np.array([
		0, 1,
		1, 0,
		1, 0,
		0, 1,

	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_only_xor3():
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0, 1,
		1, 0, 1, 0, 0,
		1, 0, 0, 1, 1,
		1, 0, 0, 1, 0,
		0, 1, 1, 0, 1,
		0, 1, 1, 0, 0,
		0, 1, 0, 1, 1,
		0, 1, 0, 1, 0,
	]
	).reshape(-1, 5)
	logic_control = np.array([
		0,
		1,
		0,
		1,
		0,
		1,
		0,
		1
	]).reshape(-1, 1)
	outputs = np.array([
		1, 0,
		0, 1,
		0, 1,
		1, 0,
		0, 1,
		1, 0,
		1, 0,
		0, 1,

	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_only_xor3_kernel():
	import numpy as np
	inputs = np.array([
		1, 0, 1, 0, 1, 0,
		1, 0, 1, 0, 0, 1,
		1, 0, 0, 1, 1, 0,
		1, 0, 0, 1, 0, 1,
		0, 1, 1, 0, 1, 0,
		0, 1, 1, 0, 0, 1,
		0, 1, 0, 1, 1, 0,
		0, 1, 0, 1, 0, 1,
	]
	).reshape(-1, 6)
	logic_control = np.array([
		0,
		0,
		0,
		0,
		0,
		0,
		0,
		1,
	]).reshape(-1, 1)
	outputs = np.array([
		1, 0,
		0, 1,
		0, 1,
		1, 0,
		0, 1,
		1, 0,
		1, 0,
		0, 1,

	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_shift_test1():
	import numpy as np
	inputs = np.array([
		1,
	]
	).reshape(-1, 1)
	logic_control = np.array([
		0,

	]).reshape(-1, 1)
	outputs = np.array([
		0, 1

	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_logic_function():
	# 暂时不用
	import numpy as np
	# 与非或非异或
	inputs = np.array([
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 0, 0, 1,
		0, 1, 1, 0,
		0, 1, 0, 1,
	]
	).reshape(-1, 4)
	logic_control = np.array([
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		1, 0, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 1, 0,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
		0, 0, 1,
	]).reshape(-1, 3)
	outputs = np.array([
		0, 1,
		1, 0,
		1, 0,
		1, 0,
		0, 1,
		0, 1,
		0, 1,
		1, 0,
		0, 1,
		1, 0,
		1, 0,
		0, 1,
	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_logic_test1():
	# 暂时不用
	import numpy as np
	# 与非或非异或
	inputs = np.array([
		0, 1, 0, 1,
		1, 0, 1, 0,
		1, 1, 1, 1,

	]
	).reshape(-1, 4)
	# logic_control = np.array([
	# 	1, 0, 0,
	# 	0, 1, 0,
	# 	0, 0, 1,
	# ]).reshape(-1, 3)
	logic_control = np.array([

	]).reshape(-1, 3)
	outputs = np.array([
		0, 1, 0,
		1, 0, 0,
		0, 0, 1,
	]).reshape(-1, 3)
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def get_true_table_NOT():
	import numpy as np
	inputs = np.array([
		1, 0,
		0, 1]
	).reshape(-1, 2)
	logic_control = []
	outputs = np.array([
		0, 1,
		1, 0,
	]).reshape(-1, 2)
	outputs = outputs[:, -1]  # 把输出视作一个分类问题，如果第一个探测器为1那么标签是0，如果第二个探测器为1那么标签是1
	# true_table = np.hstack((inputs, logic_control, outputs))
	return inputs, logic_control, outputs


def ReLU_shift(x, channels):
	mid_value = 2 ** (channels - 1)  # 中值判定为，最高位为1，其余为0的情形
	x = x - mid_value
	if x < 0:
		y = 0
	else:
		y = x
	y = min(y, 2 ** channels - 1)  # 根据位数自动截断，临时
	return y

def simple_linear(x, channels):
	# mid_value = 2 ** (channels - 1)  # 中值判定为，最高位为1，其余为0的情形
	# x = x - mid_value
	# if x < 0:
	# 	y = 0
	# else:
	y = x
	y = min(y, 2 ** channels - 1)  # 根据位数自动截断，临时
	return y

def Leaky_ReLU_shift(x, channels, alpha=0.01):
	mid_value = 2 ** (channels - 1)  # 中值判定为，最高位为1，其余为0的情形
	x = x - mid_value
	if x < 0:
		y = x * alpha
	else:
		y = x
	y = min(y, 2 ** channels - 1)  # 根据位数自动截断，临时
	return y


def Sigmoid_shift(x, channels):
	mid_value = 2 ** (channels - 1)  # 中值判定为，最高位为1，其余为0的情形
	x = x - mid_value
	import math
	y = 1 / (1 + math.e ** x ** (-x))
	return y


def Soft_Plus_shift(x, channels):
	mid_value = 2 ** (channels - 1)  # 中值判定为，最高位为1，其余为0的情形
	x = x - mid_value
	import math
	y = math.log(1 + math.e ** x)
	return y


def truth_table_to_func(outputs):
	"""
	将真值表转化为一个函数（输入是整数索引）
	:param outputs: 输出列表，例如 [1,1,1,0] 表示 2位输入的 NAND
	:return: 一个可调用的函数 f(i)
	"""
	mapping = {i: out for i, out in enumerate(outputs)}

	def func(x):
		if x not in mapping:
			raise ValueError(f"输入 {x} 超出了范围 (0~{len(outputs) - 1})")
		return mapping[x]

	return func
def func_NAND(x,channels):
	outputs = [1, 1, 1, 0]
	nand_func = truth_table_to_func(outputs)
	return nand_func(x)

def func_basic_gates(x,channels):
	outputs = [1, 1, 1, 0]
	nand_func = truth_table_to_func(outputs)
	return nand_func(x)
def call_plot_func(channels,func):
	"""实际上起作用的是plots.py文件里的plot_func()函数，但是由于太常用了，这里写一个提醒一下"""
	import diffrann_plots as lupy_plots
	lupy_plots.plot_func(channels=channels,func=func)