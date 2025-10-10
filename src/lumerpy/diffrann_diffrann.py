import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt
import math
import pandas as pd


def hello():
	print("hello2")


def get_diffractive_weights(size=(28, 28), z=1.05e-2, wave_length=299792458 / 0.4e12):
	# 原文注释说“板间距离，3 cm”，这个3cm应当来自论文提到的40个λ距离，大概就是3cm，但是不知道为什么这里改成了1.05e-2
	# weights = np.empty([N ** 2, N ** 2], dtype=complex)  # 预分配存储衍射权重的矩阵（复数类型），这一步似乎并不必要，因为后面会再写weights
	# 波长 λ（0.4 THz 频率对应的波长）
	# print(1e3 * wave_length)  # 以毫米（mm）为单位打印波长

	# 用D模拟光场的物理尺寸，wave_length/2根据奈奎斯特采样定理选的
	# 此处，N既是像素数，又是模拟中一个维度上的网格点数量，每个像素是“一块”面积
	# 意思就是：“在现实的物理空间，我们只有这么大的一块区域（边长 D），被离散为 N×N。”
	dy, dx = size  # 因为size是行*列，刚好和坐标顺序反过来
	# D = N * (wave_length / 2)  # 计算整个光场的物理尺寸
	# x1 = np.linspace(0, D, N)  # 生成 x 方向的均匀网格
	# y1 = np.linspace(0, D, N)  # 生成 y 方向的均匀网格
	# x2 = np.linspace(0, D, N)  # 目标平面上的 x 网格
	# y2 = np.linspace(0, D, N)  # 目标平面上的 y 网格
	#
	# xg1, yg1, xg2, yg2 = np.meshgrid(x1, y1, x2, y2)  # 生成4维网格，便于后面计算用来表征从1平面到2平面的所有路径
	# r = np.sqrt((xg1 - xg2) ** 2 + (yg1 - yg2) ** 2 + z ** 2)  # 计算传播距离r，注意，r是一个4维数组，表征所有可能的距离
	# dA = (D / N) ** 2  # 计算每一块像素对应的差分面积 dA
	# # 根据瑞利索末菲衍射计算衍射核，同样w也是一个4维数组
	# w = (z / r ** 2) * (1 / (2 * np.pi * r) + 1 / (1j * wave_length)) * np.exp(1j * 2 * np.pi * r / wave_length) * dA
	# weights = w.reshape(N ** 2, N ** 2)  # 重整形为2维，weights.shape=(784,784)
	Dx = dx * (wave_length / 2)  # 计算整个光场的物理尺寸
	Dy = dy * (wave_length / 2)  # 计算整个光场的物理尺寸
	x1 = np.linspace(0, Dx, dx)  # 生成 x 方向的均匀网格
	y1 = np.linspace(0, Dy, dy)  # 生成 y 方向的均匀网格
	x2 = np.linspace(0, Dx, dx)  # 目标平面上的 x 网格
	y2 = np.linspace(0, Dy, dy)  # 目标平面上的 y 网格

	xg1, yg1, xg2, yg2 = np.meshgrid(x1, y1, x2, y2)  # 生成4维网格，便于后面计算用来表征从1平面到2平面的所有路径
	r = np.sqrt((xg1 - xg2) ** 2 + (yg1 - yg2) ** 2 + z ** 2)  # 计算传播距离r，注意，r是一个4维数组，表征所有可能的距离
	dA = (Dx / dx) * (Dy / dy)  # 计算每一块像素对应的差分面积 dA
	# 根据瑞利索末菲衍射计算衍射核，同样w也是一个4维数组
	w = (z / r ** 2) * (1 / (2 * np.pi * r) + 1 / (1j * wave_length)) * np.exp(1j * 2 * np.pi * r / wave_length) * dA
	weights = w.reshape(dx * dy, dx * dy)  # 重整形为2维，weights.shape=(784,784)
	return weights


class D2NNModel(tf.keras.Model):
	def __init__(self, size, weights, pd_positions_single, pd_positions_all, num_layers=5, mod_type="all",
				 noise_phase_flag=False, noise_amp_flag=False):
		# def __init__(self, size, weights, pd_positions_single, pd_positions_all, num_layers=5):
		"""
		N: 图像尺寸（28）
		weights: 预先计算好的衍射传播矩阵 (784, 784), complex64
		pd_positions_single: (10, 784)
		pd_positions_all: (1, 784)
		num_layers: 3D 打印的衍射层数，本例中为5
		"""
		super().__init__()  # super().__init__()调用父类（即 tf.keras.Model）的构造方法，确保父类的初始化逻辑也被执行
		self.size = size
		# 将 numpy 的预计算矩阵包装成 Tensor 常量
		self.w = tf.constant(weights, dtype=tf.complex64)  # shape=(784, 784)
		self.pd_positions_single = tf.constant(pd_positions_single, dtype=tf.float32)  # shape=(10, 784)
		self.pd_positions_all = tf.constant(pd_positions_all, dtype=tf.float32)  # shape=(1, 784)
		self.num_layers = num_layers

		# 每层都有两个可训练变量：alpha (振幅) & theta (相位)
		# 整个模型 5 层 -> 5对 (alpha, theta)
		self.alpha_vars = []
		self.theta_vars = []
		row, col = size
		if mod_type == "all":
			amp_mod_flag = True
			pha_mod_flag = True
		elif mod_type == "amp":
			amp_mod_flag = True
			pha_mod_flag = False
		elif mod_type == "pha":
			amp_mod_flag = False
			pha_mod_flag = True
		else:
			print("警告！\n\t参数「mod_type」仅允许\n"
				  "\t「all」、「amp」、「pha」三种可能输入\n"
				  "\t现已禁止幅度和相位调制")
			amp_mod_flag = False
			pha_mod_flag = False
		for i in range(num_layers):
			# 这里需要注明一下，alpha初始化为1是表征一种无衰减的振幅
			# 而theta初始化为0.5是作为区间[0,1]的中间值，后面计算相位调制的时候乘了2π，也就是随机猜测初始化相位值是π
			# alpha 初始化为 1
			alpha_init = tf.ones(shape=(1, row * col), dtype=tf.float32)
			# theta 初始化为 0.5
			theta_init = tf.fill([1, row * col], 0.5)
			# 分别设置是否采用幅度调制和相位调制
			alpha_var = tf.Variable(initial_value=alpha_init,
									# trainable=True,
									trainable=amp_mod_flag,
									# trainable=False,
									name=f'alpha_{i}')
			theta_var = tf.Variable(initial_value=theta_init,
									trainable=pha_mod_flag,
									# trainable=True,
									# trainable=False,
									name=f'theta_{i}')

			self.alpha_vars.append(alpha_var)
			self.theta_vars.append(theta_var)
		self.noise_phase_flag = noise_phase_flag
		self.noise_amp_flag = noise_amp_flag

	def diffraction_layer(self, X, layer_index):
		"""
		单层衍射计算：
		X: shape=(batch_size, N*N)，复数
		layer_index: 第几层(0~4)
		"""
		# 1) 读取当前层的可训练变量
		a = self.alpha_vars[layer_index]  # shape=(1, N*N)
		t = self.theta_vars[layer_index]  # shape=(1, N*N)

		if self.noise_phase_flag:
			# 生成高斯分布噪声，shape 和 t 一致
			noise = tf.random.normal(
				shape=tf.shape(t),  # 或者直接写 [batch_size, row*col] 也行
				mean=0.0,  # 均值
				stddev=tf.reduce_max(t) * 0.15,  # 标准差，以调制值最大值的10%设定标准差
				dtype=tf.float32
			)
			noise = tf.stop_gradient(noise)
			t = t + noise
		if self.noise_amp_flag:
			# 生成高斯分布噪声，shape 和 t 一致
			noise = tf.random.normal(
				shape=tf.shape(a),  # 或者直接写 [batch_size, row*col] 也行
				mean=-0.0001,  # 均值
				stddev=tf.reduce_max(a) * 0.01,  # 标准差，以调制值最大值的10%设定标准差
				dtype=tf.float32
			)
			noise = tf.stop_gradient(noise)
			a = a + noise
		# 2) 计算相位调制: θ = 2π*t
		pi = tf.constant(math.pi, dtype=tf.float32)
		theta = 2.0 * pi * t
		phase_mod = tf.complex(tf.cos(theta), tf.sin(theta))  # shape=(1, N*N)

		# 3) 计算振幅调制: ReLU(a) 归一化
		a_relu = tf.nn.relu(a)  # 先保证非负，这是一个物理意义上的约束，并不同于非线性激活函数
		# 避免除以 0，需要加个很小的数epsilon
		a_mod = a_relu / (tf.reduce_max(a_relu) + 1e-12)  # shape=(1, N*N)
		amp_mod = tf.cast(a_mod, dtype=tf.complex64)  # 转成复数，便于相乘
		# nonlinear_flag = True
		# if nonlinear_flag:
		# 	amp_mod = amp_mod + 0.1 * amp_mod ** 3
		# 	phase_mod = phase_mod + 0.1 * phase_mod ** 3
		# 4) 组合得到整层的复数调制
		com_mod = amp_mod * phase_mod  # shape=(1, N*N), 复数

		# 5) 将 com_mod tile到 batch_size 同行数
		batch_size = tf.shape(X)[0]
		com_mod_tiled = tf.tile(com_mod, [batch_size, 1])  # shape=(batch_size, N*N)

		# 6) 执行复合调制
		layer_out = X * com_mod_tiled
		# 7) 衍射传播: 与 w^T 相乘， -> 再做衍射传播
		#    self.w shape=(784,784), transpose -> (784,784)
		#    layer_out shape=(batch_size,784)
		diffrac_result = layer_out @ tf.transpose(self.w)  # (batch_size, row*col), 复数
		# diffrac_result = diffrac_result + 0.1 * diffrac_result ** 3	# 啊啊啊到底光学怎么做非线性啊

		return diffrac_result

	# return tf.matmul(layer_out, tf.transpose(self.w))  # (batch_size, 784), 复数

	def call(self, inputs, training=False, mask=None):
		"""
		前向传播 (5 层)
		inputs: shape=(batch_size, N*N), float32
		输出: shape=(batch_size, N*N), 复数,
		特别注意：`training`和`mask`参数在本模型中未被使用！看起来定义了这两个参数，但只是为了不报keras的模型签名警告
		"""
		# if training is not False or mask is not None:
		# 	print("\t【警告】`training`和`mask`参数在本模型中未被使用！\n"
		# 		  "\t\t类看起来定义了这两个参数，但只是为了不报keras的模型签名警告")
		# 1) 转成复数方便计算，这里只是添加了虚部而已，虚部全为0
		# 看懂了，虚部为0就是相位为0，就是输入仅加载到幅度通道，如果要加载到相位通道或者幅相通道，那么就改输入！！！
		X_complex = tf.cast(inputs, tf.complex64)  # (batch_size, 784)

		# 2) 第一层: 先做一次衍射 w，再进入后续衍射层
		# 这里模拟 "输入层" -> inputs1 = X * w^T
		# 这里其实有点问题，这种做法相当于默认输入层到第一个衍射层的衍射操作和后面每个衍射层都是一样的
		x = X_complex @ tf.transpose(self.w)  # shape=(batch_size,784), complex
		# x2=tf.matmul(X_complex,tf.transpose(self.w))
		# 3) 依次通过 5 个衍射层
		for i in range(self.num_layers):
			x = self.diffraction_layer(x, i)

		# 4) 返回最终输出 (batch_size, 784), complex
		return x

	def get_photodetector_outputs(self, outputsf):
		"""
		根据 outputs_final (复数) 和 pd_positions_all 计算光电探测器输出
		outputs_final: shape=(batch_size, 784), complex
		return: shape=(batch_size, 784), float32
		"""
		# 取绝对值(光场强度)
		abs_out = tf.abs(outputsf)  # (batch_size, 784)
		# tile pd_positions_all
		batch_size = tf.shape(outputsf)[0]
		# 用tf.tile()函数把pd_positions_all复制batch_size份然后竖着拼起来
		all_d_tiled = tf.tile(self.pd_positions_all, [batch_size, 1])  # (batch_size, 784)
		# 与 all_d 相乘,模拟只有有pd的地方才有光强记录
		return abs_out * all_d_tiled  # (batch_size, 784), float32

	def get_logits(self, outputs_final):
		"""
		把最终光场输出转换为可用于分类的10维输出(batch_size,10)
		"""
		# 1) 先与 pd_positions_single 做矩阵乘 => (batch_size, 10)
		abs_out = tf.abs(outputs_final)  # (batch_size, 784)
		# out = tf.matmul(abs_out, tf.transpose(self.pd_positions_single))  # (batch_size, 10)

		# 网络通过看哪一个光强探测器所在位置的总和光强最大，获得输出
		out = abs_out @ tf.transpose(self.pd_positions_single)  # (batch_size, 10)

		# 2) 归一化
		max_per_sample = tf.reduce_max(out, axis=1, keepdims=True)  # (batch_size,1)，找到最大值
		out_norm = tf.square(out / (max_per_sample + 1e-12))  # (batch_size,10)，加一个很小的数防止除以0，平方是为了放大大小的差距
		return out_norm


def fetch_batch(data_x, data_y, epoch, batch_index, batch_size, n_batches):
	"""随机提取"""
	# total_numbers = train_x.shape[0]  # 总样本数
	# n_batches = int(np.ceil(total_numbers / batch_size))  # np.ceil()向上取整
	# 计算每个 epoch 需要迭代的批次数（batches），即 将数据集分成多个 batch，并确保即使数据量不是 batch_size 的整数倍，也不会漏掉数据
	np.random.seed(epoch * n_batches + batch_index)
	indices = np.random.randint(len(data_x), size=batch_size)
	batch_X = data_x[indices, :]
	batch_y = data_y[indices]
	return batch_X, batch_y


def predict(model, x):
	outputsf = model(x)
	logits = model.get_logits(outputsf)
	probs = tf.nn.softmax(logits)  # (batch_size, 10)
	preds = tf.argmax(probs, axis=1, output_type=tf.int32)
	return preds, probs


def compute_loss_old(model, x, y,
					 lambda_amp=0.0,
					 lambda_phase=0.0,
					 add_smooth_reg=False
					 ):
	"""
	输入:
	  model: DONNModel
	  x: shape=(batch_size, M*N), float32
	  y: shape=(batch_size,), int32
	输出:
	  loss_value: 标量, 交叉熵损失
	"""
	# 1) 得到最终输出 (batch_size, 784), complex
	outputsf = model(x)

	# 2) 得到分类logits (batch_size, 10)
	logits = model.get_logits(outputsf)  # out_norm

	# 之前的----------loss函数开始----------
	# 3) 使用 sparse_softmax_cross_entropy_with_logits
	# loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
	# 上面这行等价于(存在浮点数近似问题，但是大体上相等)：
	# loss_ls=[]
	# for i in range(len(logits)):
	# 	# Compute softmax
	# 	exp_logits = np.exp(logits[i] - np.max(logits[i]))  # for numerical stability
	# 	softmax_probs = exp_logits / np.sum(exp_logits)
	# 	# Compute cross-entropy loss for the true class y_0
	# 	loss_temp = -np.log(softmax_probs[y[i]])
	# 	loss_ls.append(loss_temp)
	# loss_custom=np.array(loss_ls)
	# 4) batch 内求平均
	# 之前的----------loss函数结束----------

	# ----------新的loss函数开始----------
	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
	loss = tf.reduce_mean(ce)

	# 4) 加入平滑正则
	if add_smooth_reg and (lambda_amp > 0.0 or lambda_phase > 0.0):
		loss += smoothness_regularizer(model, lambda_amp=lambda_amp, lambda_phase=lambda_phase)

	# ----------新的loss函数结束----------

	return tf.reduce_mean(loss)


# ==== MODIFY: compute_loss ====
def compute_loss(model, x, y,
				 lambda_amp=0.0,
				 lambda_phase=0.0,
				 add_smooth_reg=False,
				 *,
				 phase_mode="linear",
				 phase_loss_type="huber",
				 huber_delta=0.25,
				 use_theta_scale=True,
				 normalize="mean",
				 lambda_box=0.0,
				 box_lo=0.0, box_hi=1.0,
				 return_components=False):
	"""
	add_smooth_reg=True 时加邻接平滑正则；评估/测试时建议关掉。
	return_components=True 方便你打印/记录 CE 与 REG。
	"""
	# 1) 前向
	outputsf = model(x)  # (batch_size, 784), complex
	# 2) 得到 logits
	logits = model.get_logits(outputsf)  # (batch_size, 10)
	# 3) 交叉熵
	ce_vec = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
	ce = tf.reduce_mean(ce_vec)

	# 4) 平滑正则
	reg = tf.constant(0.0, tf.float32)
	if add_smooth_reg and (lambda_amp > 0.0 or lambda_phase > 0.0 or lambda_box > 0.0):
		reg = smoothness_regularizer(
			model,
			lambda_amp=lambda_amp,
			lambda_phase=lambda_phase,
			phase_mode=phase_mode,
			phase_loss_type=phase_loss_type,
			huber_delta=huber_delta,
			use_theta_scale=use_theta_scale,
			normalize=normalize,
			lambda_box=lambda_box,
			box_lo=box_lo, box_hi=box_hi
		)

	loss = ce + reg
	if return_components:
		return loss, ce, reg
	return loss


# ==== END MODIFY ====


def model_save(model, path_save="d2nn_model"):
	"""
	保存训练好的 D2NN 模型
	参数:
		model: 训练好的 D2NNModel 实例
		save_path: 保存路径 (默认为 'd2nn_model%-Y-%m-%d-%H-%M')
	"""
	import time
	time_str = time.strftime("-%Y-%m-%d-%H-%M", time.localtime())
	path_save = path_save + time_str
	model.save(path_save)
	print(f"模型已保存至文件夹: {path_save}")


def model_load(path_model="d2nn_model-2025-02-20-22-49", size=(28, 28), num_layers=5, mod_type="all",
			   # def model_load(path_model="d2nn_model-2025-02-20-22-49", size=(28, 28), num_layers=5,
			   path_pd="./datas_mnist_pds/detector_template_28.txt",
			   wave_length=299792458 / 0.4e12,
			   z=1.05e-2):
	"""
	提供一个便捷的加载模型函数
	该函数尚未完全完成，因此需要手动写入N，num_layers和detector_path
	"""
	pd_csv = pd.read_csv(path_pd, delimiter=',')
	pd_positions_all = pd_csv.iloc[0, :].values.astype('float32')  # 所有探测器的空间分布
	pd_positions_all = pd_positions_all.reshape((1, -1))  # 将一维数组调整为二维数组
	pd_positions_single = pd_csv.iloc[1:, ].values.astype('float32')  # 每个探测器的空间位
	# weights = np.empty([N ** 2, N ** 2], dtype=complex)
	weights = get_diffractive_weights(size=size, z=z, wave_length=wave_length)

	# 1. 重新创建 D2NNModel（确保传入所有初始化参数）
	model_loaded = D2NNModel(size, weights, pd_positions_single, pd_positions_all, num_layers=num_layers,
							 mod_type=mod_type)
	# )
	row, col = size
	# 2. 确保模型初始化（重要！）
	dummy_input = tf.zeros((1, row * col), dtype=tf.float32)
	_ = model_loaded(dummy_input)  # 运行一次前向传播，初始化变量

	# 3. 加载训练好的权重
	path_weight = path_model + "/variables/variables"
	model_loaded.load_weights(path_weight)

	return model_loaded


def get_AP_mod(model, save_flag=False, path_weight=""):
	A_ls = []
	P_ls = []
	for ind in range(model.num_layers):
		c = model.alpha_vars[ind].numpy() * np.exp(1j * 2 * np.pi * model.theta_vars[ind].numpy())  # 计算复数调制
		A, P = np.abs(c), np.angle(c)
		if len(A.shape) == 2 and len(P.shape) == 2:
			A = A.reshape(-1)
			P = P.reshape(-1)
		print(f"第{ind + 1}层振幅调制：{A}\n"
			  f"第{ind + 1}层相位调制：{P}")
		A_ls.append(A)
		P_ls.append(P)
	if save_flag:
		if path_weight == "":
			import time
			current_time = time.strftime("-%Y-%m-%d-%H-%M-%S", time.localtime())
			# np.savetxt("./Amps_Phases/A"+current_time+".csv", A_ls, delimiter=",", fmt="%.5f")
			# weight_path="./Amps_Phases/P" + current_time + ".csv"
			path_weight = "./P" + current_time + ".csv"  # 默认只有相位调制
		import os
		os.makedirs(os.path.dirname(path_weight), exist_ok=True)
		np.savetxt(path_weight, P_ls, delimiter=",", fmt="%.5f")
		print(f"相位权重已保存为.csv文件，路径：{path_weight}")
	else:
		print("未传入save_flag，跳过保存阶段")
	return A_ls, P_ls


# 临近正则化尝试


def _aggregate(pen, H, W, normalize="mean"):
	"""把像素/边的惩罚按指定方式聚合成标量"""
	if normalize == "mean":
		return tf.reduce_mean(pen)
	elif normalize == "sum":
		return tf.reduce_sum(pen)
	elif normalize == "hw":  # mean × 边数，近似与 sum 等价，但可读性更好
		edges = H * (W - 1) + W * (H - 1)
		return tf.reduce_mean(pen) * tf.cast(edges, tf.float32)
	else:
		raise ValueError("normalize must be 'mean' | 'sum' | 'hw'")


def _tv_l2_on_amplitude_map(alpha_var, row, col, normalize="mean"):
	"""对(归一化后的)幅度图做 L2-TV 平滑"""
	a_relu = tf.nn.relu(alpha_var)  # (1, HW)
	a_mod = a_relu / (tf.reduce_max(a_relu) + 1e-12)  # (1, HW) 与前向一致
	A = tf.reshape(a_mod, (row, col))  # (H, W)
	dx = A[:, 1:] - A[:, :-1]
	dy = A[1:, :] - A[:-1, :]
	pen = tf.concat([tf.reshape(dx * dx, [-1]), tf.reshape(dy * dy, [-1])], axis=0)
	return _aggregate(pen, row, col, normalize)


def _huber(d, delta):
	ad = tf.abs(d)
	return tf.where(ad <= delta, 0.5 * ad * ad, delta * (ad - 0.5 * delta))


def _tv_linear_on_phase(theta_norm_var, row, col,
						normalize="mean",
						loss_type="huber",  # "huber" | "l2" | "l1"
						huber_delta=0.25,
						use_theta_scale=True):
	"""
	非周期相位平滑：直接在 t 上做差分（t∈R，对应 θ=2πt）
	- 0 与 1（θ=0 与 2π）差距视为大
	- use_theta_scale=True 相当于乘 (2π)^2，使尺度等价于在 θ 上做 L2
	"""
	T = tf.reshape(theta_norm_var, (row, col))
	dt_x = T[:, 1:] - T[:, :-1]
	dt_y = T[1:, :] - T[:-1, :]

	if loss_type == "huber":
		pen_xy = tf.concat([tf.reshape(_huber(dt_x, huber_delta), [-1]),
							tf.reshape(_huber(dt_y, huber_delta), [-1])], axis=0)
	elif loss_type == "l2":
		pen_xy = tf.concat([tf.reshape(dt_x * dt_x, [-1]), tf.reshape(dt_y * dt_y, [-1])], axis=0)
	elif loss_type == "l1":
		pen_xy = tf.concat([tf.reshape(tf.abs(dt_x), [-1]), tf.reshape(tf.abs(dt_y), [-1])], axis=0)
	else:
		raise ValueError("loss_type must be 'huber' | 'l2' | 'l1'")

	if use_theta_scale:
		pen_xy = pen_xy * (2.0 * math.pi) ** 2

	return _aggregate(pen_xy, row, col, normalize)


def _box_penalty_on_t(theta_norm_var, lo=0.0, hi=1.0):
	"""可选：把 t 约束在 [lo, hi] 附近（越界二次罚）"""
	over = tf.nn.relu(theta_norm_var - hi)
	under = tf.nn.relu(lo - theta_norm_var)
	return tf.reduce_mean(over * over + under * under)


def smoothness_regularizer(model,
						   lambda_amp=0.0,
						   lambda_phase=0.0,
						   *,
						   phase_mode="linear",  # 你当前需要的：非周期
						   phase_loss_type="huber",
						   huber_delta=0.25,
						   use_theta_scale=True,
						   normalize="mean",  # "mean" | "sum" | "hw"
						   lambda_box=0.0,  # >0 时启用盒约束（可选）
						   box_lo=0.0, box_hi=1.0):
	"""
	汇总本模型所有层内的相邻平滑正则。只对 trainable=True 的变量生效。
	"""
	row, col = model.size
	reg = tf.constant(0.0, dtype=tf.float32)

	for i in range(model.num_layers):
		a = model.alpha_vars[i]
		t = model.theta_vars[i]

		if getattr(a, "trainable", False) and lambda_amp > 0.0:
			reg += lambda_amp * _tv_l2_on_amplitude_map(a, row, col, normalize)

		if getattr(t, "trainable", False) and lambda_phase > 0.0:
			# 注：若想切回“周期友好”，可另外实现 periodic 版本，这里默认 linear
			reg += lambda_phase * _tv_linear_on_phase(
				t, row, col,
				normalize=normalize,
				loss_type=phase_loss_type,
				huber_delta=huber_delta,
				use_theta_scale=use_theta_scale
			)

		if getattr(t, "trainable", False) and lambda_box > 0.0:
			reg += lambda_box * _box_penalty_on_t(t, box_lo, box_hi)

	return reg
# 临近正则化尝试结束
