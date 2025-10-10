import matplotlib.pyplot as plt
import numpy as np
import math


# def plot_initialize_old(paper_font=False):
# 	"""避免GUI交互问题和中文不显示的问题"""
# 	import matplotlib
# 	matplotlib.use('TkAgg')  # 避免 GUI 交互问题
# 	# 设置支持中文的字体，并根据是否论文需要修改中文为宋体，英文为times new roman
# 	if paper_font is False:
# 		plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体
# 	else:
# 		plt.rcParams['font.family'] = ['SimSun', 'Times New Roman']
# 	plt.rcParams['axes.unicode_minus'] = False  # 解决负号 "-" 显示为方块的问题

def plot_initialize(paper_font=False, base_fontsize=14):
	"""避免 GUI 交互问题、中文显示，并统一字号"""
	import matplotlib
	matplotlib.use('TkAgg')

	# ------- 字体族（中英） -------
	if paper_font is False:
		plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 黑体
	else:
		plt.rcParams['font.family'] = ['SimSun', 'Times New Roman']

	plt.rcParams['axes.unicode_minus'] = False  # 负号正常显示

	# ---------- 统一字号 ----------
	plt.rcParams.update({
		'font.size': base_fontsize,
		'axes.titlesize': base_fontsize + 2,
		'axes.labelsize': base_fontsize,
		'xtick.labelsize': base_fontsize - 2,
		'ytick.labelsize': base_fontsize - 2,
		'legend.fontsize': base_fontsize - 2,
	})


def plot_single_node(weights, size=(28, 28), index=None):
	"""选择中心的一个点，看单位输入的衍射情况"""
	row, col = size
	if row == col:
		N = row  # 这里定义太复杂了，先默认row和col一样大
	else:
		N = row  # 这里定义太复杂了，先默认row和col一样大
	# import matplotlib
	# matplotlib.use('TkAgg')  # 避免 GUI 交互问题
	plt.figure(figsize=(8, 12))
	if not index:
		if N % 2 == 0:  # N为偶数
			w_central = weights[:, int(N * N / 2 + N / 2)].reshape((N, N))  # N为偶数无法选取到最中心那一块，就取中心那一块的右下角那个数吧
		else:  # N为奇数
			w_central = weights[:, int((N * N - 1) / 2)].reshape((N, N))  # python索引从0开始，所以中心那个数的索引是(N-1)/2*N+(N-1)/2
	# 这里取的w_central作为当输入为1的时候的衍射情况，由于输入为1，所以直接忽略了
	else:
		w_central = weights[:, int(index)].reshape((N, N))  # N为偶数无法选取到最中心那一块，就取中心那一块的右下角那个数吧
	plt.subplot(2, 1, 1)
	plt.imshow(np.abs(w_central) / np.max(np.abs(w_central)))
	plt.colorbar()
	plt.clim(0, 1)
	plt.title('归一化幅值')

	plt.subplot(2, 1, 2)
	plt.imshow(np.angle(w_central), cmap='jet')
	plt.colorbar()
	plt.clim(0, math.pi)
	plt.title('相位')

	plt.show()


def plot_diffractive_result(weights, train_x, index=1, size=(28, 28)):
	"""画出输入图像经过衍射的输出衍射结果"""
	row, col = size
	# import matplotlib
	# matplotlib.use('TkAgg')  # 避免 GUI 交互问题
	plt.figure()
	test_input = train_x[index, :]
	plt.imshow(test_input.reshape((row, col)))
	plt.title('输入数字图像')
	plt.show()
	plt.figure()
	# test_output = np.matmul(test_input, np.transpose(weights))
	test_output = test_input @ np.transpose(weights)
	print(test_output.shape)
	plt.imshow(np.abs(test_output.reshape((row, col), order='F')))
	plt.title('经过衍射输出的数字图像')
	plt.show()


def plot_pd(pd_positions, size=(28, 28)):
	"""画出pd位置"""
	row, col = size
	# print(pd.shape)
	# import matplotlib
	# matplotlib.use('TkAgg')  # 避免 GUI 交互问题
	plt.figure()
	plt.imshow(pd_positions.reshape((row, col)))
	plt.colorbar()
	plt.title('光电器探测分布')
	plt.show()


def plot_confusion_matrix_tf2(model, x_test, y_test, normalize=False, title="混淆矩阵",
							  cmap='Blues'):
	"""
	计算并绘制混淆矩阵（适用于 TensorFlow 2.x）

	参数:
		model: 训练好的 D2NNModel
		x_test: 测试数据 (batch_size, 784)
		y_test: 测试标签 (batch_size,)
		classes: 类别标签列表（默认 0-9）
		normalize: 是否归一化混淆矩阵
		title: 图像标题
		cmap: 颜色映射（默认蓝色调）

	返回:
		None (绘制混淆矩阵)
	"""

	from sklearn.metrics import confusion_matrix
	import itertools
	from diffrann_diffrann import predict
	if cmap == "Blues":
		import matplotlib.cm as cm
		cmap = cm.get_cmap('Blues')
	# cmap=plt.cm.Blues):
	# 1) 获取模型预测
	preds, _ = predict(model, x_test)  # 调用你的 `predict` 函数

	# 2) 计算混淆矩阵
	cm = confusion_matrix(y_test, preds.numpy())

	# 3) 归一化处理（可选）
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

	# 4) 绘制混淆矩阵
	plt.figure(figsize=(8, 6))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	classes = range(len(np.unique(y_test)))
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	# 5) 在方块中显示数值（避免颜色混淆）
	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2.  # 动态阈值
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, format(cm[i, j], fmt),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	# 6) 设置轴标签
	plt.ylabel("真值")
	plt.xlabel("预测值")
	plt.tight_layout()
	plt.show()


def show_result(loaded_model, train_X, train_y, num_plot=2, batch_size=10, size=(28, 28), shuffle_mode=True,
				title1="输入数字", title2="输出图样", title3="光电探测器"):
	"""随机选一个 batch 进行演示，并按 num_plot 行 × 3 列 的方式绘图"""
	# 这里有个小bug，似乎num_plot确定后，每次运行程序，循环第一次的图都是一样的
	row, col = size
	import tensorflow as tf
	# def fetch_batch(data_x, data_y, epoch, batch_index, batch_size):
	# 	total_numbers = train_X.shape[0]  # 总样本数
	# 	n_batches = int(np.ceil(total_numbers / batch_size))  # np.ceil()向上取整
	# 	np.random.seed(epoch * n_batches + batch_index)
	# 	indices = np.random.randint(len(data_x), size=batch_size)
	# 	batch_X = data_x[indices, :]
	# 	batch_y = data_y[indices]
	# 	return batch_X, batch_y
	def fetch_batch(data_x, data_y, epoch, batch_index, batch_size, shuffle_mode=True):
		total_numbers = data_x.shape[0]  # 总样本数
		n_batches = int(np.ceil(total_numbers / batch_size))  # 计算总批次数

		if shuffle_mode:
			np.random.seed(epoch * n_batches + batch_index)  # 保证同一 epoch 内的随机性
			indices = np.random.randint(len(data_x), size=batch_size)
		else:
			# 按顺序获取数据
			start_idx = batch_index * batch_size
			end_idx = min(start_idx + batch_size, total_numbers)
			indices = np.arange(start_idx, end_idx)  # 生成顺序索引

		batch_X = data_x[indices, :]
		batch_y = data_y[indices]

		return batch_X, batch_y

	# 计算动态 figsize，确保小尺寸的图片不会过小
	# fig_width = max(10, col)  # 确保最小宽度
	# fig_height = max(10, row)  # 确保最小高度
	fig_width = col
	fig_height = row
	fig, axes = plt.subplots(num_plot, 3, figsize=(9, 3 * num_plot))  # num_plot 行, 3 列
	# fig, axes = plt.subplots(num_plot, 3, figsize=(9, 3))  # num_plot 行, 3 列
	# fig, axes = plt.subplots(num_plot, 3, figsize=(fig_width, fig_height))  # num_plot 行, 3 列
	for i in range(num_plot):
		# 每次循环随机生成一个新的 seed
		seed = np.random.randint(0, 100)
		# print(f"Row {i + 1}: seed={seed}")
		batch_X, batch_y = fetch_batch(train_X, train_y, seed, 0, batch_size, shuffle_mode=shuffle_mode)

		# 前向传播
		outputsf = loaded_model(batch_X)
		outputs_mask = loaded_model.get_photodetector_outputs(outputsf)  # (batch_size, 784)

		# 绘制三张图
		axes[i, 0].imshow(batch_X.reshape(batch_size, row, col)[0], cmap='jet')
		axes[i, 0].set_title(title1)
		axes[i, 0].axis('off')

		axes[i, 1].imshow(tf.abs(outputsf).numpy().reshape(batch_size, row, col)[0], cmap='inferno')
		axes[i, 1].set_title(title2)
		axes[i, 1].axis('off')

		axes[i, 2].imshow(outputs_mask.numpy().reshape(batch_size, row, col)[0], cmap='jet')
		axes[i, 2].set_title(title3)
		axes[i, 2].axis('off')

	plt.tight_layout()
	# plt.subplots_adjust(wspace=10, hspace=0.001)  # 减少子图间距
	plt.show()


def fetch_batch_generator(data_x, data_y, batch_size, shuffle_mode=True):
	"""生成一个数据批次的迭代器，每次调用 next(generator) 获取下一个 batch"""
	total_numbers = data_x.shape[0]  # 总样本数
	n_batches = int(np.ceil(total_numbers / batch_size))  # 计算总批次数
	indices = np.arange(total_numbers)  # 生成索引

	if shuffle_mode:
		np.random.shuffle(indices)  # 仅在 shuffle_mode=True 时打乱数据

	batch_index = 0  # 记录当前批次索引

	while True:  # 让迭代器无限循环，直到外部终止
		start_idx = batch_index * batch_size
		end_idx = min(start_idx + batch_size, total_numbers)

		if start_idx >= total_numbers:  # 如果超出数据范围，则从头开始
			batch_index = 0
			start_idx = 0
			end_idx = min(batch_size, total_numbers)
			if shuffle_mode:
				np.random.shuffle(indices)  # 重新打乱数据

		batch_indices = indices[start_idx:end_idx]
		batch_X = data_x[batch_indices, :]
		batch_y = data_y[batch_indices]

		batch_index += 1  # 递增批次索引
		yield batch_X, batch_y  # 通过 yield 返回数据，支持迭代


def fetch_batch_generator2(data_x, data_y, batch_size, shuffle_mode=True):
	total_numbers = data_x.shape[0]
	indices = np.arange(total_numbers)

	while True:  # 无限循环
		if shuffle_mode:
			np.random.shuffle(indices)

		for start_idx in range(0, total_numbers, batch_size):
			end_idx = min(start_idx + batch_size, total_numbers)
			batch_indices = indices[start_idx:end_idx]
			yield data_x[batch_indices], data_y[batch_indices]


def sample_generator(data_X, data_y, shuffle_mode=True):
	total_numbers = data_X.shape[0]
	indices = np.arange(total_numbers)

	while True:  # 无限循环
		if shuffle_mode:
			np.random.shuffle(indices)

		for idx in indices:
			yield data_X[idx], data_y[idx]


def show_result_generator(loaded_model, batch_generator, num_plot=2, size=(28, 28),
						  title1="该参数不起作用", title2="输出图样", title3="光电探测器", aspect_ratio=7, each_pix=3):
	"""使用生成器逐批获取数据，并按 num_plot 行 × 3 列 的方式绘图"""
	import tensorflow as tf
	# import datas as lupy_datas
	row, col = size
	fig, axes = plt.subplots(num_plot, 3, figsize=(9, 3 * num_plot))  # num_plot 行, 3 列

	for i in range(num_plot):
		sample_X, sample_y = next(batch_generator)  # **从迭代器获取下一批数据**
		batch_X = sample_X[np.newaxis, :]
		# batch_y = np.array([sample_y])[np.newaxis, :]

		# 前向传播
		outputsf = loaded_model(batch_X)
		outputs_mask = loaded_model.get_photodetector_outputs(outputsf)

		# 绘制三张图
		import lumerpy as lupy
		# title1 = lupy_datas.recover_original(sample_X,repeat=each_pix)
		title1 = lupy.data_process.recover_original(sample_X, repeat=each_pix)
		axes[i, 0].imshow(batch_X.reshape(batch_X.shape[0], row, col)[0], cmap='jet', aspect=aspect_ratio)
		axes[i, 0].set_title(title1)
		axes[i, 0].axis('off')

		axes[i, 1].imshow(tf.abs(outputsf).numpy().reshape(batch_X.shape[0], row, col)[0], cmap='inferno',
						  aspect=aspect_ratio)
		axes[i, 1].set_title(title2)
		axes[i, 1].axis('off')

		axes[i, 2].imshow(outputs_mask.numpy().reshape(batch_X.shape[0], row, col)[0], cmap='jet', aspect=aspect_ratio)
		axes[i, 2].set_title(title3)
		axes[i, 2].axis('off')


def plot_amp_and_phase(model, size=(28, 28)):
	"""
	绘制五层衍射层的振幅和相位分布
	model: 训练好的 D2NNModel
	N: 图像尺寸（默认28x28）
	"""
	row, col = size

	def amp_phase(c):
		return np.abs(c), np.angle(c)

	fig, ax = plt.subplots(model.num_layers, 2, figsize=(12, 22))  # 增加高度，防止标题重叠
	# 这里norws应该要改成model.num_layers，但是好像改了就报错，先不管了
	# 获取模型中的相位调制层参数
	for ind in range(model.num_layers):
		c = model.alpha_vars[ind].numpy() * np.exp(1j * 2 * np.pi * model.theta_vars[ind].numpy())  # 计算复数调制
		A, P = amp_phase(c)

		# im1 = ax[ind, 0].imshow(A.reshape((N, N)), cmap='inferno', aspect='equal')
		if len(ax.shape) == 1:  # 避免只有1层网络时绘图报错
			ax = ax.reshape(1, -1)
		im1 = ax[ind, 0].imshow(A.reshape((row, col)), cmap='viridis', aspect='equal')
		fig.colorbar(im1, ax=ax[ind, 0], fraction=0.046, pad=0.04)
		ax[ind, 0].set_title(f'第{ind + 1}层-幅值分布', fontsize=14, pad=10)  # 增加 pad 避免重叠

		# im2 = ax[ind, 1].imshow(P.reshape((N, N)), cmap='jet', aspect='equal')
		im2 = ax[ind, 1].imshow(P.reshape((row, col)), cmap='viridis', aspect='equal')
		fig.colorbar(im2, ax=ax[ind, 1], fraction=0.046, pad=0.04)
		ax[ind, 1].set_title(f'第{ind + 1}层-相位分布', fontsize=14, pad=10)  # 增加 pad 避免重叠

	plt.subplots_adjust(hspace=0.4)  # 增加子图间距，防止标题重叠
	# plt.tight_layout()
	plt.show()


def get_accuracy_and_plot_misclassified(model, x_test, y_test, plot_flag=True, num_errors=16, size=(28, 28)):
	"""
	显示预测错误的样本，并标注预测类别和真实类别，返回分类精度

	参数:
		model: 训练好的 D2NNModel
		x_test: 测试数据 (batch_size, 784)
		y_test: 测试标签 (batch_size,)
		num_errors: 需要展示的错误样本数量（默认16）
		N: 图像尺寸（默认28x28）
	"""
	row, col = size
	from diffrann_diffrann import predict
	# 1) 获取模型预测结果
	preds, probs = predict(model, x_test)  # 获取预测类别和 softmax 概率
	preds = preds.numpy()  # 转换为 NumPy 数组
	probs = probs.numpy()

	# 2) 找出错误分类的索引
	errors = preds != y_test
	x_errors = x_test[errors]  # 预测错误的样本
	y_errors = y_test[errors]  # 真实标签
	y_pred_errors = preds[errors]  # 预测标签
	probs_errors = probs[errors]  # 错误样本的 softmax 概率

	# 3) 计算错误样本的置信度差距
	pred_confidence = np.max(probs_errors, axis=1)  # 预测类别的最大置信度
	true_confidence = np.array([probs_errors[i, y_errors[i]] for i in range(len(y_errors))])  # 真实类别的置信度
	delta_confidence = pred_confidence - true_confidence  # 计算置信度差值

	# 4) 选择最严重的 `num_errors` 个错误
	sorted_errors = np.argsort(delta_confidence)[-num_errors:]  # 置信度差值最大的错误索引
	x_display = x_errors[sorted_errors]
	y_display = y_errors[sorted_errors]
	y_pred_display = y_pred_errors[sorted_errors]
	if plot_flag == True and len(y_errors) == 0:
		print("所有分类结果完全正确，跳过分类错误绘制")
		plot_flag = False
	if plot_flag == True:
		# 5) 可视化错误分类
		nrows = int(np.sqrt(num_errors))  # 计算行数
		ncols = int(np.ceil(num_errors / nrows))  # 计算列数
		fig, axes = plt.subplots(nrows, ncols, figsize=(12, 12))

		for i, ax in enumerate(axes.flat):  # 遍历子图
			if i < len(x_display):
				ax.imshow(x_display[i].reshape(row, col), cmap='jet')
				ax.set_title(f"预测: {y_pred_display[i]}\n真值: {y_display[i]}", fontsize=10)
				ax.axis('off')
			else:
				ax.axis('off')  # 关闭多余的子图

		plt.tight_layout()
		plt.show()

	accuracy = 1 - np.count_nonzero(errors) / y_test.shape[0]
	return accuracy


def plot_multiple_data(datas, titles=None, logic_function=None, data_single_scale=(28, 28), cols=4, aspect_ratio=7):
	num_samples = datas.shape[0]  # 获取数据条数
	rows = int(np.ceil(num_samples / cols))  # 计算行数

	fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))  # 创建子图
	fig.suptitle(logic_function)  # 总标题

	# 统一 `axes` 形状
	if rows == 1 or cols == 1:
		axes = np.atleast_2d(axes)  # 确保 `axes` 是二维数组
		if cols == 1:  # 如果只有 1 列，转置 `axes`
			axes = axes.T

	axes = axes.reshape(rows, cols)  # 确保 `axes` 是 (rows, cols) 形状

	# 遍历数据并绘制
	for i in range(num_samples):
		row, col = divmod(i, cols)  # 计算当前图像的行列位置
		ax = axes[row, col]  # 获取对应子图
		ax.imshow(datas[i].reshape(data_single_scale), cmap="jet", aspect=aspect_ratio)  # 显示图像
		ax.axis("off")  # 关闭坐标轴

		# 设置标题（使用提供的 titles，否则默认 "Sample {i}"）
		if np.all(np.isin(np.unique(titles), [0.3, 0.6, 0.9])):
			mapping_dict = {0.3: "o", 0.6: "x", 0.9: "e"}
			# titles=
			titles = np.vectorize(mapping_dict.get)(titles)
		title = titles[i] if titles is not None else f"PD {i}"
		ax.set_title(title)

	# 如果子图数量大于数据数量，关闭多余的子图
	for i in range(num_samples, rows * cols):
		row, col = divmod(i, cols)
		fig.delaxes(axes[row, col])  # 删除多余的子图

	plt.tight_layout(rect=[0, 0, 1, 0.95])  # 调整布局，避免标题被遮挡
	plt.show()


def plot_two_series_dual_axis_old(y1, y2, label1="y1", label2="y2", xlabel="Index", y1label="Value1", y2label="Value2",
								  title="Dual-Axis Line Plot of y1 and y2",
								  color_y1=(45 / 255, 60 / 255, 129 / 255), color_y2=(147 / 255, 117 / 255, 98 / 255)
								  , y2percent_flag=False):
	"""
	绘制两个数组的折线图，并在同一张图上使用双坐标轴以适应不同量级的数据。

	参数:
	y1 -- 第一个数据序列 (列表或数组)
	y2 -- 第二个数据序列 (列表或数组)
	"""
	# color_y1=(132/255,109/255,177/255),color_y2=(220/255,186/255,151/255),color_y3=(69/255,54/255,103/255)
	# Furina：color_y1=(45/255,60/255,129/255),color_y2=(147/255,117/255,98/255),color_y3=(78/255,164/255,239/255),color_y4=(20/255,21/255,33/255),color_y5=(78/255,164/255,239/255)
	if len(y1) != len(y2):
		raise ValueError("y1 和 y2 的长度必须相等")

	x = list(range(len(y1)))  # x轴索引

	fig, ax1 = plt.subplots(figsize=(8, 5))

	# 左侧坐标轴绘制 y1
	ax1.plot(x, y1, marker='o', linestyle='-', color=color_y1, label=label1)
	ax1.set_xlabel(xlabel)
	ax1.set_ylabel(y1label, color=color_y1, rotation=90)
	ax1.tick_params(axis='y', labelcolor=color_y1)
	ax1.grid(True)

	# 右侧坐标轴绘制 y2
	ax2 = ax1.twinx()
	# y2 = y2 * 100  # 转换成分数
	if y2percent_flag:
		y2 = [item * 100 for item in y2]  # y2的每个值乘100
		ax2.set_ylim(0, 100)  # 分类精度喜欢用百分数嘛
	ax2.plot(x, y2, marker='s', linestyle='-', color=color_y2, label=label2)
	ax2.set_ylabel(y2label, color=color_y2, rotation=90)
	ax2.tick_params(axis='y', labelcolor=color_y2)

	# ax1.legend(loc="best")  # 左侧 y 轴的图例
	# ax2.legend(loc="best")  # 右侧 y 轴的图例
	plt.title(title)
	fig.tight_layout()
	plt.show()


def plot_two_series_dual_axis(y1, y2, label1="y1", label2="y2", xlabel="Index", y1label="Value1", y2label="Value2",
							  title="Dual-Axis Line Plot of y1 and y2",
							  color_y1=(45 / 255, 60 / 255, 129 / 255), color_y2=(147 / 255, 117 / 255, 98 / 255),
							  y1percent_flag=False, y2percent_flag=False, marker_flag=True):
	"""
	绘制两个数组的折线图，并在同一张图上使用双坐标轴以适应不同量级的数据。

	参数:
	y1 -- 第一个数据序列 (列表或数组)
	y2 -- 第二个数据序列 (列表或数组)
	"""
	if len(y1) != len(y2):
		raise ValueError("y1 和 y2 的长度必须相等")

	x = list(range(len(y1)))  # x轴索引

	fig, ax1 = plt.subplots(figsize=(8, 5))

	# 左侧坐标轴绘制 y1
	if marker_flag:
		marker1 = 'o'
		marker2 = 'o'
	else:
		marker1 = ""
		marker2 = ""
	if y1percent_flag:
		y1 = [item * 100 for item in y1]
		ax1.set_ylim(0, 105)  # 控制 y1 轴范围
	line1, = ax1.plot(x, y1, marker=marker1, linestyle='-', color=color_y1, label=label1, linewidth=2)
	ax1.set_xlabel(xlabel)
	ax1.set_ylabel(y1label, color=color_y1, rotation=90)
	ax1.tick_params(axis='y', labelcolor=color_y1)
	ax1.grid(True)

	# 右侧坐标轴绘制 y2
	ax2 = ax1.twinx()
	if y2percent_flag:
		y2 = [item * 100 for item in y2]
		ax2.set_ylim(0, 105)  # 控制 y2 轴范围
	line2, = ax2.plot(x, y2, marker=marker2, linestyle='--', color=color_y2, label=label2, linewidth=2)
	ax2.set_ylabel(y2label, color=color_y2, rotation=90)
	ax2.tick_params(axis='y', labelcolor=color_y2)

	# 在图的右侧添加图例
	ax2.legend(handles=[line1, line2], loc="right", fontsize=ax1.yaxis.get_ticklabels()[0].get_size(), frameon=True)
	plt.title(title)
	fig.tight_layout()
	plt.show()


def plot_func(channels, func):
	"""
	根据给定位数 channels，枚举所有 2**channels 个输入，
	调用外部函数 func 计算输出，并用 matplotlib 绘制图像。
	示例用法：plot_func(channels=4,func=ReLU_shift)
	"""
	import matplotlib.pyplot as plt
	# 我需要一个函数，这个函数接受两个输入，位数channels和函数func，这个函数的效果是调用matplotlib画出对于给定位数情况下，所有可能的2**channels个数作为函数的输入（例如，当channels=3时，可能的输入为[0,1,2,3,4,5,6,7]，也就是2**3-1），然后调用外在函数，获得对应的输出y。之后，通过matplotlib画出他们的图像。

	# 生成所有可能的输入
	x_values = list(range(2 ** channels))

	# 调用函数获得输出
	y_values = [func(x, channels) for x in x_values]

	# 绘制图像
	plt.figure(figsize=(6, 4))
	plt.plot(x_values, y_values, marker='o', linestyle='-')
	plt.title(f"Function Plot for {channels} Channels")
	plt.xlabel("Input (0 to 2**channels - 1)")
	plt.ylabel("Output")
	plt.grid(True)
	plt.show()
