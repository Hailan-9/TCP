from collections import deque
import numpy as np
import torch 
from torch import nn
#from TCP.resnet import *
from resnet import *

class PIDController(object):
	def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
		self._K_P = K_P
		self._K_I = K_I
		self._K_D = K_D
		# deque：来自 Python 的 collections 模块，是一种双端队列的数据结构。与列表相比，deque 支持在两端快速添加和移除元素，效率更高。
		self._window = deque([0 for _ in range(n)], maxlen=n)
		self._max = 0.0
		self._min = 0.0

	def step(self, error):
		self._window.append(error)
		self._max = max(self._max, abs(error))
		self._min = -abs(self._max)

		if len(self._window) >= 2:
			integral = np.mean(self._window)  # 积分项
			derivative = (self._window[-1] - self._window[-2])  # 微分项
		else:
			integral = 0.0
			derivative = 0.0

		return self._K_P * error + self._K_I * integral + self._K_D * derivative
# 继承自 PyTorch 的 nn.Module，使得 TCP 成为一个神经网络模块，能够利用 PyTorch 提供的各种功能。
class TCP(nn.Module):

	def __init__(self, config):
		# 在构造函数中调用父类的初始化方法，以确保 TCP 类能够正确继承 nn.Module 的功能。
		super().__init__()
		self.config = config

		self.turn_controller = PIDController(K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n)  # 横向跟踪（跟踪方向盘转角）
		self.speed_controller = PIDController(K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n)  # 纵向跟踪（跟踪速度）
		# 感知部分 使用已经训练好的resnet34模型提取图像特征
		# 对于成熟的感知部分，我们完全可以使用别人训练好的模型，不要重复造轮子，主要精力放在我们的核心工作上！！！
		# https://zhuanlan.zhihu.com/p/370931435
		# https://blog.csdn.net/a_piece_of_ppx/article/details/125960098
		# 感知模型的输入是图片，输入维度就是相机采集图像的尺寸
		# 输入image 224x224x3 RGB三通道 输入是维度1000的features
		# 输入格式为 (批量大小, 通道数, 高度, 宽度)，通常为：
		# (N, 3, 224, 224)，其中 N 是批量大小，3 是通道数（RGB），224 是高度，224 是宽度。
		self.perception = resnet34(pretrained=True)    # 图像提取特征用的resnet34

		
		# MLP 创建提取measurement的神经网络模块
		self.measurements = nn.Sequential(
							# 这是一个全连接层（线性层），将输入的特征数（1 + 2 + 6）映射到输出的特征数（128）。
							nn.Linear(1+2+6, 128), # 全连接层（线性层）  输入特征的总数  输出特征的总数
							# inplace=True：表示在计算过程中会修改输入数据，而不需要额外的内存来存储中间结果。这可以提高内存效率。
							nn.ReLU(inplace=True),
							nn.Linear(128, 128),
							nn.ReLU(inplace=True),
						)  # 处理measurements的模块
		# 相机图像的features和measurement的特征的融合，参考论文结构图
		# TODO measurement为啥是128维度的输出
		self.join_traj = nn.Sequential(
							nn.Linear(128+1000, 512),  # measurement输出128维度, resnet34输出的feature_emb维度是1000
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)

		self.join_ctrl = nn.Sequential(
							nn.Linear(128+512, 512),  # 新feature_emb：（32, 512）, measurement_feature：（32, 128） cat在一起
							nn.ReLU(inplace=True),
							nn.Linear(512, 512),
							nn.ReLU(inplace=True),
							nn.Linear(512, 256),
							nn.ReLU(inplace=True),
						)

		self.speed_branch = nn.Sequential(
							nn.Linear(1000, 256),
							nn.ReLU(inplace=True),
							nn.Linear(256, 256),
							# Dropout：是一种正则化技术，旨在防止神经网络过拟合。
							# 在训练过程中，Dropout 随机地将一部分神经元的输出设置为零，从而减少模型对特定神经元的依赖，促使网络学习更具鲁棒性的特征。
							# p=0.5 意味着每个通道有 50% 的概率被设置为零
							nn.Dropout2d(p=0.5),
							nn.ReLU(inplace=True),
							nn.Linear(256, 1),
						)  # 预测速度的head

		# 定义路径点预测值的分支
		# trajectory branch
		self.value_branch_traj = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				)  # 这个是接收图像和measurement的融合的特征，然后输出路径点的值

		# 定义控制预测值的分支
		# control branch
		self.value_branch_ctrl = nn.Sequential(
					nn.Linear(256, 256),
					nn.ReLU(inplace=True),
					nn.Linear(256, 256),
					nn.Dropout2d(p=0.5),
					nn.ReLU(inplace=True),
					nn.Linear(256, 1),
				)	# 这个是接收图像和measurement的融合的特征，然后输出控制量的值

		# shared branches_neurons
		dim_out = 2
		# 策略头
		# 整体目的是处理由前面的网络层（例如卷积层或其他特征提取层）生成的特征向量（256 维）。
		# 通过两次全连接层和激活函数的组合，模型可以学习到更复杂的特征表示
		self.policy_head = nn.Sequential(
				# 输入和输出的维度相同，表示该层将特征进行线性变换。
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 256),
				nn.Dropout2d(p=0.5),
				nn.ReLU(inplace=True),
			)
		# GRU（Gated Recurrent Unit）是一种循环神经网络（RNN）单元，用于处理序列数据。与传统的 RNN 相比，GRU 具有更好的记忆能力和更少的参数，适合于长序列的学习。
		# GRUCell 是 GRU 的一种实现，适用于单步时间序列的处理，通常在序列生成或解码任务中使用。！！！
		# 序列生成任务1
		self.decoder_ctrl = nn.GRUCell(input_size=256+4, hidden_size=256)  # 定义控制命令的解码器（GRU单元）
		self.output_ctrl = nn.Sequential(
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 256),
				nn.ReLU(inplace=True),
			)
		
		self.dist_mu = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())  # 输出预测控制动作的分布的均值
		self.dist_sigma = nn.Sequential(nn.Linear(256, dim_out), nn.Softplus())  # 输出预测控制动作的分布的标准差

		# 序列生成任务2
		self.decoder_traj = nn.GRUCell(input_size=4, hidden_size=256)  # 定义路径点解码器（GRU单元）
		self.output_traj = nn.Linear(256, 2)  # 输出下一个点

		self.init_att = nn.Sequential(
				nn.Linear(128, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 29*8),
				# dim=1 表示在第二个维度上进行 Softmax 计算。
				nn.Softmax(1)
			)

		self.wp_att = nn.Sequential(
				nn.Linear(256+256, 256),
				nn.ReLU(inplace=True),
				nn.Linear(256, 29*8),
				nn.Softmax(1)
			)
		# 合并
		self.merge = nn.Sequential(
				nn.Linear(512+256, 512),
				nn.ReLU(inplace=True),
				nn.Linear(512, 256),
			)
		
	# 前向传播
	def forward(self, img, state, target_point):  # 前向传播的时候要传入img, state, target_point这几个参数 image encoder
		# resnet34是基于Resnet，这个模型返回return x, x_layer4，即：通过全连接层的x，和第四层的feature///num_classes: int = 1000,
		# feature_emb维度为（32, 1000）; cnn_feature：（32, 512, 8, 29）
		feature_emb, cnn_feature = self.perception(img)  # perception处理图像 获得feature embedding
		outputs = {}  #创建一个字典用于存放输出的预测结果
		outputs['pred_speed'] = self.speed_branch(feature_emb)  # 预测速度 （32, 1）
		measurement_feature = self.measurements(state)  # 提取state的特征，measurement encoder （32, 128）
		
		# j_traj是拼起来然后融合的特征：（32, 256）
		j_traj = self.join_traj(torch.cat([feature_emb, measurement_feature], 1))  # 把image feature和measurement的feature cat在一起然后送入join_traj层
		#----------------------------------------------------------------------------------------------------------
		outputs['pred_value_traj'] = self.value_branch_traj(j_traj)  # 输出路径点预测值 （32, 1）
		outputs['pred_features_traj'] = j_traj  # pred_features_traj（j_traj拼起来然后融合的特征） （32, 256）
		z = j_traj
		output_wp = list()
		traj_hidden_state = list()

		# 初始化路径生成的GRU输入h，初始化为0
		x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)  # （32, 2） x 的数据类型与张量 z 的数据类型相同 z.shape[0]=32

		# 输出自回归生成的航路点
		for _ in range(self.config.pred_len):  # config.pred_len默认设置为4个点
			x_in = torch.cat([x, target_point], dim=1)
			z = self.decoder_traj(x_in, z)  # 送入GRU网络 两个输入：一个来自过去z，一个是新输入，这个z就是h那个意思 （32， 256）
			traj_hidden_state.append(z)
			dx = self.output_traj(z)  # 线性层输出（32, 2）
			x = dx + x  # 计算出下一个点
			output_wp.append(x)  # 存储下一个点 这个数据格式是list

		pred_wp = torch.stack(output_wp, dim=1)  # 在第一个维度堆叠起来--->（32, 4, 2）
		outputs['pred_wp'] = pred_wp  # 存到字典中

		traj_hidden_state = torch.stack(traj_hidden_state, dim=1)  # （32, 4, 256）
		# TODO 注意力层是啥意思？？
		# measurement提取的特征经过一个注意力层，然后resize尺寸，-1 表示自动推断维度大小; 1 表示在该维度上插入一个单维度; 8 和 29 代表把8×29拆成8, 29
		# init_att： （32, 1, 8, 29）
		init_att = self.init_att(measurement_feature).view(-1, 1, 8, 29)   #
		'''
		init_att 会自动广播（broadcasting）到与 cnn_feature 相同的形状，然后逐元素相乘;
		cnn_feature*init_att： （32, 512, 8, 29）
		然后，分别把8和29这两个维度相加，得到新的feature_emb：（32, 512）
		'''
		# 自动广播是一种机制，允许不同形状的张量在进行元素级运算时自动调整形状。目的是对其形状
		feature_emb = torch.sum(cnn_feature*init_att, dim=(2, 3))  # （32, 512）

		# 把这个新的feature_emb和measurement_feature拼接在一起然后通过join_ctrl融合-->j_ctrl:(32, 256) 动作的预测特征：pred_features_ctrl
		j_ctrl = self.join_ctrl(torch.cat([feature_emb, measurement_feature], 1))
		outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl)  # (32, 1) 动作的预测值
		outputs['pred_features_ctrl'] = j_ctrl
		# TODO 这是干啥的？？
		policy = self.policy_head(j_ctrl)  # 策略头接受预测的动作特征输出策略值 （32, 256）
		outputs['mu_branches'] = self.dist_mu(policy)  # （32, 2）两个动作的均值
		outputs['sigma_branches'] = self.dist_sigma(policy)  # （32, 2）两个动作的标准差

		x = j_ctrl
		mu = outputs['mu_branches']
		sigma = outputs['sigma_branches']

		future_feature, future_mu, future_sigma = [], [], []  # 存放输出结果的数组

		# 初始化动作生成的GRU输入h，初始化为0
		h = torch.zeros(size=(x.shape[0], 256), dtype=x.dtype).type_as(x)

		for _ in range(self.config.pred_len):  # 长度为4
			x_in = torch.cat([x, mu, sigma], dim=1)  # 把动作融合的特征，动作的均值，标准差拼接在一起--->（32, 256）+（32, 2）+ （32, 2）---> （32, 260）
			h = self.decoder_ctrl(x_in, h)  # 送入GRU网络得到新的h输出，shape：（32, 256）
			'''
			traj_hidden_state[:, _]这里的_是for循环中的_你可以理解为i
			意思就是：按照循环中的第几步取第几个256 tensor和h拼接在一起，所以torch.cat([h, traj_hidden_state[:, _]], 1)输出为（32, 256+256）
			然后送入注意力层在resize为（32, 1, 8, 29）
			'''
			wp_att = self.wp_att(torch.cat([h, traj_hidden_state[:, _]], 1)).view(-1, 1, 8, 29)
			'''
			wp_att 会自动广播（broadcasting）到与 cnn_feature 相同的形状，然后逐元素相乘;
			cnn_feature*init_att： （32, 512, 8, 29）
			然后，分别把8和29这两个维度相加，得到new_feature_emb：（32, 512）
			'''
			new_feature_emb = torch.sum(cnn_feature*wp_att, dim=(2, 3))
			# 把h, new_feature_emb拼接载送入融合层 torch.cat([h, new_feature_emb], 1)--->（32, 256）+ （32, 512）----> (32, 256+512)
			merged_feature = self.merge(torch.cat([h, new_feature_emb], 1))
			# 把融合后的特征送入动作输出层
			dx = self.output_ctrl(merged_feature)
			x = dx + x  # 当前的动作预测特征向量+dx

			policy = self.policy_head(x)  # 未来预测特征送入策略头
			mu = self.dist_mu(policy)  # 预测未来动作的均值
			sigma = self.dist_sigma(policy)  # 预测未来动作的标准差
			future_feature.append(x)  # 存进future_feature
			future_mu.append(mu)  # 存进future_mu
			future_sigma.append(sigma)  # 存进future_sigma


		outputs['future_feature'] = future_feature
		outputs['future_mu'] = future_mu
		outputs['future_sigma'] = future_sigma
		return outputs   # 训练脚本使用的函数到这里就结束了

# ------------------------------------------------------------从这里开始都是验证脚本使用到的函数--------------------------------------------------------------------------------
	def process_action(self, pred, command, speed, target_point):  # pred代表预测出来的动作的分布的均值和标准差
		action = self._get_action_beta(pred['mu_branches'].view(1, 2), pred['sigma_branches'].view(1, 2))  # 这里因为是验证的时候，所以不再是（32,2）而是（1, 2），不是batch为32了
		acc, steer = action.cpu().numpy()[0].astype(np.float64)

		'''
		要是加速度是正数，就赋值给油门
		要是加速度是负数，就将绝对值赋值给刹车
		'''
		if acc >= 0.0:
			throttle = acc
			brake = 0.0
		else:
			throttle = 0.0
			brake = np.abs(acc)

		throttle = np.clip(throttle, 0, 1)  # 将油门限制在0～1之间
		steer = np.clip(steer, -1, 1)  # 将方向盘转角限制在-1～1之间
		brake = np.clip(brake, 0, 1)  # 将刹车限制在0～1之间

		metadata = {
			'speed': float(speed.cpu().numpy().astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'command': command,
			'target_point': tuple(target_point[0].data.cpu().numpy().astype(np.float64)),
		}  # 把上面的数据再组成一个字典
		return steer, throttle, brake, metadata

	'''
	_get_action_beta 函数使用了 alpha 和 beta 这两个参数来从 Beta 分布中生成动作值
	'''
	def _get_action_beta(self, alpha, beta):  # 输入：mu， sigma, shape:（1, 2）
		x = torch.zeros_like(alpha)  # 生成一个和alpha一样的尺寸的全为0的x
		x[:, 1] += 0.5  # 将 x 张量中第 2 列（索引为 1）的所有元素都加上 0.5

		'''
		mask是一个布尔类型的数组，例如 mask=[false, false, true, true]
		x[mask]就是在true的位置更新数值
		alpha[mask]就是取位置上为true的数值
		矩阵运算，对应位置加加减减
		'''
		mask1 = (alpha > 1) & (beta > 1)
		x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)

		mask2 = (alpha <= 1) & (beta > 1)
		x[mask2] = 0.0

		mask3 = (alpha > 1) & (beta <= 1)
		x[mask3] = 1.0

		# mean
		mask4 = (alpha <= 1) & (beta <= 1)
		x[mask4] = alpha[mask4]/torch.clamp((alpha[mask4]+beta[mask4]), min=1e-5)  # torch.clamp防止值小于1e-5，避免除以接近0的数字

		x = x * 2 - 1  # 将 x 张量的值从 [0, 1] 的范围映射到 [-1, 1] 的范围。因为车辆的控制参数（如转向角度）通常需要在这个范围内表示

		return x

#------------------------------------------------------pid控制器--------------------------------------------------------------------------
	''' 
	Predicts vehicle control with a PID controller.
	Args:
		waypoints (tensor): output of self.plan()  预测出来的路径点
		velocity (tensor): speedometer input  
	'''
	def control_pid(self, waypoints, velocity, target):
		assert (waypoints.size(0) == 1)  # 检查是否路径中有数据
		waypoints = waypoints[0].data.cpu().numpy()  # 拿第一个路径点
		target = target.squeeze().data.cpu().numpy()  # 拿目标点

		# flip y (forward is negative in our waypoints)
		# 反转y轴，前进方向为y轴负方向
		waypoints[:, 1] *= -1
		target[1] *= -1

		# iterate over vectors between predicted waypoints
		num_pairs = len(waypoints) - 1  # 计算路径点对的数量 ，因为索引从0开始，所以减去1
		best_norm = 1e5  # 初始化一个很大的值 best_norm，用于存储最优的范数
		desired_speed = 0  # 初始化速度
		aim = waypoints[0]  # 初始化转向目标

		'''
		for函数目的是从预测的路经点中选择一个最合适的路径点作为车辆的转向目标（预瞄点）
		具体来说，代码通过比较每对路径点中点与目标距离的差距，选出一个最接近理想距离的路径点作为车辆的转向目标
		这有助于车辆在路径上保持平稳行驶，而不会因为选择不合适的路径点作为目标而导致过度转向或路径偏离
		'''
		for i in range(num_pairs):
			# magnitude of vectors, used for speed
			# 逐渐加速（速度增量和路径点之间成比例关系）
			desired_speed += np.linalg.norm(
					waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

			# norm of vector midpoints, used for steering
			# 计算路径点对中点的范数，用于确定最佳转向目标。
			norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
			# 比较当前的范数 norm 与先前的最佳范数 best_norm，如果当前更接近理想值（self.config.aim_dist），更新 aim 和 best_norm
			if abs(self.config.aim_dist-best_norm) > abs(self.config.aim_dist-norm):  # self.config.aim_dist（default为4.0）
				aim = waypoints[i]  # 还是从waypoints[0]开始
				best_norm = norm

		aim_last = waypoints[-1] - waypoints[-2]  # 计算最后一个路径点与倒数第二个路径点的向量差值。

		'''
		aim[1], aim[0]代表下一个路点的y，x坐标
		以车辆朝向为y轴pi/2，计算从车辆位置到目标点 aim 的转向角度
		/90: 将弧度转化为角度
		'''
		angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90

		'''
		目的: 评估当前路径段的转向趋势, 如果这个角度变化较大，则意味着车辆正在进入或已经处于一个转弯的过程中
		'''
		angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90

		'''
		计算 angle_target 是为了确定车辆相对于目标点的转向角度。这对于控制车辆的方向非常重要，确保车辆能够朝向目标点正确行驶，从而实现平滑、准确的路径跟踪
		'''
		angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

		# choice of point to aim for steering, removing outlier predictions
		# use target point if it has a smaller angle or if error is large
		# predicted point otherwise
		# (reduces noise in eg. straight roads, helps with sudden turn commands)
		'''
		这个就是判断是瞄着最终目标跑还是瞄着预测路径点中最好的那个点跑，
		确保转向动作平滑不过大
		'''
		use_target_to_aim = np.abs(angle_target) < np.abs(angle)
		use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.config.angle_thresh and target[1] < self.config.dist_thresh)
		if use_target_to_aim:
			angle_final = angle_target
		else:
			angle_final = angle

		steer = self.turn_controller.step(angle_final)  # 调用pid控制器，传入转角error
		steer = np.clip(steer, -1.0, 1.0)  # 防止转角超限，限定在（-1,1）之间

		speed = velocity[0].data.cpu().numpy()
		brake = desired_speed < self.config.brake_speed or (speed / desired_speed) > self.config.brake_ratio  # 判断是否需要刹车（车速大了就要刹车呗）

		delta = np.clip(desired_speed - speed, 0.0, self.config.clip_delta)  # 计算speed error，还做了限制上下界
		throttle = self.speed_controller.step(delta)  # 调用pid控制器，传入速度error
		throttle = np.clip(throttle, 0.0, self.config.max_throttle)  # 再对油门限制一下
		throttle = throttle if not brake else 0.0  # 判断是否是刹车，要是不是刹车，那就是赋值油门值，否则油门为0

		metadata = {
			'speed': float(speed.astype(np.float64)),
			'steer': float(steer),
			'throttle': float(throttle),
			'brake': float(brake),
			'wp_4': tuple(waypoints[3].astype(np.float64)),
			'wp_3': tuple(waypoints[2].astype(np.float64)),
			'wp_2': tuple(waypoints[1].astype(np.float64)),
			'wp_1': tuple(waypoints[0].astype(np.float64)),
			'aim': tuple(aim.astype(np.float64)),
			'target': tuple(target.astype(np.float64)),
			'desired_speed': float(desired_speed.astype(np.float64)),
			'angle': float(angle.astype(np.float64)),
			'angle_last': float(angle_last.astype(np.float64)),
			'angle_target': float(angle_target.astype(np.float64)),
			'angle_final': float(angle_final.astype(np.float64)),
			'delta': float(delta.astype(np.float64)),
		}  # 把数据存到字典里

		return steer, throttle, brake, metadata  # 返回控制参数


	def get_action(self, mu, sigma):
		action = self._get_action_beta(mu.view(1, 2), sigma.view(1, 2))
		acc, steer = action[:, 0], action[:, 1]
		if acc >= 0.0:
			throttle = acc
			brake = torch.zeros_like(acc)
		else:
			throttle = torch.zeros_like(acc)
			brake = torch.abs(acc)

		throttle = torch.clamp(throttle, 0, 1)
		steer = torch.clamp(steer, -1, 1)
		brake = torch.clamp(brake, 0, 1)

		return throttle, steer, brake