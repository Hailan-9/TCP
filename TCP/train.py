import argparse
import os
from collections import OrderedDict # 有序字典

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.distributions import Beta


import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

from model import TCP
from data import CARLA_Data
from config import GlobalConfig

# Python中类的继承 LightningModule 是 PyTorch Lightning 提供的一个基类，简化了模型训练、验证和测试的流程。
class TCP_planner(pl.LightningModule):
	def __init__(self, config, lr):
		super().__init__()
		self.lr = lr # 学习率
		self.config = config
		self.model = TCP(config)  # 实例化一个模型，即TCP model
		self._load_weight()
	# 加载预训练的模型参数
	def _load_weight(self):
		# 强化学习得到的状态-动作对数据；字典类型
		# 用于从磁盘加载一个保存的模型、张量或其他数据。
		# 此处应该是加载模型 加载的对象是一个字典，其中包含多个键值对，这里通过 ['policy_state_dict'] 访问字典中的特定键
		rl_state_dict = torch.load(self.config.rl_ckpt, map_location='cpu')['policy_state_dict']
		self._load_state_dict(self.model.value_branch_traj, rl_state_dict, 'value_head')
		self._load_state_dict(self.model.value_branch_ctrl, rl_state_dict, 'value_head')
		self._load_state_dict(self.model.dist_mu, rl_state_dict, 'dist_mu')
		self._load_state_dict(self.model.dist_sigma, rl_state_dict, 'dist_sigma')
	# 加载状态字典 参数分别为模型学习网络 强化学习状态字典、key_word
	def _load_state_dict(self, il_net, rl_state_dict, key_word):
		# 从 rl_state_dict 字典中提取所有包含 key_word 的键，并将这些键存储在列表 rl_keys 中
		rl_keys = [k for k in rl_state_dict.keys() if key_word in k]
		il_keys = il_net.state_dict().keys()
		# 断言 如果不相等，程序将抛出一个 AssertionError，并显示错误消息
		assert len(rl_keys) == len(il_net.state_dict().keys()), f'mismatch number of layers loading {key_word}'
		new_state_dict = OrderedDict() # 创建一个有序字典
		for k_il, k_rl in zip(il_keys, rl_keys): # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
			new_state_dict[k_il] = rl_state_dict[k_rl] # 将 rl_state_dict 中的键值对复制到 new_state_dict 中
		il_net.load_state_dict(new_state_dict)
	
	def forward(self, batch):
		# forward 方法用于定义模型的前向传播逻辑。即，输入数据通过模型时的计算过程。
		pass

	# 这块就是计算训练loss的
	# batch：通常是一个数据批次，包含模型的输入数据和相应的标签。在训练过程中，batch 通常来自于数据加载器（DataLoader），它将数据分成小批次以便于处理。
	def training_step(self, batch, batch_idx):
		front_img = batch['front_img']  # shape: [32, 3, 256, 900]
		# view(-1, 1)：这个方法用于重新调整张量的形状（reshape）
		# -1 表示该维度的大小将自动推断，以使总元素数量保持不变。
		# 1 表示将张量的第二维设置为 1。
		speed = batch['speed'].to(dtype=torch.float32).view(-1, 1) / 12.  # 除以 12 可以将速度归一化到 [0, 1] 的范围内，另外shape：(32,1)
		target_point = batch['target_point'].to(dtype=torch.float32)  # shape:(32,2)
		command = batch['target_command']  # command是one-hot编码; shape:[32,6] 32个[0,1,0,0,0,0]这种
		# 1 表示沿着第二维进行拼接 也就是列方向
		state = torch.cat([speed, target_point, command], 1)  # shape:(32,9) 9=1+2+6
		value = batch['value'].view(-1, 1)  # shape:(32,1)
		feature = batch['feature']  # shape:(32,256)
		# 从一个名为 batch 的数据字典中提取与“目标路径点”（ground truth waypoints）相关的数据
		# gt通常表示ground truth，表示真实值，用于计算损失。也就是真实的、参考的或目标的值
		gt_waypoints = batch['waypoints']  # shape:(32,4,2) gt标签
		# 预测
		# 这部分代码调用模型的前向传播方法，将 front_img、state 和 target_point 作为输入。
		# 模型将根据这些输入计算输出，通常是预测值，例如动作、分类结果或其他任务相关的输出。
		pred = self.model(front_img, state, target_point)  # 把参数丢进模型进行预测

		'''
		# 32是batch_size
		model输出是一个字典（pred是一个大小为11的字典）：
			'pred_speed'  (32, 1)
			'pred_value_traj' (32, 1)
			'pred_features_traj' (32, 256)
			'pred_wp' (32, 4, 2) --->预测的路径
			'pred_value_ctrl' (32, 1)
			'pred_features_ctrl' (32, 256)
			'mu_branches' (32, 2) 均值
			'sigma_branches' (32, 2) 标准差
			'future_feature' (list 4) 存储了4个（32, 256）的tensor 张量
			'future_mu' (list 4) 存储了4个（32, 2）的tensor
			'future_sigma' (list 4) 存储了4个（32, 2）的tensor
		'''

		# ----------------------------------计算当前的误差loss--------------------------------------------
		# Beta 分布是一个定义在 [0, 1] 区间上的连续概率分布，它通常用于表示随机变量的概率分布，该变量的值在 0 和 1 之间
		dist_sup = Beta(batch['action_mu'], batch['action_sigma'])  # gt概率分布 ground truth
		dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])  # 预测的概率分布
		# action_mu = batch['action_mu'] # shape： (32, 2)
		# action_sigma = batch['action_sigma'] # shape： (32, 2)
		kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)  # 计算预测的分布和gt的分布之间的KL散度(即：这两个概率分布的相似度) （32, 2）

		# 开始计算loss
		# 提取 kl_div 张量的第一列，代表与第一个动作（例如，加速度）相关的 KL 散度。
		# 提取 kl_div 张量的第二列，代表与第二个动作（例如，转向）相关的 KL 散度。
		action_loss = torch.mean(kl_div[:, 0]) * 0.5 + torch.mean(kl_div[:, 1]) * 0.5  # 两个动作(acc，steer)，各占0.5权重
		'''
		L1 Loss（L1 损失），也称为 绝对误差损失（Mean Absolute Error, MAE），
		是一种损失函数，用于衡量预测值与真实值之间的差异。它的计算方法非常简单，即计算预测值与真实值之间的绝对差值，然后求平均。
		
		MSE Loss（Mean Squared Error Loss），也称为 均方误差损失，是一种常见的损失函数，用于回归任务中，衡量模型的预测值与真实值之间的差异。
		回归简单说就是，给一堆离散点，找出一条曲线，去拟合这些离散点。
		MSE Loss 计算的是预测值与真实值之间的平方差的平均值。
		'''
		# 第二个参数是ground truth，也就是真实值
		# 下面这个函数的两个参数分别是预测值和真实值，类型是张量tensor
		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
		value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight
		# feature loss是干啥的
		# TODO
		feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) + F.mse_loss(pred['pred_features_ctrl'], feature)) * self.config.features_weight

		#------------------------计算未来的误差loss---------------------------------------

		future_feature_loss = 0
		future_action_loss = 0
		for i in range(self.config.pred_len):  # 未来4帧 pred_len=4
			# batch是输入的数据，包括未来4帧的数据，包括未来4帧的标签，包括未来4帧的路径点，包括未来4帧的动作，包括未来4帧的动作的概率分布，包括未来4帧的动作的均值和方差
			dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
			dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
			future_action_loss += torch.mean(kl_div[:, 0]) * 0.5 + torch.mean(kl_div[:, 1]) * 0.5  # 累加loss
			future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight  # 累加loss
		future_feature_loss /= self.config.pred_len  # 计算平均loss
		future_action_loss /= self.config.pred_len  # 计算平均loss

		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()  # 计算预测路径的loss 就一个数值
		loss = action_loss + speed_loss + value_loss + feature_loss + wp_loss + future_feature_loss + future_action_loss  # 计算整体的loss
		self.log('train_action_loss', action_loss.item())
		self.log('train_wp_loss_loss', wp_loss.item())
		self.log('train_speed_loss', speed_loss.item())
		self.log('train_value_loss', value_loss.item())
		self.log('train_feature_loss', feature_loss.item())
		self.log('train_future_feature_loss', future_feature_loss.item())
		self.log('train_future_action_loss', future_action_loss.item())
		return loss

	def configure_optimizers(self):  # 配置优化器和learning rate
		# weight_decay=1e-7：设置权重衰减（L2 正则化），用于防止过拟合。这个值控制参数的衰减程度。
		# self.parameters()：调用模型的参数，返回一个可迭代的参数列表，优化器将对这些参数进行优化。
		optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-7)
		# 每 30 个 epoch（或训练步骤）后调整一次学习率。每次调整时将学习率乘以 0.5，即将学习率减半。
		lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 30, 0.5)
		# 返回一个包含优化器的列表和一个包含学习率调度器的列表
		return [optimizer], [lr_scheduler]
	# 验证step
	def validation_step(self, batch, batch_idx):
		# batch是一个字典，包括：
		# 'front_img'：前视图图像，shape：(32, 3, 160, 320)
		# 'speed'：车速，shape：(32, 1)
		# 'target_point'：目标点，shape：(32, 2)
		# 'target_command'：目标命令，shape：(32, 1)
		# 'action_mu'：动作的均值，shape：(32, 2)
		front_img = batch['front_img']
		speed = batch['speed'].to(dtype=torch.float32).view(-1, 1) / 12.
		target_point = batch['target_point'].to(dtype=torch.float32)
		command = batch['target_command']
		# 按照第二个维度 也就是列进行组合 cat
		state = torch.cat([speed, target_point, command], 1)
		# reshape  
		# -1 表示该维度的大小将自动推断，以使总元素数量保持不变。
		value = batch['value'].view(-1, 1)
		feature = batch['feature']
		# ground truth waypoints
		gt_waypoints = batch['waypoints']
		# 功能类似于前向传播
		pred = self.model(front_img, state, target_point)
		# 可能是“supervision”的缩写，表示监督或指导。 dist表示分布
		dist_sup = Beta(batch['action_mu'], batch['action_sigma'])
		dist_pred = Beta(pred['mu_branches'], pred['sigma_branches'])
		kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
		action_loss = torch.mean(kl_div[:, 0]) * 0.5 + torch.mean(kl_div[:, 1]) * 0.5
		speed_loss = F.l1_loss(pred['pred_speed'], speed) * self.config.speed_weight
		value_loss = (F.mse_loss(pred['pred_value_traj'], value) + F.mse_loss(pred['pred_value_ctrl'], value)) * self.config.value_weight
		feature_loss = (F.mse_loss(pred['pred_features_traj'], feature) + F.mse_loss(pred['pred_features_ctrl'], feature)) * self.config.features_weight
		# waypoint loss
		wp_loss = F.l1_loss(pred['pred_wp'], gt_waypoints, reduction='none').mean()
		# 从 batch 字典中提取 action_mu 张量的形状信息，并将第一个维度的大小赋值给变量 B
		B = batch['action_mu'].shape[0]  # B: 32
		# 方向盘 刹车 加速 
		batch_steer_l1 = 0 
		batch_brake_l1 = 0
		batch_throttle_l1 = 0
		for i in range(B):
			throttle, steer, brake = self.model.get_action(pred['mu_branches'][i], pred['sigma_branches'][i])
			# L1误差
			batch_throttle_l1 += torch.abs(throttle-batch['action'][i][0])  # 把一批次的loss都加上
			batch_steer_l1 += torch.abs(steer-batch['action'][i][1])
			batch_brake_l1 += torch.abs(brake-batch['action'][i][2])

		batch_throttle_l1 /= B  # 计算一批次的loss的平均值
		batch_steer_l1 /= B
		batch_brake_l1 /= B

		future_feature_loss = 0
		future_action_loss = 0
		for i in range(self.config.pred_len-1):
			dist_sup = Beta(batch['future_action_mu'][i], batch['future_action_sigma'][i])
			dist_pred = Beta(pred['future_mu'][i], pred['future_sigma'][i])
			kl_div = torch.distributions.kl_divergence(dist_sup, dist_pred)
			future_action_loss += torch.mean(kl_div[:, 0]) * 0.5 + torch.mean(kl_div[:, 1]) * 0.5
			future_feature_loss += F.mse_loss(pred['future_feature'][i], batch['future_feature'][i]) * self.config.features_weight
		future_feature_loss /= self.config.pred_len
		future_action_loss /= self.config.pred_len
		# 验证loss
		val_loss = wp_loss + batch_throttle_l1 + 5*batch_steer_l1 + batch_brake_l1  # 路径的损失再加上油门，转向以及刹车的损失，其中比较关注转向操作

		self.log("val_action_loss", action_loss.item(), sync_dist=True)
		self.log('val_speed_loss', speed_loss.item(), sync_dist=True)
		self.log('val_value_loss', value_loss.item(), sync_dist=True)
		self.log('val_feature_loss', feature_loss.item(), sync_dist=True)
		self.log('val_wp_loss_loss', wp_loss.item(), sync_dist=True)
		self.log('val_future_feature_loss', future_feature_loss.item(), sync_dist=True)
		self.log('val_future_action_loss', future_action_loss.item(), sync_dist=True)
		self.log('val_loss', val_loss.item(), sync_dist=True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# 添加命令行参数的定义
	parser.add_argument('--id', type=str, default='TCP', help='Unique experiment identifier.')
	parser.add_argument('--epochs', type=int, default=60, help='Number of train epochs.')
	parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate.')
	parser.add_argument('--val_every', type=int, default=3, help='Validation frequency (epochs).')
	parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('--logdir', type=str, default='log', help='Directory to log data to.')
	parser.add_argument('--gpus', type=int, default=1, help='number of gpus')

	args = parser.parse_args()
	args.logdir = os.path.join(args.logdir, args.id)

	# Config
	config = GlobalConfig()  # 实例化配置文件的类

	# Data
	'''
	batch是一次加载32组数据，每组数据一个大小为14的dict(字典)，如下：
	'front_img' (32, 3, 256, 900)
	'waypoints' (32, 4, 2)
	'action' (32, 3) ---->油门，转向，刹车
	'action_mu' (32, 2)
	'action_sigma' (32, 2)
	'target_point' (32, 2) 目标点的位置（局部坐标系下），用于导航或路径规划。
	'target_point_aim' (32, 2) 目标点的另一种表示形式，可能用于调整导航目标
	'speed' (32,) --->速度就一个数值
	'feature' （32, 256）
	'value' (32, ) 与当前样本相关的值函数，可能用于强化学习中的状态价值估计
	'target_command' (32, 6) 目标命令的 one-hot 编码，用于表示当前样本中车辆应执行的命令，如转弯、直行、换道等
	'future_action_mu' list: 4	----->0: (32, 2); 1: (32, 2); 2:(32, 2); 3: (32, 2)
	'future_action_sigma' list: 4	----->0: (32, 2); 1: (32, 2); 2:(32, 2); 3: (32, 2)
	'future_feature' ----->0: (32, 256); 1: (32, 256); 2:(32, 256); 3: (32, 256)
	'''

	train_set = CARLA_Data(root=config.root_dir_all, data_folders=config.train_data, img_aug=config.img_aug)  # 使用数据增强
	print(len(train_set))  # 189524
	val_set = CARLA_Data(root=config.root_dir_all, data_folders=config.val_data,)
	print(len(val_set))  # 27201
	'''
	注意： 
		在最原始的读取的npy文件中数据为:
		'action', 车辆控制指令（如油门、刹车、方向）
		'action_mu', 动作分布的均值参数，通常用于表示动作的期望值
		'action_sigma', 动作分布的标准差参数，表示动作的随机性或不确定性
		'command', 
		'feature', 
		'front_img', 
		'functions', 
		'future_action', 
		'future_action_mu', 未来几个时间步内的动作分布均值参数
		'future_action_sigma', 未来几个时间步内的动作分布标准差参数
		'future_feature', 未来几个时间步内的特征数据，可能用于预测或规划------------------------------------特征数据
		'future_only_ap_brake', 
		'future_theta', 
		'future_x', 
		'future_y', 
		'only_ap_brake', 
		'speed',
		'target_command', 目标命令的 one-hot 编码，用于表示当前样本中车辆应执行的命令，如转弯、直行、换道等
		'target_gps', 
		'theta', 
		'value',  与当前样本相关的值函数，可能用于强化学习中的状态价值估计
		'x', 
		'x_command', 
		'y', 
		'y_command'
		但是，经过CARLA_Data处理完后变成了batch中使用的14个
		CARLA_Data在数据增强和预处理是在 __getitem__ 方法中动态应用的，而不是改变了原始数据本身的种类。当模型训练或推理过程中需要从数据集中获取数据时，才会调用 __getitem__ 方法。
		所以，你会发现train_set里就是原先npy的数据，这就是为什么batch里的数据和train_set中不一样的原因
		# TODO
	'''

	dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8)
	dataloader_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=8)

	TCP_model = TCP_planner(config, args.lr)  # 实例化模型

	# 回调类，用于在训练过程中自动保存模型的权重
	checkpoint_callback = ModelCheckpoint(
										  save_weights_only=False,  # 保存完整的模型状态，包括模型的参数、优化器状态、学习率调度器等
										  mode="min",  # 选择最小的 val_loss 作为最优模型
										  monitor="val_loss",  # 监控的指标
										  save_top_k=2,  # 只保存最好的2个模型
										  save_last=True,  # 在训练结束时，保存当前训练周期的最后一个模型状态; 这个检查点不一定是性能最好的，但它代表了模型在训练结束时的状态
										  dirpath=args.logdir,  # 检查点保存路径
										  filename="best_{epoch:02d}-{val_loss:.3f}"  # 文件名格式
	)
	checkpoint_callback.CHECKPOINT_NAME_LAST = "{epoch}-last"  # 自定义最后一个检查点的文件名格式

	# 从命令行参数中创建一个 Trainer 实例。这个方法可以方便地将命令行参数解析并传递给 Trainer
	# 创建 Trainer 并传入回调和其他配置参数
	trainer = pl.Trainer.from_argparse_args(
											args,  # args 是通过 argparse.ArgumentParser 解析命令行参数后得到的对象，包含了用户在命令行中输入的所有参数及其值
											default_root_dir=args.logdir,  # 设置训练日志和检查点的默认保存目录
											gpus=args.gpus,  # 指定使用的 GPU 数量
											accelerator='ddp',  # 设置加速器类型，这里使用的是分布式数据并行（DDP）
											sync_batchnorm=True,  # 在多 GPU 训练中同步批归一化层
											plugins=DDPPlugin(find_unused_parameters=False),  # 插件配置，这里使用 DDPPlugin 并设置 find_unused_parameters=False 以加速训练
											profiler='simple',  #  设置性能分析器，这里使用简单分析器（simple）， 最后会输出分析结果，耗时等等这些
											benchmark=True,  # 设置为 True 以提高 CUDNN 后端的性能
											log_every_n_steps=1,  # 设置每多少步记录一次日志
											flush_logs_every_n_steps=5,  # 设置每多少步刷新一次日志
											callbacks=[checkpoint_callback,
														],
											check_val_every_n_epoch=args.val_every,  # 设置每多少个 epoch 进行一次验证
											max_epochs=args.epochs  # 设置最大训练 epoch 数
	)
	# 训练数据的加载器，通常是一个 DataLoader 实例，用于批量加载训练数据。DataLoader 会处理数据的迭代、打乱、批次大小等。
	# 这是验证数据的加载器，通常也是一个 DataLoader 实例。它在训练过程中用于评估模型的性能，以防止过拟合并监控模型的泛化能力。
	trainer.fit(TCP_model, dataloader_train, dataloader_val)  # 传入模型和数据，启动训练




		




