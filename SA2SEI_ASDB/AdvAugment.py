import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvAug(nn.Module):
    """
    We define a function of regularization, specifically VAT.
    """

    def __init__(self, online_network, n_power=1, XI=0.01, epsilon=1.0):
        super(AdvAug, self).__init__()
        # 初始化函数，接收网络模型、VAT中计算对抗扰动的次数、生成对抗扰动的超参数
        self.online_network = online_network
        self.n_power = n_power
        self.XI = XI
        self.epsilon = epsilon

    def forward(self, X):
        # 前向传播函数，生成对抗样本增强后的样本并返回
        aug_sample = generate_adv_aug_sample(X, self.online_network, self.n_power, self.XI, self.epsilon)
        return aug_sample

def get_normalized_vector(d):
    # 计算归一化向量
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

def generate_adv_perturbation(x, online_network, n_power, XI, epsilon):
    # 生成对抗扰动的函数
    d = torch.randn_like(x)
    loss_mse = nn.MSELoss()  # 使用均方误差作为损失函数
    if torch.cuda.is_available():
        loss_mse = loss_mse.cuda()

    for _ in range(n_power):
        d = XI * get_normalized_vector(d).requires_grad_()
        feature = online_network(x)[0]
        feature_ad = online_network(x + d)[0]
        dist = loss_mse(feature, feature_ad)
        grad = torch.autograd.grad(dist, [d])[0]
        d = grad.detach()  # 更新扰动
    return epsilon * get_normalized_vector(d)

def generate_adv_aug_sample(x, online_network, n_power, XI, epsilon):
    # 生成对抗样本增强的函数
    r_adv = generate_adv_perturbation(x, online_network, n_power, XI, epsilon)
    x_adv = x + r_adv  # 将对抗性扰动添加到输入样本上
    return x_adv.detach()   # 返回增强后的对抗样本
