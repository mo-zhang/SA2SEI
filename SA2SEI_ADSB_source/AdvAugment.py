import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvAug(nn.Module):
    def __init__(self, online_network, n_power=1, XI=0.01, epsilon=1.0):
        super(AdvAug, self).__init__()
        self.online_network = online_network  # 初始化在线网络
        self.n_power = n_power  # 对抗性增强的次数
        self.XI = XI  # 扰动大小
        self.epsilon = epsilon  # 扰动幅度

    def forward(self, X):
        # 前向传播函数，生成对抗性增强样本
        aug_sample = generate_adv_aug_sample(X, self.online_network, self.n_power, self.XI, self.epsilon)
        return aug_sample

# 计算两个 logit 之间的 KL 散度
def kl_divergence_with_logit(q_logit, p_logit):
    q = F.softmax(q_logit, dim=1)
    qlogq = torch.mean(torch.sum(q * F.log_softmax(q_logit, dim=1), dim=1))
    qlogp = torch.mean(torch.sum(q * F.log_softmax(p_logit, dim=1), dim=1))
    return qlogq - qlogp

# 标准化向量
def get_normalized_vector(d):
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
    return d

# 生成对抗性扰动
def generate_adv_perturbation(x, online_network, n_power, XI, epsilon):
    d = torch.randn_like(x)  # 生成随机扰动
    loss_mse = nn.MSELoss()  # 均方误差损失函数
    if torch.cuda.is_available():  # 如果可用 GPU
        loss_mse = loss_mse.cuda()  # 将损失函数移动到 GPU

    for _ in range(n_power):
        # 多次应用对抗性扰动的迭代更新过程
        d = XI * get_normalized_vector(d).requires_grad_()  # 标准化扰动，并设置梯度
        feature = online_network(x)[0]  # 提取原始样本的特征
        feature_ad = online_network(x + d)[0]  # 提取增强后样本的特征
        dist = loss_mse(feature, feature_ad)  # 计算特征之间的均方误差
        grad = torch.autograd.grad(dist, [d])[0]  # 计算特征之间的梯度
        d = grad.detach()  # 断开梯度连接，避免梯度传播

    return epsilon * get_normalized_vector(d)  # 返回扰动

# 生成对抗性增强样本
def generate_adv_aug_sample(x, online_network, n_power, XI, epsilon):
    r_adv = generate_adv_perturbation(x, online_network, n_power, XI, epsilon)  # 生成对抗性扰动
    x_adv = x + r_adv  # 将扰动应用到输入样本上，生成增强样本
    return x_adv.detach()  # 返回增强样本
