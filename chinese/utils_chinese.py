from __future__ import division

import torch
import torch.nn as nn
import logging
import numpy as np
import os

# 平均值计算器，用于跟踪和计算平均消耗
class AverageMeter(object):   #当使用AverageMeter对象时，通常会先调用reset()方法来初始化所有属性，然后在每个迭代或批次结束时调用update()方法来更新平均值。
    def __init__(self):
        self.reset()  #构造函数

    def reset(self):
        self.val = 0  # 当前值
        self.avg = 0  # 平均值
        self.sum = 0  # 总和
        self.count = 0  # 数量

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n  # 更新总和
        self.count += n  # 更新数量
        self.avg = self.sum / self.count  # 计算平均值

# 初始化日志记录器
def initialize_logger(file_dir):    #参数file_dir，这是日志文件要写入的目录路径。
    logger = logging.getLogger()  # 无参构造一个全局logger对象
    fhandler = logging.FileHandler(filename=file_dir, mode='a')  # logging.FileHandler是一个日志记录器的构造方法，它将日志输出到文件。 filename=file_dir：设置日志文件的名称为file_dir指定的路径。 mode='a'：设置文件模式为追加（append），这意味着日志文件会被追加到已有内容后面，而不是每次都重写文件。
    formatter = logging.Formatter('%(asctime)s - %(message)s', "%Y-%m-%d %H:%M:%S")  # logging.Formatter是标准化日志消息格式类的构造方法。 '%(asctime)s - %(message)s'：设置日志消息的格式。%(asctime)s是日志消息的时间戳，%(message)s是日志消息的内容。"%Y-%m-%d %H:%M:%S"：设置时间戳的格式，即年月日时分秒。
    fhandler.setFormatter(formatter)  # 将格式应用到文件句柄
    logger.addHandler(fhandler)  # 将文件句柄添加到logger
    logger.setLevel(logging.INFO)  # 设置日志记录级别 这行代码设置日志记录器的级别为INFO。这意味着只有当消息的级别为INFO时，它们才会被记录。日志级别通常包括DEBUG < INFO < WARNING < ERROR < CRITICAL
    return logger

# 保存模型检查点
def save_checkpoint(model_path, epoch, iteration, model, optimizer):  #model_path：模型检查点要保存到的目录路径。epoch：当前训练的轮数。 iteration：当前迭代次数。model：训练的模型。optimizer：用于训练模型的优化器。
    state = {   #定义了一个字典state  key-value
        'epoch': epoch,  # 当前训练的轮数
        'iter': iteration,  # 当前迭代次数
        'state_dict': model.state_dict(),  # state_dict()会返回一个包含模型参数的字典   模型的权重和偏置等参数
        'optimizer': optimizer.state_dict(),  # 优化器的状态字典  优化器的内部状态，如学习率等。
    }

    torch.save(state, os.path.join(model_path, 'net_%depoch.pth' % epoch))  # 保存检查点 torch.save是一个用于保存PyTorch模型和状态的字典的函数。 state：要保存的状态字典。os.path.join函数来合并路径，并使用%d作为格式化字符串，将epoch的值插入到路径中，生成一个文件名，如net_10epoch.pth。

# 平均相对绝对误差(MRAE)损失函数
class Loss_MRAE(nn.Module):
    def __init__(self):
        super(Loss_MRAE, self).__init__()  #调用父类的构造函数

    def forward(self, outputs, label):  #定义前向传播过程   outputs：模型输出的张量，通常是与真实标签label形状相同的预测值。 label：真实标签的张量，与outputs形状相同。
        assert outputs.shape == label.shape  # 确保输入输出形状相同  否则返回抛出一个AssertionError
        error = torch.abs(outputs - label) / label  # 计算误差  torch.abs用于计算张量的绝对值
        mrae = torch.mean(error.view(-1))  # 计算平均相对绝对误差   error.view(-1)：这行代码将误差张量error展平成一个一维张量
        return mrae
#torch是用于张量处理的函数库
# 均方根误差(RMSE)损失函数
class Loss_RMSE(nn.Module):
    def __init__(self):
        super(Loss_RMSE, self).__init__()

    def forward(self, outputs, label):
        assert outputs.shape == label.shape  # 确保输入输出形状相同
        error = outputs - label  # 计算误差
        sqrt_error = torch.pow(error, 2)  # 计算平方误差
        rmse = torch.sqrt(torch.mean(sqrt_error.view(-1)))  # 计算均方根误差
        return rmse

# 峰值信噪比(PSNR)损失函数
class Loss_PSNR(nn.Module):
    def __init__(self):
        super(Loss_PSNR, self).__init__()
#在 PyTorch 中，通常希望图像数据是 [BatchSize, Channel, Height, Width]   但是，PIL Image 默认是 [Height, Width, Channel] 的格式。
    def forward(self, im_true, im_fake, data_range=255):  #im_true：真实图像的张量。 im_fake：生成图像的张量。 data_range：数据的范围，默认值为255。这可能是图像数据的动态范围，比如0到255。
        N = im_true.size()[0]  # 获取批次大小
        C = im_true.size()[1]  # 获取通道数   即RGB的通道数为3
        H = im_true.size()[2]  # 获取图像高度
        W = im_true.size()[3]  # 获取图像宽度
        Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)  # 处理真实图像im_true.clamp(0., 1.)：将图像的像素值限制在0到1之间。.mul_(data_range)：将图像的像素值乘以data_range，这可能是为了将图像的动态范围从0到1转换到另一个范围。.resize_(N, C * H * W)：将处理后的图像张量展平成一个一维张量，其大小为N * C * H * W。
        Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)  # 处理生成图像
        mse = nn.MSELoss(reduce=False)  # 计算均方误差   nn.MSELoss：这是一个用于计算均方误差的PyTorch损失函数。 reduce=False：这表示在计算损失时，不会对所有样本的损失值进行求和或平均。reduce=True：只返回一个标量
        err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)  # 计算误差  mse(Itrue, Ifake)：计算真实图像和生成图像之间的均方误差。 .sum(dim=1, keepdim=True)：沿着第一个维度（即批次维度）求和，但保持该维度的大小不变。 .div_(C * H * W)：将求和后的误差值除以C * H * W，得到每个图像块的误差。
        psnr = 10. * torch.log((data_range ** 2) / err) / np.log(10.)  # 计算PSNR  10. * torch.log((data_range ** 2) / err)：计算PSNR的值。  / np.log(10.)：将PSNR的值除以10的对数，以得到标准的PSNR值。
        return torch.mean(psnr)  #这行代码返回计算得到的PSNR值的平均

# 将时间字符串转换为文件名格式
def time2file_name(time):
    year = time[0:4]
    month = time[5:7]
    day = time[8:10]
    hour = time[11:13]
    minute = time[14:16]
    second = time[17:19]
    time_filename = year + '_' + month + '_' + day + '_' + hour + '_' + minute + '_' + second  # 拼接时间字符串
    return time_filename

# 记录损失到CSV文件中
def record_loss(loss_csv, epoch, iteration, epoch_time, lr, train_loss, test_loss):
#loss_csv：一个指向CSV文件的文件对象。
#epoch：当前训练的轮数。
#iteration：当前迭代次数。
#epoch_time：当前轮训练所需的时间。
#lr：当前的学习率。
#train_loss：当前轮的训练损失。
#test_loss：当前轮的测试损失。
    """记录多种结果到CSV文件。"""
    loss_csv.write('{},{},{},{},{},{}\n'.format(epoch, iteration, epoch_time, lr, train_loss, test_loss))  #format将函数参数epoch、iteration、epoch_time、lr、train_loss和test_loss的值格式化为字符串，并将它们作为逗号分隔的值写入到CSV文件中。
    #'{},{},{},{},{},{}\n'是一个格式化字符串，它的作用是将epoch、iteration、epoch_time、lr、train_loss和test_loss这些变量值格式化为CSV格式，并将它们写入到loss_csv指向的CSV文件中。
    #这个格式化字符串中的每个大括号{}都对应一个要插入变量的位置。当使用format方法时，它会将变量的值替换到这些位置，并在变量值之间添加逗号作为分隔符。\n表示新的一行。
    loss_csv.flush()   #这行代码调用flush方法，强制将缓冲区中的数据写入到文件中。 在多线程或I/O操作中，flush方法可以确保数据立即写入文件，而不是在某个时间点统一写入。
    loss_csv.close