import torch
import torch.nn as nn
import argparse
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os
from hsi_dataset import TrainDataset, ValidDataset
from architecture import *
from utils import AverageMeter, initialize_logger, save_checkpoint, record_loss, \
    time2file_name, Loss_MRAE, Loss_RMSE, Loss_PSNR
import datetime

# 解析命令行参数  在python中使用input读入数据，使用命令行参数来解析输入
parser = argparse.ArgumentParser(description="Spectral Recovery Toolbox")
parser.add_argument('--method', type=str, default='mst_plus_plus')  # 模型方法名称
parser.add_argument('--pretrained_model_path', type=str, default=None)  # 预训练模型路径  初始化新模型或继续训练
parser.add_argument("--batch_size", type=int, default=20, help="batch size")  # 批大小
parser.add_argument("--end_epoch", type=int, default=300, help="number of epochs")  # 训练轮数
parser.add_argument("--init_lr", type=float, default=4e-4, help="initial learning rate")  # 初始学习率
parser.add_argument("--outf", type=str, default='./exp/mst_plus_plus/', help='path log files')  # 日志文件路径
parser.add_argument("--data_root", type=str, default='../dataset/')  # 数据集路径
parser.add_argument("--patch_size", type=int, default=128, help="patch size")  # 图像块大小
parser.add_argument("--stride", type=int, default=8, help="stride")  # 步长
parser.add_argument("--gpu_id", type=str, default='0', help='path log files')  # GPU ID
opt = parser.parse_args()  #将用户通过命令行提供的参数解析为一个对象，然后将这个对象存储在opt变量中

# 设置CUDA设备
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'  #os.environ是Python中的环境变量字典  "CUDA_DEVICE_ORDER"是一个CUDA环境变量的名称  'PCI_BUS_ID'是这个环境变量的值。  这个环境变量告诉PyTorch如何按照PCI总线ID来排序和选择GPU。
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id  #opt.gpu_id是从之前解析的命令行参数中获取的GPU ID，告诉PyTorch哪些GPU是可见的，也就是哪些GPU可以被PyTorch使用。

# 加载数据集
print("\nloading dataset ...")
train_data = TrainDataset(data_root=opt.data_root, crop_size=opt.patch_size, bgr2rgb=True, arg=True, stride=opt.stride)  #crop_size表示图像块大小，arg 表示是否进行数据增强 ，bgr2rgb表示是否从BGR图像转换为RGB
print(f"Iteration per epoch: {len(train_data)}")
val_data = ValidDataset(data_root=opt.data_root, bgr2rgb=True)
print("Validation set samples: ", len(val_data))

# 设置迭代次数
per_epoch_iteration = 1000
total_iteration = per_epoch_iteration * opt.end_epoch  #迭代次数*训练轮数=总迭代次数

# 损失函数
criterion_mrae = Loss_MRAE()
criterion_rmse = Loss_RMSE()
criterion_psnr = Loss_PSNR()

# 创建模型
pretrained_model_path = opt.pretrained_model_path  #预训练模型路径
method = opt.method  #模型方法
model = model_generator(method, pretrained_model_path).cuda()  #model_generator函数通过接收模型方法和预训练权重路径，生成并返回一个深度学习模型实例，并加载相应的预训练权重。 .cuda()将模型及其参数转移到GPU上。这通常用于加速模型的训练和推理
print('Parameters number is ', sum(param.numel() for param in model.parameters()))  #model.parameters()返回一个包含模型所有参数的迭代器  计算模型所包含参数的总和
#numel()是PyTorch中一个函数，用于计算张量中元素的数量。

# 创建输出路径
date_time = str(datetime.datetime.now())  #只有一个datatime是不行的，datetime.now()是错误的用法，因为它返回的是time对象，只包含时间信息，没有日期信息。
date_time = time2file_name(date_time)  #time2file_name是一个自定义函数，它的作用是将日期和时间字符串转换为适合文件名的格式。
opt.outf = opt.outf + date_time  #opt.outf 日志文件路径  将转换后的日期和时间字符串添加到opt.outf的末尾，形成一个新的输出路径。
if not os.path.exists(opt.outf):  #使用Python的os模块检查opt.outf指定的路径是否存在 如果路径不存在，则执行下面的代码。
    os.makedirs(opt.outf)  #os为python内置与操作系统交互的模块，makedirs()是os模块中的一个函数，用于创建目录。

# 移动模型和损失函数到GPU
if torch.cuda.is_available():  #用于检查当前的PyTorch环境是否支持CUDA，即是否有可用的CUDA支持
    model.cuda()  #如果CUDA支持可用，则将模型（model）转移到GPU上
    criterion_mrae.cuda()
    criterion_rmse.cuda()
    criterion_psnr.cuda()

# 如果有多个GPU, 使用DataParallel
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# 优化器和学习率调度器
optimizer = optim.Adam(model.parameters(), lr=opt.init_lr, betas=(0.9, 0.999))  #Adam是optim模块中的一个优化器类  管理并更新模型中可学习参数的值，使得模型输出更接近真实标签。  model.parameters()返回一个包含模型所有参数的迭代器
# lr=opt.init_lr将学习率设置为opt.init_lr  betas=(0.9, 0.999)设置Adam优化器的动量系数（beta1）和一阶矩估计的指数衰减率（beta2）
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_iteration, eta_min=1e-6)  #CosineAnnealingLR是lr_scheduler模块中的一个学习率调度器类
# total_iteration表示整个训练过程中迭代的总次数  eta_min=1e-6设置退火过程结束后学习率将减少到的值。

# 日志记录
log_dir = os.path.join(opt.outf, 'train.log')  #path.join()将多个路径片段组合成一个路径
logger = initialize_logger(log_dir)  #建一个日志记录器，并将日志记录到指定的文件中。

# 恢复训练
resume_file = opt.pretrained_model_path  #预训练模型路径
if resume_file is not None:
    if os.path.isfile(resume_file):  #检查resume_file指定的路径是否是一个文件。
        print("=> loading checkpoint '{}'".format(resume_file))  #表示正在尝试加载检查点（checkpoint）
        checkpoint = torch.load(resume_file)  #load()函数从指定的文件路径加载检查点
        start_epoch = checkpoint['epoch']  #从加载的检查点中获取'epoch'键对应的值，并将其赋值给start_epoch变量  表示在检查点中记录的最后一个训练周期（epoch）的索引。
        iteration = checkpoint['iter']  #从加载的检查点中获取'iter'键对应的值，并将其赋值给iteration变量  表示在检查点中记录的最后一个迭代次数（iteration）。
        model.load_state_dict(checkpoint['state_dict'])  #load_state_dict()方法将检查点中的'state_dict'键对应的值加载到模型（model）中，使模型恢复到检查点中的状态，包括权重和任何其他参数
        optimizer.load_state_dict(checkpoint['optimizer'])  #load_state_dict()方法将检查点中的'optimizer'键对应的值加载到优化器（optimizer）中，使优化器恢复到检查点中的状态，包括学习率、动量
#上述epoch iter state_dict optimizer均在checkpoint字典中定义

# 主函数
def main():
    cudnn.benchmark = True  #可以加速卷积层的前向和反向传播，但可能会牺牲一些准确性。cudnn是PyTorch中的一个模块，它提供了CUDA加速的深度神经网络库 benchmark = True是cudnn模块的一个属性，用于优化卷积神经网络的执行。
    iteration = 0  #用于记录当前的迭代次数
    record_mrae_loss = 1000  # 记录最小MRAE损失值  初始化值大，训练时逐渐缩小
    while iteration < total_iteration:  #total_iteration是一个之前定义的变量，表示整个训练过程中迭代的总次数
        model.train()  # 设置模型为训练模式
        losses = AverageMeter()  #AverageMeter()为自定义的一个函数，用于存储训练过程中的损失值

        # 创建训练数据加载器  train_data与train_loader的区别为train_data是训练数据集，train_loader为训练数据加载器，train_loader是在数据集上的进一步处理
        train_loader = DataLoader(dataset=train_data, batch_size=opt.batch_size, shuffle=True, num_workers=2,
                                  pin_memory=True, drop_last=True)
        #Dataloader为python内置方法
        #dataset=train_data指定数据集为之前创建的TrainDataset实例。
        #batch_size=opt.batch_size指定每个批次的样本数量，这个值是从命令行参数中解析得到的。
        #shuffle=True表示在每次迭代时对数据进行随机打乱。
        #num_workers=2指定用于加载数据的进程数，这有助于提高数据加载的速度。
        #pin_memory=True表示将数据加载到固定内存中，以提高GPU的访问速度。
        #drop_last=True表示如果数据集的大小不是批大小的整数倍，则丢弃最后一个不完整的批次。
        #train_loader变量存储了创建的训练数据加载器实例。
        val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)  #验证数据加载器

        # 训练
        for i, (images, labels) in enumerate(train_loader):  #(images, labels)是当前批次的输入图像和标签
            labels = labels.cuda()  #将标签labels转移到GPU上
            images = images.cuda()
            images = Variable(images)  #将图像images包装在一个Variable对象中  Variable是PyTorch中的一个类，用于表示一个可微的数值对象
            labels = Variable(labels)
            lr = optimizer.param_groups[0]['lr']  #param_groups包含多个字典，每个字典代表优化器中的一个参数组，包含一组模型参数和与这些参数相关的优化选项。
            #param_groups[0]：这表示从列表中获取第一个参数组
            #optimizer.param_groups[0]['lr']这行代码的作用是获取优化器中第一个参数组的学习率
            optimizer.zero_grad()  #清除优化器optimizer中的所有参数的梯度    内置方法
            output = model(images)  #执行模型的前向传播。     内置方法
            loss = criterion_mrae(output, labels)  # 计算mrae(平均相对绝对误差)  自定义方法
            loss.backward()  # 反向传播 计算当前损失值相对于模型参数的梯度，并将这些梯度值存储在相应的Parameter对象中  内置方法
            optimizer.step()  # 更新权重  执行优化器的梯度下降步骤
            scheduler.step()  # 更新学习率  执行学习率调度器scheduler的更新步骤
            losses.update(loss.data)  #更新损失值losses  是AverageMeter的实例化对象
            iteration += 1  #增加迭代次数iteration

            # 每20次迭代输出一次训练损失
            if iteration % 20 == 0:
                print('[iter:%d/%d],lr=%.9f,train_losses.avg=%.9f'
                      % (iteration, total_iteration, lr, losses.avg))

            # 每1000次迭代进行一次验证
            if iteration % 1000 == 0:
                mrae_loss, rmse_loss, psnr_loss = validate(val_loader, model)
                print(f'MRAE:{mrae_loss}, RMSE: {rmse_loss}, PNSR:{psnr_loss}')

                # 保存模型
                if torch.abs(
                        mrae_loss - record_mrae_loss) < 0.01 or mrae_loss < record_mrae_loss or iteration % 5000 == 0:
                    print(f'Saving to {opt.outf}')
                    #torch.abs(mrae_loss - record_mrae_loss) < 0.01检查当前MRAE损失与记录的最小MRAE损失之间的差异是否小于0.01。
                    #mrae_loss < record_mrae_loss检查当前MRAE损失是否小于记录的最小MRAE损失。
                    #iteration % 5000 == 0检查当前迭代次数是否是5000的倍数。
                    save_checkpoint(opt.outf, (iteration // 1000), iteration, model, optimizer)  #自定义函数
                    if mrae_loss < record_mrae_loss:
                        record_mrae_loss = mrae_loss

                # 输出并记录验证损失
                print(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train MRAE: %.9f, Test MRAE: %.9f, "
                      "Test RMSE: %.9f, Test PSNR: %.9f " % (
                      iteration, iteration // 1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
                #每迭代1000次，输出一次训练信息
                logger.info(" Iter[%06d], Epoch[%06d], learning rate : %.9f, Train Loss: %.9f, Test MRAE: %.9f, "
                            "Test RMSE: %.9f, Test PSNR: %.9f " % (
                            iteration, iteration // 1000, lr, losses.avg, mrae_loss, rmse_loss, psnr_loss))
                #使用日志记录器记录训练信息
    return 0


# 验证函数
def validate(val_loader, model):  #val_loader用于加载验证数据集，而model是已经训练好的模型，用于进行预测。
    model.eval()  # 设置模型为评估模式
    losses_mrae = AverageMeter()  #创建一个AverageMeter对象，用于计算和存储验证过程中的RMSE（均方根误差）损失值。
    losses_rmse = AverageMeter()
    losses_psnr = AverageMeter()

    for i, (input, target) in enumerate(val_loader):
        input = input.cuda()
        target = target.cuda()
        with torch.no_grad():  #使用torch.no_grad()上下文管理器，禁用梯度计算
            # 计算输出
            output = model(input)
            loss_mrae = criterion_mrae(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])  #高度和宽度只采用图像中间部位
            loss_rmse = criterion_rmse(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])
            loss_psnr = criterion_psnr(output[:, :, 128:-128, 128:-128], target[:, :, 128:-128, 128:-128])

        # 记录损失
        losses_mrae.update(loss_mrae.data)
        losses_rmse.update(loss_rmse.data)
        losses_psnr.update(loss_psnr.data)

    return losses_mrae.avg, losses_rmse.avg, losses_psnr.avg


if __name__ == '__main__':  #用于执行一个名为main的函数
    main()
    print(torch.__version__)  #打印当前PyTorch的版本号
