from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import h5py


# 定义训练数据集类
class TrainDataset(Dataset):   # Dataset为基类，自定义Dataset必须实现 __init__、__getitem__、__len__三个函数，分别为自定义构造器，获取指定索引的数据，返回数据集总长度
    def __init__(self, data_root, crop_size, arg=True, bgr2rgb=True, stride=8):   #__init__是一个构造器，self为Dataset
        #data_root: ../dataset/
        #crop_size: 128
        #arg=True
        #bgr2rgb=True
        #stride=8
        self.crop_size = crop_size  # 裁剪大小
        self.hypers = []  # 保存高光谱图片的数据列表
        self.bgrs = []  # 保存RGB图片的数据列表
        self.arg = arg  # 是否进行数据增强
        h, w = 482, 512  # 图片的高度和宽度
        self.stride = stride  # 裁剪的步长
        self.patch_per_line = (w - crop_size) // stride + 1  # 每行可以采集的裁剪块数量
        self.patch_per_colum = (h - crop_size) // stride + 1  # 每列可以采集的裁剪块数量
        self.patch_per_img = self.patch_per_line * self.patch_per_colum  # 每张图片可以采集的裁剪块总数

        # 高光谱和RGB图片的路径
        hyper_data_path = f'{data_root}/Train_Spec/'    # f'可直接将data_root转化成他对应的值-路径
        bgr_data_path = f'{data_root}/Train_RGB/'

        # 读取训练数据列表文件
        with open(f'{data_root}/split_txt/train_list.txt', 'r') as fin:    #将打开文件的对象赋给fin
            # 生成高光谱和RGB图片的文件列表
            hyper_list = [line.replace('\n', '.mat') for line in fin]     #for line in fin代表将fin文件中的每一行后面的'\n'变成'.mat'（方便用matlab进行处理），变成一哥字符串列表
            bgr_list = [line.replace('mat', 'jpg') for line in hyper_list]

        hyper_list.sort()  # 对高光谱文件列表进行排序   ['class1.mat', 'class2.mat', 'class3.mat']  无返回值，直接修改原列表
        bgr_list.sort()  # 对RGB文件列表进行排序

        print(f'len(hyper) of ntire2022 dataset: {len(hyper_list)}')  #输出  len(hyper) of ntire2022 dataset:列表长度
        print(f'len(bgr) of ntire2022 dataset: {len(bgr_list)}')

        # 读取每张图片并进行预处理
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]     #选中目标文件地址hyper_data_path中具体的文件hyper_path，类似于i++
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:        #打开具体文件中HDF5格式的文件，并命名为mat
                hyper = np.float32(np.array(mat['cube']))  # 读取mat中名为'cube'的高光谱数据集；np.array将其转换为numpy数组；np.float32将数组中每一个元素其转换为位浮点数；数组名为hyper
            hyper = np.transpose(hyper, [0, 2, 1])  # 调整维度顺序    hyper是一个三维数组，分别代表横坐标，纵坐标，波段；这里是指交换第二维度与第三维度  得看原始数据里面的顺序是（channel，width，height）

            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            #assert语句来检查高光谱数据文件名和RGB图像文件名在去除文件扩展名后的部分是否相同； split('.')[0]表示按照’.‘把路径名分割，选取第0部分；
            bgr = cv2.imread(bgr_path)  # 读取BGR图片  imread 函数会打开bgr_path指定的文件，并将其为存入名为bgr的数组。imread 读取的图像是以 BGR 顺序存储的，这意味着红色和蓝色通道的顺序与通常的 RGB 顺序相反。
            if bgr2rgb:    #表示是否将图片颜色空间从BGR转换为RGB
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # BGR转RGB，之下的图像都是RGB
            bgr = np.float32(bgr)  # 转为float32类型
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())  # 归一化  将图像的像素值缩放到0到1的范围内。
            bgr = np.transpose(bgr, [2, 0, 1])  # 调整维度顺序     对RGB图像的维度进行转置，将颜色通道放在第一个维度，高度放在第二个维度，宽度放在第三个维度。

            self.hypers.append(hyper)  # 将当前处理好的高光谱数据（hyper数组）添加到类实例的self.hypers列表中
            self.bgrs.append(bgr)  # 将处理好的RGB图像（bgr数组）添加到类属性self.bgrs的列表中    bgrs数组里面的每一个bgr项都是(channel，height，width)数组，channel的顺序是BGR顺序
            mat.close()  #这行代码关闭之前打开的HDF5文件
            print(f'Ntire2022 scene {i} is loaded.')  #这行代码打印出当前处理的Ntire2022数据集的场景编号。i是循环的索引，表示当前图片的地址eg：ARAD_1K_0004

        self.img_num = len(self.hypers)  # 图片数量
        self.length = self.patch_per_img * self.img_num  # 数据集的总长度

    # 数据增强函数
    def arguement(self, img, rotTimes, vFlip, hFlip):
        # 随机旋转
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))   #img.copy() 创建了一个img数组的副本，这是因为rot90函数会修改原始数组；轴的索引是从0开始的，所以axes=(1, 2)意味着图像将在垂直和水平方向上旋转90度。
        # 随机垂直翻转
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()   #这表示不改变第一个和第二个维度（高度和宽度）。::-1：这表示在第三个维度（波段或通道）上进行反向切片，即从最后一个波段开始，到第一个波段结束。
        # 随机水平翻转
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()  #::-1：这表示在第二个维度（宽度）上进行反向切片，即从最后一个宽度开始，到第一个宽度结束，应该是反向第二维度的数组
        return img

    # 获取指定索引的数据
    def __getitem__(self, idx):
        stride = self.stride   #在裁剪图像时每个块之间的步长。
        crop_size = self.crop_size   #裁剪图像块的大小
        img_idx, patch_idx = idx // self.patch_per_img, idx % self.patch_per_img   #patch_per_img 每张图片可以采集的裁剪块总数；img_idx为第几张图片；patch_idx为图片内的第几个裁剪快
        h_idx, w_idx = patch_idx // self.patch_per_line, patch_idx % self.patch_per_line  #patch_per_line 每行可以采集的裁剪块数量；h_idx为第几行；w_idx为第几列    直接定位到裁剪块

        bgr = self.bgrs[img_idx]  #获取目标索引块的RGB裁剪块信息
        hyper = self.hypers[img_idx]   #获取目标索引块的高光谱信息

        # 裁剪出指定位置的块    bgr和hyper为裁剪出来的目标块
        bgr = bgr[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]   # h_idx * stride为长度上的起始索引，h_idx * stride + crop_size为长度上的结束索引
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size, w_idx * stride:w_idx * stride + crop_size]

        # 随机旋转和翻转
        rotTimes = random.randint(0, 3)   #random.randint(0, 3)表示生成的随机整数将在0到3之间（包括0和3）。   rotTimes变量被设置为这个随机整数，它表示图像将进行随机旋转的次数。
        vFlip = random.randint(0, 1)    #vFlip变量被设置为这个随机整数，它表示图像将进行垂直翻转的概率。如果vFlip为1，图像将进行垂直翻转；如果为0，则不进行垂直翻转。
        hFlip = random.randint(0, 1)    #hFlip变量被设置为这个随机整数，它表示图像将进行水平翻转的概率。如果hFlip为1，图像将进行水平翻转；如果为0，则不进行水平翻转。
        if self.arg:    #如果arg为true，表示启用了数据增强
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)    #self.arguement是一个数据增强函数，它接收BGR图像、旋转次数、垂直翻转概率和水平翻转概率作为参数，并返回应用了数据增强的BGR图像。
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)

        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)    #这行代码使用NumPy的ascontiguousarray函数来确保BGR图像数组bgr是连续存储的。

    def __len__(self):
        return self.patch_per_img * self.img_num  # 返回数据集的总长度


# 定义验证数据集类
class ValidDataset(Dataset):
    def __init__(self, data_root, bgr2rgb=True):
        self.hypers = []
        self.bgrs = []

        # 高光谱和RGB图片的路径
        hyper_data_path = f'{data_root}/Train_Spec/'
        bgr_data_path = f'{data_root}/Train_RGB/'

        # 读取验证数据列表文件
        with open(f'{data_root}/split_txt/valid_list.txt', 'r') as fin:
            # 生成高光谱和RGB图片的文件列表
            hyper_list = [line.replace('\n', '.mat') for line in fin]
            bgr_list = [line.replace('mat', 'jpg') for line in hyper_list]

        hyper_list.sort()
        bgr_list.sort()

        print(f'len(hyper_valid) of ntire2022 dataset: {len(hyper_list)}')
        print(f'len(bgr_valid) of ntire2022 dataset: {len(bgr_list)}')

        # 读取每张图片并进行预处理
        for i in range(len(hyper_list)):
            hyper_path = hyper_data_path + hyper_list[i]
            if 'mat' not in hyper_path:
                continue
            with h5py.File(hyper_path, 'r') as mat:
                hyper = np.float32(np.array(mat['cube']))  # 读取高光谱数据
            hyper = np.transpose(hyper, [0, 2, 1])  # 调整维度顺序

            bgr_path = bgr_data_path + bgr_list[i]
            assert hyper_list[i].split('.')[0] == bgr_list[i].split('.')[0], 'Hyper and RGB come from different scenes.'
            bgr = cv2.imread(bgr_path)  # 读取BGR图片
            if bgr2rgb:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
            bgr = np.float32(bgr)  # 转为float32类型
            bgr = (bgr - bgr.min()) / (bgr.max() - bgr.min())  # 归一化
            bgr = np.transpose(bgr, [2, 0, 1])  # 调整维度顺序

            self.hypers.append(hyper)  # 将处理好的高光谱数据添加到列表
            self.bgrs.append(bgr)  # 将处理好的RGB数据添加到列表
            mat.close()
            print(f'Ntire2022 scene {i} is loaded.')

    def __getitem__(self, idx):
        hyper = self.hypers[idx]  # 获取高光谱数据
        bgr = self.bgrs[idx]  # 获取RGB数据
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)  # 返回数据集的长度