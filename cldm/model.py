import os
import torch

from omegaconf import OmegaConf
from ldm.util import instantiate_from_config


def get_state_dict(d):
    # 从字典d中获取'state_dict'键对应的值，如果不存在则返回d本身
    return d.get('state_dict', d)

def load_state_dict(ckpt_path, location='cpu'):
    # 加载指定路径的检查点文件，并返回其中的状态字典
    _, extension = os.path.splitext(ckpt_path)  # 获取文件扩展名
    if extension.lower() == ".safetensors":  # 针对.safetensors格式的检查点
        import safetensors.torch  # 动态导入safetensors模块
        # 使用safetensors库加载状态字典，指定存放位置
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        # 使用torch.load加载一般的PyTorch检查点
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))

    # 获取状态字典，确保返回的是状态字典而非其他数据
    state_dict = get_state_dict(state_dict)

    # 打印已加载的状态字典来源
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict  # 返回加载的状态字典


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model
