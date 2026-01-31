import gzip
import struct
import numpy as np
import torch
from torch.utils.data import Dataset

# 固定文件路径
train_labels_path = './data/trainl.gz'  # 标签文件
train_images_path = './data/traini.gz'  # 图像文件
test_labels_path = './data/t10kl.gz'  # 测试标签
test_images_path = './data/t10ki.gz'  # 测试图像


def read_mnist_idx(gz_file_path, is_image=False):
    with gzip.open(gz_file_path, 'rb') as f:
        magic_num = struct.unpack('>I', f.read(4))[0]
        if is_image and magic_num != 2051:
            raise ValueError(f"图像文件魔数错误，预期2051，实际{magic_num}")
        if not is_image and magic_num != 2049:
            raise ValueError(f"标签文件魔数错误，预期2049，实际{magic_num}")
        num_data = struct.unpack('>I', f.read(4))[0]
        if is_image:
            rows = struct.unpack('>I', f.read(4))[0]
            cols = struct.unpack('>I', f.read(4))[0]
            data = np.frombuffer(f.read(), dtype=np.uint8)
            data = data.reshape(num_data, rows, cols)
        else:
            data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def main():

    train_images = read_mnist_idx(train_images_path, is_image=True)
    train_labels = read_mnist_idx(train_labels_path, is_image=False)
    test_images = read_mnist_idx(test_images_path, is_image=True)
    test_labels = read_mnist_idx(test_labels_path, is_image=False)

    print("===== 训练集（原始数据） =====")
    print(f"图像形状：{train_images.shape}  # (N,H,W)，CNN需转为(N,C,H,W)")
    print(f"标签形状：{train_labels.shape} ")
    print(f"像素值范围：{train_images.min()} ~ {train_images.max()} ")
    print(f"标签值范围：{train_labels.min()} ~ {train_labels.max()} ")

    print("\n===== 测试集（原始数据） =====")
    print(f"图像形状：{test_images.shape}")
    print(f"标签形状：{test_labels.shape}")


def data_load():
    train_images = read_mnist_idx(train_images_path, is_image=True)
    train_labels = read_mnist_idx(train_labels_path, is_image=False)
    test_images = read_mnist_idx(test_images_path, is_image=True)
    test_labels = read_mnist_idx(test_labels_path, is_image=False)

    train_x = torch.from_numpy(train_images.copy()).float() / 255.0
    train_x = train_x.unsqueeze(1)
    test_x = torch.from_numpy(test_images.copy()).float() / 255.0
    test_x = test_x.unsqueeze(1)
    train_y = torch.from_numpy(train_labels.copy()).long()
    test_y = torch.from_numpy(test_labels.copy()).long()

    print("===== CNN适配后张量信息 =====")
    print(f"训练图像形状：{train_x.shape}  # (N,C,H,W) 符合CNN输入规范")
    print(f"测试图像形状：{test_x.shape}")
    print(f"训练标签类型：{train_y.dtype}  # long型，适配交叉熵损失")

    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = data_load()


class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):

        return self.images[idx], self.labels[idx]


train_dataset = MNISTDataset(train_x, train_y)
test_dataset = MNISTDataset(test_x, test_y)

batch_size = 64





if __name__ == "__main__":
    main()
