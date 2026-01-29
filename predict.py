import torch
import torch.nn as nn
from torchvision import transforms
from model import Net
from PIL import Image
import os

MODEL_WEIGHT_PATH = "best_model.pth"
CLASS_MAP = {0: "猫", 1: "狗"}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {DEVICE}")


test_trans = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
def load_model():
    model = Net().to(DEVICE)
    if os.path.exists(MODEL_WEIGHT_PATH):
        model.load_state_dict(torch.load(MODEL_WEIGHT_PATH, map_location=DEVICE))
        print(f"成功加载模型权重: {MODEL_WEIGHT_PATH}")
    else:
        raise FileNotFoundError(f"未找到模型权重文件，请检查路径：{MODEL_WEIGHT_PATH}")
    model.eval()
    return model

def predict_single_image(model, img_path):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise Exception(f"加载图片失败: {e}，请检查图片路径是否正确、图片是否损坏")

    img_tensor = test_trans(img)

    img_tensor = torch.unsqueeze(img_tensor, 0).to(DEVICE)
    with torch.no_grad():
        output = model(img_tensor)

    prob = torch.softmax(output, dim=1)
    max_prob, pred_idx = torch.max(prob, dim=1)
    pred_label = CLASS_MAP[pred_idx.item()]
    pred_prob = max_prob.item()

    return pred_label, pred_prob

if __name__ == "__main__":
    model = load_model()
    test_imgs = [
        "/home/jack/python project/pic_classify/cat.jpg",
        "/home/jack/python project/pic_classify/cat1.jpg",
        "/home/jack/python project/pic_classify/dog.jpg",
        "/home/jack/python project/pic_classify/dog1.jpg"
    ]

    for img_path in test_imgs:
        pred_label, pred_prob = predict_single_image(model, img_path)

        print(f"{img_path} | 预测结果：{pred_label} | 预测概率：{pred_prob:.4f} ({pred_prob * 100:.2f}%)")
