import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from src.model import ResNetTransfer
from src.data_preprocessing import build_dataloaders

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 註冊 Hook 來獲取梯度與特徵圖
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_image, class_idx):
        # 前向傳播
        output = self.model(input_image)
        self.model.zero_grad()
        
        # 反向傳播目標類別的梯度
        score = output[:, class_idx]
        score.backward()
        
        # 計算特徵圖權重 (GAP of gradients)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # 加權求和並通過 ReLU
        cam = torch.sum(weights * self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        
        # 歸一化到 0-1
        cam -= cam.min()
        cam /= cam.max()
        return cam.detach().cpu().numpy()

# 視覺化函式：將熱力圖蓋在原圖上
def overlay_heatmap(img, heatmap, alpha=0.5):
    img_h, img_w, _ = img.shape
    heatmap_resized = cv2.resize(heatmap, (img_w, img_h))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # 疊加
    result = cv2.addWeighted(np.uint8(255 * img), 1 - alpha, heatmap_color, alpha, 0)
    return result