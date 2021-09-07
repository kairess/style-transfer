import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from tqdm import trange
import os
from datetime import datetime

STYLE_LOSS_WEIGHT = 1e-05
CONTENT_IMG_PATH = os.path.join('imgs', '01.jpg')
STYLE_IMG_PATH = os.path.join('imgs', 'candinsky.jpg')
IMG_SIZE = 400
OUTPUT_DIR_PATH = os.path.join('output', '%s' % datetime.now().strftime('%Y%m%d_%H%M%S'))

os.makedirs(OUTPUT_DIR_PATH, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.vgg19_bn(pretrained=True)
print(model)
model = nn.Sequential(*list(model.features.children())[:52]).to(device)

for param in model.parameters():
    param.requires_grad = False

content_img = Image.open(CONTENT_IMG_PATH)
style_img = Image.open(STYLE_IMG_PATH)

T = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])])

iT = transforms.Normalize(
   mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
   std=[1/0.229, 1/0.224, 1/0.225]
)

content_tensor = T(content_img).unsqueeze(0)
style_tensor = T(style_img).unsqueeze(0)

class Extractor(nn.Module):
    features = None

    def __init__(self, layer):
        super().__init__()
        self.hook = layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def close(self):
        self.hook.remove()

# Extract features from content image
content_exts = [Extractor(model[i]) for i in [45]]
model(content_tensor.to(device))
content_features = [ext.features.clone() for ext in content_exts]

# Extract features from style image
style_exts = [Extractor(model[i]) for i in [2, 9, 16, 29, 42]]
model(style_tensor.to(device))
style_features = [ext.features.clone() for ext in style_exts]

input_tensor = content_tensor.clone().requires_grad_()
# input_tensor = torch.randn(content_tensor.shape, requires_grad=True)

# optimizer = torch.optim.Adam([input_tensor], lr=2e-02)
optimizer = torch.optim.SGD([input_tensor], lr=100.)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96)

def content_loss(y, y_hat):
    loss = 0
    for i in range(len(y_hat)):
        loss += F.mse_loss(y[i], y_hat[i])
    return loss / len(y_hat)

def gram_matrix(x):
    b, c, h, w = x.size()
    x = x.view(b * c, -1)
    return torch.mm(x, x.t())

def style_loss(y, y_hat):
    loss = 0
    for i in range(len(y_hat)):
        y_gram = gram_matrix(y[i])
        y_hat_gram = gram_matrix(y_hat[i])
        loss += F.mse_loss(y_gram, y_hat_gram)
    return loss / len(y_hat)

pbar = trange(10000+1)
for i in pbar:
    model(input_tensor.to(device))

    current_content_features = [ext.features.clone() for ext in content_exts]
    current_style_features = [ext.features.clone() for ext in style_exts]

    c_loss = content_loss(content_features, current_content_features)
    s_loss = style_loss(style_features, current_style_features) * STYLE_LOSS_WEIGHT
    loss = c_loss + s_loss
    pbar.set_description('Content loss %.6f Style loss %.6f' % (c_loss, s_loss))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 500 == 0:
        output_img = iT(input_tensor).detach().cpu().squeeze().permute(1, 2, 0).clamp(0, 1).numpy() * 255
        Image.fromarray(output_img.astype('uint8')).save(os.path.join(OUTPUT_DIR_PATH, '%s.png' % i))

        scheduler.step()
