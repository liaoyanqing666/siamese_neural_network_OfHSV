import numpy
import torch
from flask import Flask, request, render_template
from model import *
from PIL import Image
import torchvision.transforms as transforms


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('test.html')


@app.route('/', methods=['POST'])
def process():

    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = siamese()
    model.load_state_dict(torch.load('model_siamese.pth', map_location=device))
    model.eval()

    # 获取上传的文件
    img1 = request.files['image1']
    img2 = request.files['image2']
    img1 = Image.open(img1)
    img2 = Image.open(img2)
    img1 = img1.convert('L')
    img2 = img2.convert('L')

    # 进行图像处理
    transform = transforms.Compose([
        transforms.Resize((160, 160)),  # 调整图片大小
        transforms.ToTensor()  # 转换为 Tensor 格式
    ])
    img1 = transform(img1)
    img2 = transform(img2)
    print(img1)
    ans = model(img1, img2)
    print(ans)
    if ans > 0.5:
        result = '两个签名属于同一个人'
    else:
        result = '两个签名不属于同一个人'

    # 将处理后的结果返回到前端页面
    return render_template('test.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
