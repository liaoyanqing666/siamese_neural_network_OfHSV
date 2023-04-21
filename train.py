import torch
from dataset import dataset
from model import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn import BCELoss
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.set_printoptions(precision=20)

def accuracy(outputs, labels):
    right = 0
    outputs[outputs > 0.5] = 1
    outputs[outputs <= 0.5] = 0
    for i in range(len(labels)):
        if outputs[i] == labels[i]:
            right +=1
    return right


# 初始化tensorboard
writer = SummaryWriter()

batch_size = 20
learning_rate = 0.01
epochs = 200

if torch.cuda.is_available():
    cuda = 1
    print("GPU")
else:
    cuda = 0
    print("CPU")

train_set = dataset(train=True)
test_set = dataset(train=False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

model = siamese()
if cuda:
    model = model.cuda()

loss = BCELoss(reduction='mean')
if cuda:
    loss = loss.cuda() # 损失函数转移到cuda

# 优化器
optim = torch.optim.SGD(model.parameters(),lr=learning_rate)
# 开始训练
total_train_step = 0
for i in range(epochs):
    print("-----第{}轮训练开始-----".format(i+1))

    # 网络训练
    for now_train_step, (first_imgs, second_imgs, labels) in enumerate(train_loader):
        # 数据转移到cuda
        if cuda:
            first_imgs = first_imgs.cuda()
            second_imgs = second_imgs.cuda()
            labels = labels.cuda()
        output = model(first_imgs, second_imgs)
        # 求损失
        now_loss = loss(output, labels)
        # 梯度清零
        optim.zero_grad()
        # 反向传播
        now_loss.backward()
        # 优化
        optim.step()

        total_train_step += 1
        if (now_train_step % 300 == 0):
            print("第{}轮训练第{}次迭代，训练损失函数{}".format(i+1, now_train_step, now_loss))
            writer.add_scalar("train_loss (per 300 iterations)", now_loss, total_train_step)

    # 网络测试
    total_test_loss = 0.0
    total = 0
    total_right = 0
    with torch.no_grad():
        for now_test_step, (first_imgs, second_imgs, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                first_imgs = first_imgs.cuda()
                second_imgs = second_imgs.cuda()
                labels = labels.cuda()
            output = model(first_imgs, second_imgs)
            if (now_test_step == 0):
                print(output[0],output[1])
            # 求总损失函数
            now_loss = loss(output, labels)
            total_test_loss += now_loss
            # 求正确个数
            right = accuracy(output, labels)
            total += len(labels)
            total_right += right
    print("第{}轮训练，测试集总损失函数{}".format(i+1, total_test_loss))
    print("第{}轮训练，测试集总正确率{}".format(i+1, total_right/total))
    writer.add_scalar("test_loss", total_test_loss, i)
    writer.add_scalar("test_accuracy", total_right/total, i)
    torch.save(model.state_dict(),'model_siamese{}.pth'.format(i + 1))
