from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize
import torch

class dataset(Dataset):
    def __init__(self, root='CEDAR/', train=True, max_num=100000000):
        super(dataset, self).__init__()
        if train:
            path = root + "gray_train.txt"
        else:
            path = root + "gray_test.txt"

        self.datas = []
        self.labels = []

        with open(path, 'r') as f:
            lines = f.readlines()

        print("读取{}数据中...共{}行".format('训练' if train else '测试', len(lines)))
        h = 160 # 最好是32的倍数，因为VGG16中maxpooling的padding是0
        w = 160
        now_line = 0
        totensor = Compose([Resize((h, w)), ToTensor()])
        for line in lines:
            first, second, label = line.split()
            first_image = Image.open(root + first)
            first_image = first_image.convert('L')
            first_image = totensor(first_image)
            second_image = Image.open(root + second)
            second_image = second_image.convert('L')
            second_image = totensor(second_image)
            self.datas.append([first_image, second_image])
            self.labels.append(int(label))
            now_line += 1
            if (now_line % 100 == 0):
                print("当前读取到{}行".format(now_line))
            if (now_line == max_num):
                break

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.FloatTensor(self.datas[index][0]), torch.FloatTensor(self.datas[index][1]), torch.tensor(self.labels[index], dtype=torch.float32)


if __name__ == "__main__":
    train_test = dataset(max_num=400)
    train_loader = DataLoader(train_test, batch_size=16, shuffle=False)
    for i, (first, second, label) in enumerate(train_loader):
        print(first.shape)
        print("first", first)
        print("second", second)
        print("label{}".format(label))