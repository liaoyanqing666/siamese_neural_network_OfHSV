import numpy

from dataset import *
from model import *
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


if __name__ == '__main__':
    test_size = 1000
    batch_size = 40
    test_data = dataset(train=False, max_num=test_size)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = siamese()
    model.load_state_dict(torch.load('model_siamese.pth', map_location=device))
    model.eval()

    y_pred = []
    y_test = []
    # 对测试数据进行预测
    for first, second, label in test_loader:
        ans = model(first, second)
        y_test += label.detach().numpy().tolist()
        y_pred += ans.detach().numpy().tolist()

    # 计算假正率、真正率和阈值
    print(y_test)
    print(y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    # 计算曲线下面积
    roc_auc = auc(fpr, tpr)

    # 绘制 ROC 曲线
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.10f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # 输出 AUC
    print("AUC:", roc_auc)
