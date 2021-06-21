import argparse
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import torch.nn.functional as F
from dataset.evaluation_set import EvaluationSet
from model.resnet50_attention import DCEHAAModel


# Define entropy loss
class EntropyLoss(nn.Module):
    def __init__(self, tau=0.1):
        super(EntropyLoss, self).__init__()
        self.tau = tau

    def forward(self, x):
        x = x/self.tau
        h = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        h = -1.0 * h.sum()
        h = h / x.shape[0]
        return h


def main():
    parser = argparse.ArgumentParser(description='PyTorch GIF Classification')
    parser.add_argument('data',  type=str, help='path to dataset')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--classes', type=int, default=73)
    args = parser.parse_args()

    # Data loading code--
    test_set_path = os.path.join(args.data, 'test_gif')
    eval_test_dataset = EvaluationSet(test_set_path, num_frames=args.num_frames)
    eval_test_dataset_loader = DataLoader(eval_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Loading pretrained model
    net = DCEHAAModel(backbone='resnet50', num_classes=args.classes, num_frames=args.num_frames)
    state_dict = torch.load('pretrained/best_top1.pth')
    net.load_state_dict(state_dict)
    # Use multiple GPUs
    net = nn.DataParallel(net).cuda()

    # Define loss function (criterion) and optimizer
    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = EntropyLoss()

    # Evaluate on test set
    loss_test_set, acc1_test_set, acc3_test_set = evaluation(eval_test_dataset_loader, net, criterion_1, criterion_2)
    print('loss of test_set:{0:<9.3f}top1 accuracy of test_set:{1:<9.2%}top3 accuracy of test_set:{2:<9.2%}\n'.format(loss_test_set, acc1_test_set, acc3_test_set))


def evaluation(eval_dataset_loader, net, criterion_1, criterion_2):
    net.eval()
    eval_loss = 0.0
    eval_corr1 = 0
    eval_corr3 = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(eval_dataset_loader):
            images = images.cuda()
            labels = labels.cuda()

            B, T, _, _, _ = images.shape
            # Compute batch outputs and loss
            outputs, atten, frame_preds = net(images)
            loss1 = criterion_1(outputs, labels)
            loss2 = criterion_1(frame_preds, labels.view(B, 1).expand(B, T).reshape(B * T))
            loss3 = criterion_2(atten)
            loss = 0.8 * loss1 + 0.2 * loss2 + 0.02 * loss3
            # Compute loss and accuracy
            eval_loss += loss.item()
            corr1, corr3 = corrects(outputs, labels, topk=(1, 3))
            eval_corr1 += corr1.item()
            eval_corr3 += corr3.item()
            total += labels.size(0)
        return eval_loss / (i + 1), eval_corr1 / total, eval_corr3 / total


def corrects(outputs, labels, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        _, pred = torch.topk(outputs, maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k)
        return res


if __name__ == '__main__':
    torch.manual_seed(1)
    main()
