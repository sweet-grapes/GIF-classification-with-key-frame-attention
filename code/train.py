import argparse
import shutil
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F
from dataset.training_set import TrainingSet
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
    parser.add_argument('data', type=str, help='path to dataset')
    parser.add_argument('save', type=str, help='path to model saved')
    parser.add_argument('--batch_size', type=int, default=48)
    parser.add_argument('--num_frames', type=int, default=8)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--classes', type=int, default=73)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--print_freq', type=int, default=10)

    args = parser.parse_args()

    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.makedirs(args.save, exist_ok=True)

    # Load data
    training_set_path = os.path.join(args.data, 'training_gif')
    validation_set_path = os.path.join(args.data, 'validation_gif')

    training_dataset = TrainingSet(training_set_path, num_frames=args.num_frames)
    training_dataset_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)

    eval_training_dataset = EvaluationSet(training_set_path, num_frames=args.num_frames)
    eval_training_dataset_loader = DataLoader(eval_training_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    eval_validation_dataset = EvaluationSet(validation_set_path, num_frames=args.num_frames)
    eval_validation_dataset_loader = DataLoader(eval_validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    # Load model
    net = DCEHAAModel(backbone='resnet50', num_classes=args.classes, num_frames=args.num_frames)

    # Use multiple GPUs
    net = nn.DataParallel(net).cuda()

    # Define loss function (criterion) and optimizer
    criterion_1 = nn.CrossEntropyLoss()
    criterion_2 = EntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # Decay learning rate
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3333)

    writer = SummaryWriter('experiment_results_curve')
    best_acc1_validation_set = 0.0

    for epoch in range(args.epochs):

        # Train for one epoch
        train(training_dataset_loader, net, criterion_1, criterion_2, optimizer, epoch, args)

        # Evaluate on training_set
        loss_training_set, acc1_training_set, acc3_training_set = evaluation(eval_training_dataset_loader, net, criterion_1, criterion_2)
        print('epoch:{0:<5d}loss of training_set:{1:<10.3f}top1 accuracy of training_set:{2:<10.2%}top3 accuracy of training_set:{3:<10.2%}'.format(epoch + 1, loss_training_set, acc1_training_set, acc3_training_set))
        # Evaluate on validation_set
        loss_validation_set, acc1_validation_set, acc3_validation_set = evaluation(eval_validation_dataset_loader, net, criterion_1, criterion_2)
        print('epoch:{0:<5d}loss of validation_set:{1:<8.3f}top1 accuracy of validation_set:{2:<8.2%}top3 accuracy of validation_set:{3:<8.2%}\n'.format(epoch + 1, loss_validation_set, acc1_validation_set, acc3_validation_set))
        scheduler.step(loss_training_set)

        # Visualize the loss and accuracy of the training_set and validation_set
        writer.add_scalars('Loss', {'training_set': loss_training_set, 'validation_set': loss_validation_set}, epoch)
        writer.add_scalars('Top1 accuracy', {'training_set': acc1_training_set, 'validation_set': acc1_validation_set}, epoch)
        writer.add_scalars('Top3 accuracy', {'training_set': acc3_training_set, 'validation_set': acc3_validation_set}, epoch)

        # Save top1 model on validation_set
        if acc1_validation_set > best_acc1_validation_set:
            best_acc1_validation_set = acc1_validation_set
            torch.save(net.module.state_dict(), os.path.join(args.save, 'best_top1.pth'))  # Existing will be overwritten
    # Save final model
    torch.save(net.module.state_dict(), os.path.join(args.save, 'latest.pth'))
    writer.close()


def train(training_dataset_loader, net, criterion_1, criterion_2, optimizer, epoch, args):
    net.train()
    running_loss = 0.0
    running_corr1 = 0
    running_corr3 = 0
    running_total = 0
    for i, (images, labels) in enumerate(training_dataset_loader):
        images = images.cuda()
        labels = labels.cuda()

        B, T, _, _, _ = images.shape
        # Compute batch outputs and loss
        outputs, atten, frame_preds = net(images)
        loss1 = criterion_1(outputs, labels)
        loss2 = criterion_1(frame_preds, labels.view(B, 1).expand(B, T).reshape(B*T))
        loss3 = criterion_2(atten)
        loss = 0.8 * loss1 + 0.2 * loss2 + 0.02 * loss3
        # Compute the average loss and accuracy of args.print_freq batches
        running_loss += loss.item()
        corr1, corr3 = corrects(outputs, labels, topk=(1, 3))
        running_corr1 += corr1.item()
        running_corr3 += corr3.item()
        running_total += labels.size(0)
        if i % args.print_freq == (args.print_freq-1):
            print('epoch:{:<5d}batch:{:<5d}train loss:{:<10.3f}train top1 acc:{:<10.2%}train top3 acc:{:<10.2%}'.format(epoch + 1, i + 1, running_loss / args.print_freq, running_corr1 / running_total, running_corr3 / running_total))
            running_loss = 0.0
            running_corr1 = 0
            running_corr3 = 0
            running_total = 0

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


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
        # total = labels.size(0)
        _, pred = torch.topk(outputs, maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            # res.append(correct_k / total)
            res.append(correct_k)
        return res


if __name__ == '__main__':
    np.random.seed(1)
    torch.manual_seed(1)
    main()
