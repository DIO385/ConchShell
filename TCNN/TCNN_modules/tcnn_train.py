import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import argparse
import os
from tcnn_module import TCNN_ablation
from tcnn_data_loader import TCNNDataset
from tqdm import tqdm

def plot_loss_and_acc(train_loss, val_loss, val_acc, epoch):
    """ Draw loss curve and accuracy curve. """
    if not os.path.exists(args.save_loss_img_path):
        os.mkdir(args.save_loss_img_path)

    # Save the loss curve.
    plt.plot(train_loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.legend(loc='best')
    plt.ylabel('loss', fontsize=12)
    plt.xlabel('epoch', fontsize=12)
    plt.title("Train-Loss [VS] Val-Loss")
    plt.savefig(f'{args.save_loss_img_path}/tcnn_net18_loss-epoch={epoch}.jpg')
    # plt.show()
    plt.clf()

    # Save the accuracy curve.
    plt.title("tcnn_acc")
    plt.plot(val_acc, label='acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig(f'{args.save_loss_img_path}/tcnn_net18_acc-epoch={epoch}.jpg')
    # plt.show()
    plt.clf()


def logs(t_loss, v_loss, acc, epoch):
    """ log """
    if not os.path.exists(args.save_log_path):
        os.mkdir(args.save_log_path)

    log = f"Epoch:{epoch} \t train_loss:%.4f \t val_loss:%.4f \t acc:%.2f%%\t Time:{time.asctime()}\n" % (
        t_loss, v_loss, acc * 100)
    log_path = os.path.join(args.save_log_path, f"TCNN_net18_log.txt")
    with open(log_path, mode="a", encoding="utf8") as log_file:
        log_file.write(log)
        log_file.close()


def val(net, val_loader, loss_fn):
    """ val function """
    net.eval()
    loss, acc, n, num = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for batch, (val_image, val_label, _) in enumerate(val_loader):
            val_image = val_image.to(device)
            val_label = val_label.to(device)

            outputs = net.forward(val_image)
            cur_loss = loss_fn(outputs, val_label)

            result = torch.max(outputs, dim=1)[1]

            acc += sum(result == val_label).item()

            _loss = cur_loss.item()
            loss += _loss
            n += 1
            num += val_image.shape[0]

        val_loss = loss / n
        val_acc = acc / num

        print(f'val_loss:{val_loss}\tval_acc:{val_acc}')
    return val_loss, val_acc


def train(net, train_loader, optim, loss_fun, device):
    """ training function """
    net.train()
    loss, n = 0.0, 0
    for step, (image, label, _) in enumerate(tqdm(train_loader)):
        image = image.to(device)
        label = label.to(device)  # [B, 6]

        optimizer.zero_grad()
        outputs = net.forward(image)  # [B, 6]
        cur_loss = loss_fun(outputs, label)

        cur_loss.backward()
        optim.step()
        _loss = cur_loss.item()
        loss += _loss
        n += 1
        # print(f"{step + 1}/{len(train_loader)}\tloss={_loss}")

    train_loss = loss / n

    print(f'train_loss:{train_loss}')
    return train_loss


def train_main(net, train_loader, val_loader, optim, loss_fun, device, args):
    """ Training main function, including training, testing, drawing, logging, saving models and so on.
    :param net: TCNN network
    :param optim: Adam
    :param loss_fun: CrossEntropyLoss
    """
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)
    torch.backends.cudnn.benchmark = True
    loss_train, loss_val, acc_list = [], [], []
    print("===========Start training=============")
    for epoch in range(args.epochs):
        print(f"epoch = {epoch + 1}")
        # training
        train_loss = train(net=net,
                           train_loader=train_loader,
                           optim=optim,
                           loss_fun=loss_fun,
                           device=device)
        # verification
        val_loss, val_acc = val(net, val_loader, loss_fun)

        loss_train.append(train_loss)
        loss_val.append(val_loss)
        acc_list.append(val_acc)

        # plot loss img(default 100 epochs)
        if epoch % args.plot_loss_interval == 0:
            plot_loss_and_acc(loss_train, loss_val, acc_list, epoch + 1)

        # Whether to log (logging is for each round).
        if args.is_log:
            logs(train_loss, val_loss, val_acc, epoch + 1)

        # Save model parameters (default 100 epochs)
        if epoch % args.model_interval == 0:
            model_path = os.path.join(args.save_model_path, f"tcnn__epoch={epoch + 1}.pth")
            torch.save(net.state_dict(), model_path)

    print("===========Fininsh training=============")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.00005)
    parser.add_argument("--model_interval", type=int, default=50)
    parser.add_argument("--plot_loss_interval", type=int, default=50)

    parser.add_argument("--is_log", type=bool, default=False)

    parser.add_argument("--save_model_path", default='../TCNN_models')
    parser.add_argument("--save_loss_img_path", default='../TCNN_loss_img')
    parser.add_argument("--save_log_path", default='../TCNN_logs')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # path of your data set.
    TRAIN_DATA_SET = r"../../dataset/Img/train_set"
    VAL_DATA_SET = r"../../dataset/Img/val_set"

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"using device：{device}")

    # Initialize data loader.
    train_data_set = TCNNDataset(x_img_path=TRAIN_DATA_SET, data_type="train")
    train_loader = DataLoader(train_data_set, batch_size=args.batch_size, num_workers=8, shuffle=True)

    val_data_set = TCNNDataset(x_img_path=VAL_DATA_SET, data_type="val")
    val_loader = DataLoader(val_data_set, batch_size=1, num_workers=8, shuffle=False)
    print("Data loading completed.")

    # Initialize model, optimizer, loss function.
    model = TCNN_ablation().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.99, 0.999))
    # loss_function = nn.L1Loss()
    loss_function = nn.CrossEntropyLoss()

    print("Model initialization completed, training started.")
    train_main(model, train_loader, val_loader, optimizer, loss_function, device, args)
