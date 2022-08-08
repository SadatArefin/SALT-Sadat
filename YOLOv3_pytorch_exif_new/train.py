import argparse
import time

import test  # Import test.py to get mAP after each epoch
import test_on_train
from models import *
from utils.datasets import *
from utils.utils import *


def train(
        cfg,
        data_cfg,
        img_size=416,
        resume=False,   # 重新开始
        epochs=10,
        batch_size=16,
        accumulate=1,
        multi_scale=False,
        freeze_backbone=False,      # TODO All layer parameters learn to update?
):
    weights = 'weights' + os.sep    # path: weights\
    latest = weights + 'latest.pt'  # path: weights\latest.pt
    best = weights + 'best.pt'      # path: weights\best.pt
    device = torch_utils.select_device()

    if multi_scale:
        print('enable multi-scalse training')
        img_size = 608  # initiate with maximum multi_scale size
    else:
        torch.backends.cudnn.benchmark = True  # No multi-scale training

    # Configure to run
    train_path = parse_data_cfg(data_cfg)['train']  # train_path= data\train.txt，path to save training set

    # Initialize model
    model = Darknet(cfg, img_size)  # Load network model

    # Get dataloader
    dataloader = LoadImagesAndLabelsAndExif(train_path, batch_size, img_size, augment=True)    # create dataloader，It contains the training set，label，batch, other information

    lr0 = 0.0001  # initial learning rate
    cutoff = -1  # don't want darknet53 last layer? backbone reaches to cutoff layer
    start_epoch = 0
    best_loss = float('inf')
    if resume:      # If you train from scratch
        print('start from scratch')
        print(torch.cuda.get_device_name(0))
        # print(torch.cuda.get_device_name(1))
        # checkpoint = torch.load(latest, map_location='cpu')
        # checkpoint = torch.load(latest, map_location=torch.device(0))

        # Load weights to resume from
        # model.load_state_dict(checkpoint['model'])

        # Transfer learning (train only YOLO layers)
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     p.requires_grad = True if (p.shape[0] == 255) else False

        # Set optimizer
        optimizer = torch.optim.SGD(filter(lambda x: x.requires_grad, model.parameters()), lr=lr0, momentum=.9)

        # start_epoch = checkpoint['epoch'] + 1
        # if checkpoint['optimizer'] is not None:
        # # if checkpoint['optimizer'] is None:
        #     print('using checkpoint optimizer')
        #     optimizer.load_state_dict(checkpoint['optimizer'])
        #     best_loss = checkpoint['best_loss']

        # del checkpoint  # current, saved

    else:   # if not from scratch，transfer learning，Load pretrained parameters：
        # Initialize model with backbone (optional)
        if cfg.endswith('yolov3.cfg'):
            print('using darknet53.conv.74')
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')
        if cfg.endswith('yolov3-1cls.cfg'):
            print('using darknet53.conv.74')
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')
        if cfg.endswith('yolov3-spp-2cls.cfg'):
            print('using darknet53.conv.74')
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')
        if cfg.endswith('yolov3-spp-1cls.cfg'):
            print('using darknet53.conv.74')
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')
        if cfg.endswith('yolov3-2cls.cfg'):
            print('using darknet53.conv.74')
            cutoff = load_darknet_weights(model, weights + 'darknet53.conv.74')
        if cfg.endswith('yolov3-tiny-2cls.cfg'):
            print('using yolov3-tiny.conv.15')
            cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')
        elif cfg.endswith('yolov3-tiny.cfg'):   # corresponding model，Load model weights and biases
            print('using yolov3-tiny.conv.15')
            cutoff = load_darknet_weights(model, weights + 'yolov3-tiny.conv.15')

        # Set optimizer
        optimizer = torch.optim.SGD(model.parameters(), lr=lr0, momentum=.9)    # tomodelofparameters()useSGDlearning optimizer

    if torch.cuda.device_count() > 1:   # parallel training
        model = nn.DataParallel(model)
    model.to(device).train()

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[54, 61], gamma=0.1)

    # Start training
    t0 = time.time()
    loss_list = []  # for drawing
    # model_info(model)     # Model information
    n_burnin = min(round(dataloader.nB / 5 + 1), 1000)  # Update for learning rate number of burn-in batches
    for epoch in range(epochs):
        model.train()
        epoch += start_epoch    # from the very beginning epoch+this time（if previously trained）

        print(('\n%8s%12s' + '%10s' * 7) % (
            'Epoch', 'Batch', 'xy', 'wh', 'conf', 'cls', 'total', 'nTargets', 'time'))

        # Update scheduler (automatic)
        # scheduler.step()

        # Update scheduler (manual)
        if epoch > 250:     # 250 indivual epoch The learning rate decays to 0.0001
            lr = lr0 / 10
        else:
            lr = lr0
        for x in optimizer.param_groups:    # x is the training parameter of the optimizer
            x['lr'] = lr

        # Freeze backbone parameters，no update（Only learn the yolo classification layer）（Freeze at epoch 0, Thaw at epoch 1）
        if freeze_backbone and epoch < 2:
            for i, (name, p) in enumerate(model.named_parameters()):
                if int(name.split('.')[1]) < cutoff:  # 冻结Darknet的层0-cutoff，只学习分类层参数 （yolo是75层，yolo tiny是15）
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        rloss = defaultdict(float)

        # Revise
        for i, (imgs, targets, exifs, _, _) in enumerate(dataloader):
            # imgs：4*3*416*416,（batch=4）    targets：Number of targets*6 6 represents：[Which picture, class, 4 positions]
            targets = targets.to(device)
            nT = targets.shape[0]   # number of targets
            if nT == 0:  # if no targets continue
                continue

            #lr starts from 0 and gradually increases to lr0，no change after(Start too big and spread out，But don't worry about transfer learning） （However, lr should start bigger and get smaller.）
            if (epoch == 0) and (i <= n_burnin):
                lr = lr0 * (i / n_burnin) ** 4
                # print(lr)
                for x in optimizer.param_groups:
                    x['lr'] = lr
            # Run model, Here is an example of yolo_tiny，it has two scales
            # pred：2 tuples, corresponding to 2 scales： [4,3,13,13,85] , [4,3,26,26,85]

            # Revise
            exifs = exifs.to(device)
            # print('Test if output exifs are single：', exifs)
            pred = model(imgs.to(device), exifs)

            # Build targets，Convert training needs format，Select positive and negative samples。
            # Contains 5 tuples，for txy, twh, tcls, tconf, indices
            # Revise
            target_list = build_targets(model, targets, pred)

            # calculate loss
            loss, loss_dict = compute_loss(pred, target_list)   # loss.Size=1
            loss_list.append(loss.item())

            # Compute gradient
            loss.backward()

            # How many times the gradient is accumulated for a parameter update Accumulate gradient for x batches before optimizing
            if (i + 1) % accumulate == 0 or (i + 1) == len(dataloader):
                optimizer.step()
                optimizer.zero_grad()   # 梯度清零

            # Calculate the average error per epoch and print  Running epoch-means of tracked metrics
            ui += 1
            for key, val in loss_dict.items():
                rloss[key] = (rloss[key] * ui + val) / (ui + 1)

            s = ('%8s%12s' + '%10.3g' * 7) % (
                '%g/%g' % (epoch, epochs - 1),
                '%g/%g' % (i, len(dataloader) - 1),
                rloss['xy'], rloss['wh'], rloss['conf'],
                rloss['cls'], rloss['total'],
                nT, time.time() - t0)
            t0 = time.time()
            print(s)

            # Multi-Scale training (320 - 608 pixels) every 10 batches
            if multi_scale and (i + 1) % 10 == 0:
                dataloader.img_size = random.choice(range(10, 20)) * 32
                # print('multi_scale img_size = %g' % dataloader.img_size)

        # Update best loss,The validation set should be used here to verify it to prevent overfitting，But in fact the entire program validation set is not used
        if rloss['total'] < best_loss:
            best_loss = rloss['total']

        # save training results
        save = True
        if save:
            # Save latest checkpoint
            checkpoint = {'epoch': epoch,
                          'best_loss': best_loss,
                          'model': model.module.state_dict() if type(model) is nn.DataParallel else model.state_dict(),
                          'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, latest)

            # Save best checkpoint
            if best_loss == rloss['total']:
                torch.save(checkpoint, best)

            # Save backup weights every 5 epochs (optional)
            if epoch > 0 and epoch % 5 == 0:
                torch.save(checkpoint, weights + 'backup%g.pt' % epoch)

        # Calculate mAP In fact, mAP is useless in the network (use test set)
        with torch.no_grad():   # TODO Look, it doesn't seem right, precision and recall are synchronized?
            P, R, mAP = test.test(cfg, data_cfg, weights=latest, batch_size=batch_size, img_size=img_size, model=model)
            P_train, R_train, mAP_train = test_on_train.test_on_train(cfg, data_cfg, weights=latest, batch_size=batch_size, img_size=img_size, model=model)


        # Write epoch results
        with open('results.txt', 'a') as file:
            file.write(s + '%11.3g' * 3 % (P, R, mAP) + '\n')
    return loss_list

# Remember to change the class and convolutional layer output of cfg for migration learning 3*(4+4+class/type)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='size of each image batch')
    parser.add_argument('--accumulate', type=int, default=1, help='accumulate gradient x batches before optimizing')
    parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp-1cls.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='data/animals.data', help='coco.data file path')  # TODO 改
    parser.add_argument('--multi-scale', action='store_false', help='random image sizes per batch 320 - 608')
    parser.add_argument('--img-size', type=int, default=32 * 13, help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    opt = parser.parse_args()
    print(opt, end='\n\n')
    print(opt.resume)

    init_seeds()


    loss_list = train(
                        opt.cfg,
                        opt.data_cfg,
                        img_size=opt.img_size,
                        resume=opt.resume,
                        epochs=opt.epochs,
                        batch_size=opt.batch_size,
                        accumulate=opt.accumulate,
                        multi_scale=opt.multi_scale,
                    )
