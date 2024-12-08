import argparse
import copy
import json
import os

from IPython.core.display import display

from BH_LymphDS.custom import metrics

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import shutil
import time

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils as utils
import torch.utils.data
import torch.utils.data.distributed
from torch import Tensor
import numpy as np

from BH_LymphDS.custom.ClassificationDataset import create_classification_dataset
from BH_LymphDS.core import create_model, create_optimizer, create_lr_scheduler
from BH_LymphDS.core import create_standard_image_transformer
from BH_LymphDS.core.losses_factory import create_losses
from BH_LymphDS.custom import create_dir_if_not_exists, truncate_dir
from BH_LymphDS.custom.about_log import ColorPrinter
from BH_LymphDS.custom.about_log import logger, log_long_str
from BH_LymphDS import get_param_in_cwd


def metric_details(log_file, metric_spec, epoch, subset='',
                   metric_spec_agg='mean', metric_spec_ids=None):
    log = pd.read_csv(log_file, names=['fname', 'pred_score', 'pred_label', 'gt'], sep='\t')
    ul_labels = np.unique(log['pred_label'])
    metric_results = []
    if metric_spec_ids is None:
        if len(ul_labels) > 2:
            metric_spec_ids = list(ul_labels)
        else:
            metric_spec_ids = [1]
    elif not isinstance(metric_spec_ids, (list, tuple)):
        metric_spec_ids = [metric_spec_ids]
    for ul in metric_spec_ids:
        pred_score = list(map(lambda x: x[0] if x[1] == ul else 1 - x[0],
                              np.array(log[['pred_score', 'pred_label']])))
        gt = [1 if gt_ == ul else 0 for gt_ in np.array(log['gt'])]
        acc, auc, ci, tpr, tnr, ppv, npv, _, _, _, thres = metrics.analysis_pred_binary(gt, pred_score)
        ci = f"{ci[0]:.4f}-{ci[1]:.4f}"
        metric_results.append([epoch, ul, acc, auc, ci, tpr, tnr, ppv, npv, thres, subset])
    metric_results = pd.DataFrame(metric_results,
                                  columns=['Epoch', 'SpecID', 'Acc', 'AUC', '95% CI', 'Sensitivity', 'Specificity',
                                           'PPV', 'NPV', 'Threshold', 'Cohort'])
    bst_metric = metric_results.describe()[metric_spec][metric_spec_agg]
    return bst_metric, metric_results


def train_model(model, device, dataloaders, batch_size, criterion, optimizer, lr_scheduler, num_epochs=30,
                iters_start=0, iters_verbose=1000, save_dir='./', is_inception=False, save_per_epoch=False, **kwargs):
    """
    Train and valid core.

    :param model: The core generate from `create_model`.
    :param device: Which device to be used!
    :param dataloaders: Dataset loaders, `train` and `valid` can be specified.
    :param batch_size: Batch size.
    :param criterion: Loss function definition. Generated from `create_losses`
    :param optimizer: Optimizer. Generated from `create_optimizer`
    :param lr_scheduler: Learning scheduler. Generated from `create_lr_scheduler`
    :param num_epochs: Number of epochs
    :param iters_start: Iters start, default 0
    :param iters_verbose: Iters to display log. default 10.
    :param save_dir: Where to save core.
    :param is_inception: Is inception core, for different image resolution.
    :param save_per_epoch: 是否每个Epoch都保存。
    :return: core parameters, validation accuracy.
    """
    time_since = time.time()
    time_verbose = time.time()
    valid_time_since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = None
    iters_done = 0
    cp = ColorPrinter()

    train_dir = truncate_dir(os.path.join(save_dir, 'train'), del_directly=True)
    valid_dir = truncate_dir(os.path.join(save_dir, 'valid'), del_directly=True)
    viz_dir = truncate_dir(os.path.join(save_dir, 'viz'), del_directly=True)
    train_log = [('Epoch', 'Iters', 'Loss', 'Acc@1')]
    valid_log = [('Epoch', 'Iters', 'acc')]
    train_file = None
    valid_file = None
    train_file_spec = None
    valid_file_spec = None

    num_classes = kwargs['task_spec']['num_classes']
    metric_spec = get_param_in_cwd('metric_spec', 'acc')
    metric_spec_ids = get_param_in_cwd('metric_spec_ids', None)
    metric_spec_agg = get_param_in_cwd('metric_spec_agg', 'mean')
    display_metric = get_param_in_cwd('display_metric', False)
    for epoch in range(num_epochs):
        epoch_iters_done = 0
        # Each epoch has a training and validation phase
        metric_results_ = []
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set core to training mode
                train_file = open(os.path.join(train_dir, f'Epoch-{epoch}.txt'), 'w')
                train_file_spec = open(os.path.join(train_dir, f'Epoch-{epoch}_spec.csv'), 'w')
                print(f'fpath,{",".join([f"label-{iid}" for iid in range(num_classes)])}', file=train_file_spec)
            else:
                model.eval()  # Set core to evaluate mode
                valid_time_since = time.time()
                valid_file = open(os.path.join(valid_dir, f'Epoch-{epoch}.txt'), 'w')
                valid_file_spec = open(os.path.join(valid_dir, f'Epoch-{epoch}_spec.csv'), 'w')
                print(f'fpath,{",".join([f"label-{iid}" for iid in range(num_classes)])}', file=valid_file_spec)

            running_loss = 0.0
            running_corrects = 0
            running_size = 0
            # Iterate over data.
            for inputs, labels, fnames in dataloaders[phase]:
                inputs = inputs.to(device)
                input_size = inputs.shape[0]
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    # Multi head outputs for inception.
                    if is_inception and phase == 'train':
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4 * loss2
                    else:
                        outputs = model(inputs)
                        # outputs = outputs[0] # For LeVit
                        if not isinstance(outputs, Tensor):
                            loss1 = criterion(outputs.logits, labels)
                            loss2 = criterion(outputs.aux_logits2, labels)
                            loss3 = criterion(outputs.aux_logits1, labels)
                            loss = loss1 + 0.4 * loss2 + 0.2 * loss3
                        else:
                            loss = criterion(outputs, labels)
                    if isinstance(outputs, Tensor):
                        probabilities = nn.functional.softmax(outputs, dim=1)
                        probability, predictions = torch.max(probabilities, 1)
                    else:
                        probabilities = nn.functional.softmax(outputs.logits, dim=1)
                        probability, predictions = torch.max(probabilities, 1)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        iters_done += 1
                        epoch_iters_done += 1
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                        for fname, probs, prob, pred, label in zip(fnames, probabilities, probability, predictions,
                                                                   labels):
                            train_file.write('%s\t%.3f\t%d\t%d\n' % (fname, prob, pred, label))
                            train_file_spec.write('%s,%s\n' % (fname, ','.join(map(str, probs.detach().cpu().numpy()))))
                    else:
                        for fname, probs, prob, pred, label in zip(fnames, probabilities, probability, predictions,
                                                                   labels):
                            valid_file.write('%s\t%.3f\t%d\t%d\n' % (fname, prob, pred, label))
                            valid_file_spec.write('%s,%s\n' % (fname, ','.join(map(str, probs.detach().cpu().numpy()))))
                running_loss += loss.item()
                running_corrects += int(torch.sum(predictions == labels.data))
                running_size += input_size
                if epoch_iters_done % iters_verbose == 0 and phase == 'train':
                    iters_epoch = (iters_done + iters_start) * batch_size / len(dataloaders[phase].dataset)
                    total_time = int(time.time() - time_since)
                    speed_info = iters_verbose * input_size / (time.time() - time_verbose)
                    hours_ = total_time // 3600
                    minutes_ = total_time // 60 % 60
                    acc_ = running_corrects * 100 / running_size
                    info = 'Phase: {phase}\tEpoch: {epoch}\tLR: {lr}\tLoss: {loss}\tAcc: {acc}\t ' \
                           'Speed: {speed} img/s\tTime: {time}'
                    logger.info(info.format(phase=cp.color_text(phase, color='green' if phase == 'train' else 'cyan'),
                                            epoch=cp.color_text('%.3f,%d' % (iters_epoch, iters_done), 'magenta'),
                                            lr=cp.color_text(','.join(['%.6f' % lr_ for
                                                                       lr_ in lr_scheduler.get_last_lr()]), 'cyan'),
                                            loss=cp.color_text('%.3f' % (running_loss / iters_verbose), 'red'),
                                            acc=cp.color_text('%.4f' % acc_, 'green' if acc_ > 70 else 'yellow'),
                                            speed=cp.color_text('%.2f' % speed_info, 'yellow'),
                                            time=cp.color_text(f'{hours_}hrs:{minutes_}min')))
                    train_log.append((iters_epoch, iters_done, running_loss / iters_verbose, acc_))
                    time_verbose = time.time()
                    running_loss = 0.0
                    running_corrects = 0
                    running_size = 0

            epoch_acc = running_corrects * 100 / len(dataloaders[phase].dataset)
            epoch_metric = epoch_acc
            if phase == 'train':
                train_file.close()
                train_file_spec.close()
                if metric_spec.lower() != 'acc':
                    _, metric_results = metric_details(os.path.join(train_dir, f'Epoch-{epoch}.txt'),
                                                       metric_spec, epoch, 'Train',
                                                       metric_spec_agg, metric_spec_ids)
                    metric_results_.append(metric_results)
                if save_per_epoch:
                    torch.save({'global_step': iters_done + iters_start,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'lr_scheduler_state_dict': lr_scheduler.state_dict()},
                               os.path.join(train_dir, f'training-params-{iters_done + iters_start}.pth'))
            else:
                valid_file.close()
                valid_file_spec.close()
                if metric_spec.lower() != 'acc':
                    epoch_metric, metric_results = metric_details(os.path.join(valid_dir, f'Epoch-{epoch}.txt'),
                                                                  metric_spec, epoch, 'Test',
                                                                  metric_spec_agg, metric_spec_ids)
                    metric_results_.append(metric_results)
                val_acc_history.append(epoch_metric)

                info = 'Phase: {phase}\t{metric}: {acc}\tSpeed: {speed}img/s\tTime: {time}s'
                speed = len(dataloaders[phase].dataset) / (time.time() - valid_time_since)
                logger.info(info.format(phase=cp.color_text('valid', 'cyan'),
                                        metric=metric_spec,
                                        acc=cp.color_text("%.3f" % epoch_metric),
                                        speed=cp.color_text("{:.4f}".format(speed), 'red'),
                                        time=cp.color_text('%.2f' % (time.time() - valid_time_since))))
                valid_log.append((iters_epoch, iters_done, epoch_metric))
        if display_metric and metric_results_:
            display(pd.concat(metric_results_, axis=0))
    # Save training log to csv
    pd.DataFrame(train_log).to_csv(os.path.join(viz_dir, 'training_log.txt'),
                                   header=False, index=False, encoding='utf8')
    pd.DataFrame(valid_log).to_csv(os.path.join(viz_dir, 'validing_log.txt'),
                                   header=False, index=False, encoding='utf8')
    # Save labels file to viz directory
    shutil.copyfile(kwargs['labels_file'], os.path.join(viz_dir, 'labels.txt'))

    # Save task settings.
    with open(os.path.join(viz_dir, 'task.json'), 'w') as task_file:
        kwargs['task_spec'].update({"acc": best_acc, 'best_epoch': best_epoch})
        print(json.dumps(kwargs['task_spec'], indent=True, ensure_ascii=False), file=task_file)
    return model, val_acc_history


def __config_list_or_folder_dataset(records, data_pattern):
    if not isinstance(records, (list, tuple)):
        records = [records]
    if records:
        if all(os.path.exists(r) for r in records) and all(os.path.isdir(r) for r in records):
            return {'records': None, 'ori_img_root': records}
        elif all(os.path.isfile(r) for r in records):
            return {'records': records, 'ori_img_root': data_pattern}
    else:
        return {'records': None, 'ori_img_root': data_pattern}
    raise ValueError(f"records({records}) or data_pattern({data_pattern}) config error! 大概率是文件不存在！")


def main(args):
    assert args.gpus is None or args.batch_size % len(args.gpus) == 0, 'Batch size must exactly divide number of gpus'
    cp = ColorPrinter()
    # if 'convolutionalvisiontransformer' in args.model_name.lower():
    #     image_size = (384, 384)
    # else:
    #     image_size = (512, 512)
    image_size = (512, 512)
    # Initialize data transformer for this run
    kwargs = {}
    data_transforms = {'train': create_standard_image_transformer(image_size, phase='train',
                                                                  normalize_method=args.normalize_method),
                       'valid': create_standard_image_transformer(image_size, phase='valid',
                                                                  normalize_method=args.normalize_method)}
    # Initialize datasets and dataloader for this run
    image_datasets = {'train': create_classification_dataset(
        recursive=True, transform=data_transforms['train'], labels_file=args.labels_file,
        check_sample_exists=get_param_in_cwd('check_sample_exists', True),
        batch_balance=args.batch_balance, **__config_list_or_folder_dataset(args.train, args.data_pattern))}
    # Save labels file if needed!
    # save_classification_dataset_labels(image_datasets['train'], os.path.join(args.model_root, 'labels.txt'))
    image_datasets['valid'] = create_classification_dataset(
        transform=data_transforms['valid'], classes=image_datasets['train'].classes,
        check_sample_exists=get_param_in_cwd('check_sample_exists', True),
        recursive=True, **__config_list_or_folder_dataset(args.valid, args.data_pattern))
    assert image_datasets['train'].classes == image_datasets['valid'].classes
    log_long_str('Train:%s\nValid:%s' % (image_datasets['train'], image_datasets['valid']))
    # Initialize the core for this run
    logger.info(f'Creating model {args.model_name}...')
    try:
        kwargs.update({'e2e_comp': args.e2e_comp})
    except:
        pass

    # Creat Models
    model = create_model(args.model_name, num_classes=image_datasets['train'].num_classes, **kwargs)

    # Send the core to GPU
    if args.gpus and len(args.gpus) > 1:
        model_ft = nn.DataParallel(model, device_ids=args.gpus)
    elif len(args.gpus) == 1:
        device = torch.device(f"cuda:{args.gpus[0]}" if torch.cuda.is_available() else "cpu")
        model_ft = model.to(device)
    else:
        model_ft = model.to('cpu')
    dataloaders_dict = {x: utils.data.DataLoader(image_datasets[x],
                                                 batch_size=args.batch_size,
                                                 drop_last=False,
                                                 shuffle=True,
                                                 num_workers=args.j)
                        for x in ['train', 'valid']}
    # Initialize optimizer and learning rate for this run
    optimizer = create_optimizer(args.optimizer, parameters=model_ft.parameters(), lr=args.init_lr)
    lr_scheduler = create_lr_scheduler('cosine', optimizer,
                                       T_max=args.epochs * len(image_datasets['train']) // args.batch_size)
    # Setup the loss function
    criterion = create_losses('softmax_ce')

    # Pretrain Weights
    if args.weights != "":
        assert os.path.exists(args.weights), "weights file: '{}' not exist.".format(args.weights)
        weights_dict = torch.load(args.weights, map_location=device)
        # if 'swintransformer' in args.model_name.lower():
            # weights_dict = weights_dict['model_state_dict']
            # for k in list(weights_dict.keys()):
            #     if "head" in k:
            #         del weights_dict[k]
        model_ft.load_state_dict(weights_dict['model_state_dict'], strict=False)
        optimizer.load_state_dict(weights_dict['optimizer_state_dict'])
        lr_scheduler.load_state_dict(weights_dict['lr_scheduler_state_dict'])

    # Train and evaluate
    model_dir = create_dir_if_not_exists(args.model_root, add_date=False)
    logger.info('Start training...')
    model_ft, hist = train_model(model_ft, device, dataloaders_dict, args.batch_size, criterion, optimizer,
                                 lr_scheduler, num_epochs=args.epochs,
                                 is_inception=("inception" in args.model_name),
                                 save_dir=os.path.join(model_dir, args.model_name),
                                 iters_verbose=args.iters_verbose,
                                 labels_file=args.labels_file,
                                 task_spec={'num_classes': image_datasets['train'].num_classes,
                                            'model_name': args.model_name,
                                            'transform': {'input_size': image_size,
                                                          'normalize_method': args.normalize_method},
                                            'device': str(device),
                                            'type': 'what'},
                                 save_per_epoch=args.save_per_epoch)

    # logger.info(f'Done for training. Best metric is {max(hist)}@{hist.index(max(hist)) + 1} epoch. '
    #             f'Average for last 5 iters is {sum(hist[-5:]) / len(hist[-5:])}')


# Modify this parameter if necessary!
DATA_ROOT = os.path.expanduser(r'.\Data\Label_Files')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--train', nargs='*', default=os.path.join(DATA_ROOT, 'train.txt'), help='Training dataset')
    parser.add_argument('--valid', nargs='*', default=os.path.join(DATA_ROOT, 'val.txt'), help='Validation dataset')
    parser.add_argument('--labels_file', default=os.path.join(DATA_ROOT, 'labels.txt'), help='Labels file')
    parser.add_argument('--data_pattern', default=os.path.join(DATA_ROOT, 'images'), nargs='*',
                        help='Where to save origin image data.')
    parser.add_argument('-j', '--worker', dest='j', default=4, type=int, help='Number of workers.(default=0)')
    parser.add_argument('--batch_balance', default=False, action='store_true', help='Batch balance samples for train.')
    parser.add_argument('--normalize_method', default='imagenet', choices=['-1+1', 'imagenet'],
                        help='Normalize method.')
    parser.add_argument('--model_name', default='resnet18', help='Model name')
    parser.add_argument('--gpus', type=int, nargs='*', default=[0], help='GPU index to be used!')
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=3, type=int, help='number of total epochs to run')
    parser.add_argument('--init_lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--optimizer', default='sgd', help='Optimizer')
    parser.add_argument('--model_root', default='models', help='path where to save')
    parser.add_argument('--iters_verbose', default=1, type=int, help='print frequency')
    parser.add_argument('--iters_start', default=0, type=int, help='Iters start')
    parser.add_argument('--save_per_epoch', action='store_true', help='save weights of per epoch or not')
    parser.add_argument('--weights', type=str, default=r'./Data/Patch_Results/train/BH_LymphDS.pth')

    main(parser.parse_args())
