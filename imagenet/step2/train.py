import argparse
import datetime
import sys
import time

import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils
import torch.utils
import torch.utils.data.distributed
import torchvision.models as models
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel
from torchvision import datasets

from imagenet.step2.reactnet import reactnet, HardBinaryConv
from imagenet.utils import KD_loss
from imagenet.utils.utils import *
from optimizers.cema_optimizer import CEMA_Optimizer

parser = argparse.ArgumentParser("birealnet18")
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--epochs', type=int, default=256, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--save', type=str, help='path for saving trained models')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--teacher', type=str, default='resnet34', help='path of ImageNet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

########################################################################################################################

parser.add_argument('--MASTER_PORT', default=12356, type=int)
parser.add_argument('--data', metavar='DIR', help='path to dataset')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--validate', action='store_true', default=False)

parser.add_argument('--step1', type=str, default=None)

parser.add_argument("--binary_optimizer", type=str, default="Adam")
parser.add_argument("--alpha", type=float)
parser.add_argument("--gamma", type=float)


args = parser.parse_args()

CLASSES = 1000


def main_process():
    # set address for master process to localhost since we use a single node
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = f'{args.MASTER_PORT}'

    # use all gpus pytorch can find
    args.world_size = torch.cuda.device_count()
    print('Found {} GPUs:'.format(args.world_size))
    for i in range(args.world_size):
        print('{} : {}'.format(i, torch.cuda.get_device_name(i)))

    args.batch_size_per_gpu = args.batch_size / args.world_size

    if not (args.batch_size / args.world_size).is_integer():
        logging.error(f"Can not split batch size {args.batch_size} evenly over {args.world_size} gpus.")
        return

    args.batch_size_per_gpu = args.batch_size // args.world_size

    print(f"\nCUDNN VERSION: {torch.backends.cudnn.version()}\n")
    cudnn.benchmark = True
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if not args.data:
        raise Exception("error: No data set provided")

    # start processes for all gpus
    context = mp.spawn(gpu_process, nprocs=args.world_size, args=(args,), join=False)

    try:
        while not context.join():
            time.sleep(5)
    except BaseException as e:
        if not isinstance(e, KeyboardInterrupt):
            print(f"Caught exception in main process: {e}")
            import traceback
            traceback.print_exc()
        print("Terminating child processes!")
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join()


def gpu_process(gpu, args):
    if gpu == 0:
        logging.getLogger().setLevel("INFO")
    else:
        logging.getLogger().setLevel("ERROR")

    log_format = f'{gpu}: %(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')

    # each gpu runs in a separate process
    try:
        torch.cuda.set_device(gpu)
        # torch.distributed.init_process_group(backend='nccl', init_method='env://',
        #                                      rank=gpu, world_size=args.world_size)
        torch.distributed.init_process_group(backend='gloo', init_method='env://',
                                             rank=gpu, world_size=args.world_size)

        start_t = time.time()

        cudnn.benchmark = True
        cudnn.enabled = True
        logging.info(f"args = {args}")

        # load model
        model_teacher = models.__dict__[args.teacher](pretrained=True).cuda()
        model_teacher = DistributedDataParallel(model_teacher, device_ids=[gpu], output_device=gpu)
        for p in model_teacher.parameters():
            p.requires_grad = False
        model_teacher.eval()

        model_student = reactnet()
        model_student = model_student.cuda(gpu)
        model_student = DistributedDataParallel(model_student, device_ids=[gpu], output_device=gpu)

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()
        criterion_kd = KD_loss.DistributionLoss()

        logging.info(f"Learning rate: {args.learning_rate}")

        if args.step1:
            logging.info(f"Loading from pretrained step1: {args.step1}")
            checkpoint = torch.load(args.step1, map_location=lambda storage, loc: storage.cuda(gpu))
            model_student.load_state_dict(checkpoint["state_dict"])

        binary_optimizer, real_optimizer = get_optimizers(args, model_student)


        start_epoch = 0
        best_top1_acc = 0

        # optionally resume from a checkpoint
        checkpoint_tar = args.resume
        if args.resume:
            if os.path.exists(checkpoint_tar):
                logging.info(f"=> loading checkpoint '{checkpoint_tar}'")
                checkpoint = torch.load(checkpoint_tar, map_location=lambda storage, loc: storage.cuda(gpu))
                start_epoch = checkpoint['epoch'] + 1
                best_top1_acc = checkpoint['best_top1_acc']
                model_student.load_state_dict(checkpoint['state_dict'], strict=False)
                logging.info(f"loaded checkpoint {checkpoint_tar}"
                             f" epoch = {checkpoint['epoch']} top-1 acc:{best_top1_acc}")
                if not args.validate:
                    real_optimizer.load_state_dict(checkpoint["real_optimizer"])
                    binary_optimizer.load_state_dict(checkpoint["binary_optimizer"])
            else:
                logging.error("Tried to resume but no checkpoint found!")
                return

        schedulers = {}
        if not args.validate:
            schedulers = {
                "binary_lr": torch.optim.lr_scheduler.LambdaLR(binary_optimizer,
                                                               lambda step: (1.0 - step / args.epochs),
                                                               last_epoch=start_epoch - 1),
                "real_lr": torch.optim.lr_scheduler.LambdaLR(real_optimizer, lambda step: (1.0 - step / args.epochs),
                                                             last_epoch=start_epoch - 1)
            }

        logging.info("Loading data")
        if "tfrecords" in args.data:
            args.dali_enabled = True
            from imagenet.imagenet_tfrecord import ImageNet_TFRecord_loader

            logging.info("Loading tfrecords")
            train_loader = ImageNet_TFRecord_loader(args.data, 'train', args.batch_size_per_gpu, args.workers,
                                                    gpu, args.world_size, augment=True)
            val_loader = ImageNet_TFRecord_loader(args.data, 'val', args.batch_size_per_gpu, args.workers,
                                                  gpu, args.world_size, augment=False)
        else:
            args.dali_enabled = False
            logging.info("Loading raw images")
            # load training data
            traindir = os.path.join(args.data, 'train')
            valdir = os.path.join(args.data, 'val')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

            if not args.validate:
                # data augmentation
                crop_scale = 0.08
                lighting_param = 0.1
                train_transforms = transforms.Compose([
                    transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
                    Lighting(lighting_param),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize])

                train_dataset = datasets.ImageFolder(
                    traindir,
                    transform=train_transforms)

                train_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=args.batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True)

            # load validation data
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True)

        logging.info("Done loading data")
        if args.validate:
            valid_obj, valid_top1_acc, valid_top5_acc = validate(val_loader, model_student, criterion, gpu, args)
            logging.info(f'Loss: {valid_obj} acc@1 {valid_top1_acc:.3f} acc@5 {valid_top5_acc:.3f}')
            return

        for epoch in range(start_epoch, args.epochs):
            train_loss, train_top1, train_top5 = train(epoch, train_loader, model_student, model_teacher,
                                                       criterion_kd, real_optimizer, binary_optimizer, gpu, args)
            valid_obj, valid_top1_acc, valid_top5_acc = validate(val_loader, model_student, criterion, gpu, args)

            is_best = False
            if valid_top1_acc > best_top1_acc:
                best_top1_acc = valid_top1_acc
                is_best = True

            if gpu == 0:
                save_file_base_path = args.save

                save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model_student.state_dict(),
                    'best_top1_acc': best_top1_acc,
                    'real_optimizer': real_optimizer.state_dict(),
                    'binary_optimizer': binary_optimizer.state_dict(),
                }, is_best, save_file_base_path)

            scheduled_hyper_params = {}
            if "alpha" in binary_optimizer.param_groups[0]:
                scheduled_hyper_params['alpha'] = binary_optimizer.param_groups[0]['alpha']
            else:
                scheduled_hyper_params['binary_lr'] = real_optimizer.param_groups[0]['lr']
            scheduled_hyper_params['real_lr'] = real_optimizer.param_groups[0]['lr']

            for _, scheduler in schedulers.items():
                scheduler.step()
            epoch += 1

            logging.info(f"Average time per epoch: "
                         f"{datetime.timedelta(seconds=int((time.time() - start_t) / (epoch - start_epoch + 1)))}")

        training_time = (time.time() - start_t) / 3600
        logging.info('total training time = {} hours'.format(training_time))

    except Exception as e:
        logging.error(f"Exception caught in process: {gpu} - {e}")
        raise e
    finally:
        dist.destroy_process_group()


def train(epoch, train_loader, model_student, model_teacher, criterion, real_optimizer, binary_optimizer, gpu, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    metric_time = AverageMeter('Metrics', ':6.3f')

    student_time = AverageMeter('stud', ':6.3f')
    teacher_time = AverageMeter('teach', ':6.3f')
    back_time = AverageMeter('back', ':6.3f')
    opt_time = AverageMeter('opt', ':6.3f')

    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    ff_ratio = AverageMeter("ff_ratio")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, student_time, teacher_time, back_time, opt_time, metric_time, losses, top1, top5,
         ff_ratio],
        prefix="Epoch: [{}]".format(epoch))

    model_student.train()
    model_teacher.eval()
    end = time.time()

    for param_group in binary_optimizer.param_groups:
        if "alpha" in param_group:
            cur_alpha = param_group['alpha'] * param_group['lr']
            logging.info(f'binary alpha: {cur_alpha}')
        else:
            cur_lr = param_group['lr']
            logging.info(f'binary learning_rate: {cur_lr}')
    for param_group in real_optimizer.param_groups:
        cur_lr = param_group['lr']
        logging.info(f'real learning_rate: {cur_lr}')

    prev_binary_weights = torch.cat([sign(module.weight.detach().view(-1)) for module in model_student.modules()
                                     if isinstance(module, HardBinaryConv)])

    for i, data in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.dali_enabled:
            images = data[0]["data"]
            target = data[0]["label"].squeeze().cuda(gpu).long()
        else:
            images, target = data
            images = images.cuda(gpu, non_blocking=True)
            target = target.cuda(gpu, non_blocking=True)

        # compute output
        student_start_time = time.time()

        logits_student = model_student(images)

        student_end_time = time.time()
        student_time.update(student_end_time - student_start_time)

        logits_teacher = model_teacher(images)

        teacher_time.update(time.time() - student_end_time)

        loss = criterion(logits_student, logits_teacher)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))
        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        # compute gradient and do SGD step
        back_start_time = time.time()

        real_optimizer.zero_grad()
        binary_optimizer.zero_grad()
        loss.backward()

        back_end_time = time.time()

        back_time.update(back_end_time - back_start_time)

        real_optimizer.step()
        binary_optimizer.step()

        opt_time.update(time.time() - back_end_time)

        ################################################################################################################
        metric_start_time = time.time()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(logits_student, target, topk=(1, 5))

        n = images.size(0)
        losses.update(loss.item(), n)  # accumulated loss
        top1.update(prec1.item(), n)
        top5.update(prec5.item(), n)

        new_binary_weights = torch.cat(
            [sign(module.weight.detach().view(-1)) for module in model_student.modules()
             if isinstance(module, HardBinaryConv)])

        ff_ratio.update(
            torch.count_nonzero((prev_binary_weights.not_equal(new_binary_weights))) / prev_binary_weights.numel())

        prev_binary_weights = new_binary_weights

        metric_end_time = time.time()
        metric_time.update(metric_end_time - metric_start_time)
        ################################################################################################################

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    return losses.avg, top1.avg, top5.avg


def validate(val_loader, model, criterion, gpu, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluation mode
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, data in enumerate(val_loader):
            if args.dali_enabled:
                images = data[0]["data"]
                target = data[0]["label"].squeeze().cuda(gpu).long()
            else:
                images = data[0].cuda(gpu)
                target = data[1].cuda(gpu)

            # compute output
            logits = model(images)
            loss = criterion(logits, target)

            # measure accuracy and record loss
            pred1, pred5 = accuracy(logits, target, topk=(1, 5))
            n = images.size(0)

            # Because we do distritubed training we need to average the statistics over all the processes
            loss = reduce_tensor(loss.data, args.world_size)
            pred1 = reduce_tensor(pred1, args.world_size)
            pred5 = reduce_tensor(pred5, args.world_size)

            losses.update(loss.item(), n)
            top1.update(pred1[0], n)
            top5.update(pred5[0], n)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        logging.info(' * acc@1 {top1.avg:.3f} acc@5 {top5.avg:.3f}'
                     .format(top1=top1, top5=top5))

    return losses.avg, top1.avg, top5.avg


def get_optimizers(args, model):
    logging.info("Intializing optimizer")

    binary_params = []
    bn_params = []
    real_params = []
    for module in model.modules():
        if isinstance(module, HardBinaryConv):
            for param in module.parameters(recurse=False):
                binary_params.append(param)

        elif isinstance(module, nn.BatchNorm2d):
            for param in module.parameters(recurse=False):
                bn_params.append(param)
        else:
            for param in module.parameters(recurse=False):
                real_params.append(param)

    real_optimizer = torch.optim.Adam(
        [{'params': bn_params, "name": "BN"},
         {'params': real_params, "name": "Real", 'weight_decay': args.weight_decay}],
        lr=args.learning_rate)

    if args.binary_optimizer == "Adam":
        binary_optimizer = torch.optim.Adam(
            [{'params': binary_params, 'weight_decay': args.weight_decay, "name": "Binary"}],
            lr=args.learning_rate)
    elif args.binary_optimizer == "CEMA":
        binary_optimizer = CEMA_Optimizer(binary_params=binary_params,
                                          alpha=args.alpha,
                                          gamma=args.gamma)
        logging.info(f"Binary optimizer: CEMA(alpha={args.alpha}, gamma={args.gamma})")
        HardBinaryConv.LATENT_WEIGHTS = False
    else:
        raise Exception(f"Unkonwn binary optimizer requested: {args.binary_optimizer}")

    return binary_optimizer, real_optimizer


def sign(input: torch.Tensor):
    output = input.sign()
    output[output == 0] = 1
    return output


def reduce_tensor(tensor, world_size):
    rt = tensor
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt


if __name__ == '__main__':
    main_process()
