import datetime
import os
from functools import partial

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.ninet import MyNet
from utils.loss import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.smoke_dataloader import CustomDataset, deeplab_dataset_collate
from utils.common_util import seed_everything, show_config, worker_init_fn
from utils.fit_function import fit_one_epoch

if __name__ == "__main__":
    # ---------------------------------#
    #   Cuda    是否使用Cuda 没有GPU可以设置成False
    # ---------------------------------#
    Cuda = True
    # ----------------------------------------------#
    #  随机种子
    # ----------------------------------------------#
    seed = 42
    seed_everything(seed)
    # ---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    distributed = False
    # ---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #  是否使用混合精度训练, 可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = False
    # -----------------------------------------------------#
    #   类别数
    # -----------------------------------------------------#
    num_classes = 2
    # ---------------------------------#
    #   主干网络：
    # ---------------------------------#
    backbone = "resnet50conv3x3stem"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = False
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   模型的权值文件路径
    # ----------------------------------------------------------------------------------------------------------------------------#
    model_path = 'model_data/smoke/ninet/81.55.pth'
    # ---------------------------------------------------------#
    #   下采样的倍数
    # ---------------------------------------------------------#
    downsample_factor = 8
    # ------------------------------#
    #   输入图片的大小
    # ------------------------------#
    input_shape = [256, 256]
    # ------------------------------------------------------------------#
    #  是否冻结训练,默认先冻结主干训练后解冻训练。
    # ------------------------------------------------------------------#
    Freeze_Train = True
    # ------------------------------------------------------------------#
    #   冻结阶段训练参数
    #   Init_Epoch          模型当前开始的训练世代，其值可以大于Freeze_Epoch，如设置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、Total_Epoch = 100
    #                       会跳过冻结阶段，直接从60代开始，并调整对应的学习率。
    #                       （断点续练时使用）
    #   Freeze_Epoch        模型冻结训练的Freeze_Epoch(当Freeze_Train=False时失效)
    #   Freeze_batch_size   模型冻结训练的batch_size(当Freeze_Train=False时失效)
    # ------------------------------------------------------------------#
    Init_Epoch = 0
    Freeze_Epoch = 3
    Freeze_batch_size = 32
    # ------------------------------------------------------------------#
    #   解冻阶段训练参数
    #   Total_Epoch          模型总共训练的epoch
    #   Unfreeze_batch_size     模型在解冻后的batch_size
    # ------------------------------------------------------------------#
    Total_Epoch = 200
    Unfreeze_batch_size = 32
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率:
    #                   当使用Adam优化器时建议设置  Init_lr=5e-4
    #                   当使用SGD优化器时建议设置   Init_lr=7e-3
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 1e-4
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有'step'、'cos'
    # ------------------------------------------------------------------#
    lr_decay_type = 'step'
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值
    # ------------------------------------------------------------------#
    save_period = 1
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    save_dir = 'smoke_logs'
    # ------------------------------------------------------------------#
    #   eval_flag       是否在训练时进行评估，评估对象为验证集
    #   eval_period     代表多少个epoch评估一次，不建议频繁的评估
    # ------------------------------------------------------------------#
    eval_flag = True
    eval_period = 1
    # ------------------------------------------------------------------#
    #   数据集路径
    # ------------------------------------------------------------------#
    dataset_path = 'datasets/virtual_smoke_v2'
    # ------------------------------------------------------------------#
    #   是否使用dice_loss，建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    # ------------------------------------------------------------------#
    dice_loss = True
    # ------------------------------------------------------------------#
    #   是否使用focal loss来防止正负样本不平衡
    # ------------------------------------------------------------------#
    focal_loss = False
    # ------------------------------------------------------------------#
    #   是否给不同种类赋予不同的损失权值，默认是平衡的。设置的话，注意设置成numpy形式的，长度和num_classes一样。
    #   如：num_classes = 3   cls_weights = np.array([1, 2, 3], np.float32)
    # ------------------------------------------------------------------#
    cls_weights = np.ones([num_classes], np.float32)
    # ------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据，1代表关闭多线程，开启后会加快数据读取速度，但是会占用更多内存
    # ------------------------------------------------------------------#
    num_workers = 1
    # ---------------------#
    #   是否使用辅助分支,会占用大量显存（效果不明显）
    # ---------------------#
    aux_branch = False
    # ------------------------------------------------------------------#
    #   init_type  使用到的初始化设置，可选的有normal、kaiming、xavier、orthogonal
    # ------------------------------------------------------------------#
    init_type = "kaiming"
    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0


    model = MyNet(num_classes=num_classes, backbone=backbone, down_rate=downsample_factor, pretrained=pretrained, aux_branch=aux_branch)

    # 没有使用预训练权重那么需要初始化
    if not pretrained:
        weights_init(model, init_type=init_type)

    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   加载完整模型权重，非预训练骨干权重
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print('Load weights Finished!')

    # ----------------------#
    #   记录Loss
    # ----------------------#
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    if local_rank == 0:
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ------------------------------------------------------------------#
    #   混合精度处理
    # ------------------------------------------------------------------#
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    # ----------------------------#
    #   多卡同步Bn
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   多卡平行运行
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()

    # ---------------------------#
    #   读取数据集对应的txt，使用自己数据集时需要修改路径
    # ---------------------------#
    with open(os.path.join(dataset_path, "images_index/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(dataset_path, "images_index/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    if local_rank == 0:
        show_config(
            num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
            Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, Total_Epoch=Total_Epoch,
            Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train,
            Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
            lr_decay_type=lr_decay_type,
            save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
        )
        # ---------------------------------------------------------#
        #   总训练世代指的是遍历全部数据的总次数
        #   总训练步长指的是梯度下降的总次数
        #   每个训练世代包含若干训练步长，每个训练步长进行一次梯度下降。
        #   此处仅建议最低训练世代，上不封顶，计算时只考虑了解冻部分
        # ----------------------------------------------------------#
        wanted_step = 1.5e4 if optimizer_type == "sgd" else 0.5e4
        total_step = num_train // Unfreeze_batch_size * Total_Epoch
        if total_step <= wanted_step:
            if num_train // Unfreeze_batch_size == 0:
                raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
            wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
            print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m" % (
                optimizer_type, wanted_step))
            print(
                "\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m" % (
                    num_train, Unfreeze_batch_size, Total_Epoch, total_step))
            print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m" % (
                total_step, wanted_step, wanted_epoch))


    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.backbone.parameters():
                param.requires_grad = False
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size


        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr, momentum=momentum, nesterov=True,weight_decay=weight_decay)
        }[optimizer_type]
        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Total_Epoch)
        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataset = CustomDataset(train_lines, input_shape, num_classes, True, dataset_path)
        val_dataset = CustomDataset(val_lines, input_shape, num_classes, False, dataset_path)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,drop_last=True, collate_fn=deeplab_dataset_collate, sampler=train_sampler,
                         worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,drop_last=True, collate_fn=deeplab_dataset_collate, sampler=val_sampler,
                             worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

        # ----------------------#
        #   记录eval的map曲线
        # ----------------------#
        if local_rank == 0:
            eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, dataset_path, log_dir, Cuda,
                                         eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback = None
        # ---------------------------------------#
        #   开始模型训练
        # ---------------------------------------#
        for epoch in range(Init_Epoch, Total_Epoch):
            # ---------------------------------------#
            #   如果模型有冻结学习部分
            #   则解冻，并设置参数
            # ---------------------------------------#
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                # ---------------------------------------#
                #   获得学习率下降的公式
                # ---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, Total_Epoch)

                for param in model.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=deeplab_dataset_collate, sampler=train_sampler,
                                 worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=deeplab_dataset_collate, sampler=val_sampler,
                                     worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                          epoch_step, epoch_step_val, gen, gen_val, Total_Epoch, Cuda, dice_loss, focal_loss,
                          cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank, aux_branch)

            if distributed:
                dist.barrier()

        if local_rank == 0:
            loss_history.writer.close()
