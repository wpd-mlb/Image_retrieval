'''
@Author      : WangPeide
@Contact     : 377079627@qq.com
LastEditors: Please set LastEditors
@Description :
LastEditTime: 2022/1/30 12:40
'''
import logging
from ignite.engine import Events,create_supervised_trainer,create_supervised_evaluator
from ignite.handlers import Timer,TerminateOnNan,ModelCheckpoint
from ignite.metrics import Loss,RunningAverage,Accuracy
import torch
import os
from tqdm import tqdm
import numpy as np
from ignite.engine import Engine
from ignite.utils import convert_tensor
from torch.utils.tensorboard import SummaryWriter
from model.net import freeze_layers,fix_bn
from torch.cuda.amp import autocast as autocast
def do_train(cfg,model,train_loader,val_loader,optimizer,scheduler,metrics,device):

    def _prepare_batch(batch, device=None, non_blocking=False):
        """Prepare batch for training: pass to a device with options.

        """
        x, y = batch
        return (convert_tensor(x, device=device, non_blocking=non_blocking),
                convert_tensor(y, device=device, non_blocking=non_blocking))


    def create_supervised_dp_trainer(model, optimizer,
                                device=None, non_blocking=False,
                                prepare_batch=_prepare_batch,
                                output_transform=lambda x, y, y_pred, loss: loss.item()):
        """
        Factory function for creating a trainer for supervised models.

        Args:
            model (`torch.nn.Module`): the model to train.
            optimizer (`torch.optim.Optimizer`): the optimizer to use.
            loss_fn (torch.nn loss function): the loss function to use.
            device (str, optional): device type specification (default: None).
                Applies to both model and batches.
            non_blocking (bool, optional): if True and this copy is between CPU and GPU, the copy may occur asynchronously
                with respect to the host. For other cases, this argument has no effect.
            prepare_batch (callable, optional): function that receives `batch`, `device`, `non_blocking` and outputs
                tuple of tensors `(batch_x, batch_y)`.
            output_transform (callable, optional): function that receives 'x', 'y', 'y_pred', 'loss' and returns value
                to be assigned to engine's state.output after each iteration. Default is returning `loss.item()`.

        Note: `engine.state.output` for this engine is defind by `output_transform` parameter and is the loss
            of the processed batch by default.

        Returns:
            Engine: a trainer engine with supervised update function.
        """
        # pytorch 使用amp.autocast半精度加速训练
        # https://blog.csdn.net/weixin_42216799/article/details/110876374?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522164353992716780269849959%2522%252C%2522scm%2522%253A%252220140713.130102334..%2522%257D&request_id=164353992716780269849959&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduend~default-1-110876374.first_rank_v2_pc_rank_v29&utm_term=autocast&spm=1018.2226.3001.4187
        if device:
            model.to(device)

        def _update(engine, batch):
            # model.train()
            optimizer.zero_grad()
            x, y = prepare_batch(batch, device=device, non_blocking=non_blocking)
            # 前向过程(model + loss)开启 autocast

            with autocast():
                triplet_value,ce_value,total_loss = model(x.half(),y.half())
            total_loss = total_loss.mean() # model 里求均值
            # Scales loss. 为了梯度放大.
            # 反向过程 loss backward 和optimizer的step
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            writer.add_scalar("Triplet Loss", triplet_value.cpu().data.numpy(),engine.state.iteration)
            writer.add_scalar("CE Loss", ce_value.cpu().data.numpy(),engine.state.iteration)
            writer.add_scalar("total loss",total_loss.cpu().data.numpy(),engine.state.iteration)
            scaler.update()
            # total_loss.backward()
            # optimizer.step()
            return output_transform(x, y, None, total_loss)

        return Engine(_update)

    scaler = torch.cuda.amp.GradScaler() # pytorch自动精度
    master_device = device[0]  # 默认设置第一块为主卡
    trainer = create_supervised_dp_trainer(model, optimizer, device=master_device)
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan()) # 当一个 iteration 结束时, 会触发此事件
    # TerminateOnNan：防止 NaN or Infinite value
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss') # compute average

    log_dir = cfg['log_dir']
    writer = SummaryWriter(log_dir=log_dir)

    # create pbar
    len_train_loader = len(train_loader)
    pbar = tqdm(total=len_train_loader)


    froze_num_layers = cfg['warm_up']['froze_num_lyers']
    if cfg['multi_gpu']:
        freeze_layers(model.module,froze_num_layers)
    else:
        freeze_layers(model,froze_num_layers)

    # Finetuning 模式下,patch较大,batch较小冻结全模型bn
    # Normal 模式下, 冻结对应网络层数
    if 'mode' in cfg and cfg['mode'] == "Finetuning":
        if cfg['multi_gpu']:
            fix_bn(model.module)
        else:
            fix_bn(model)


    ##########################################################################################
    ###########                    Events.ITERATION_COMPLETED                    #############
    ##########################################################################################

    # 每 log_period 轮迭代结束输出train_loss
    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        i = engine.state.iteration
        e = engine.state.epoch
        print("train:", e, i, )
        log_period = cfg['log_period']
        log_per_iter = int(log_period * len_train_loader) if int(log_period * len_train_loader) >= 1 else 1  # 计算打印周期
        current_iter = (engine.state.iteration - 1) % len_train_loader + 1 + (
                    engine.state.epoch - 1) * len_train_loader  # 计算当前 iter
        lr = optimizer.state_dict()['param_groups'][0]['lr']

        if current_iter % log_per_iter == 0:
            pbar.write("Epoch[{}] Iteration[{}] lr {:.7f} Loss {:.7f}".format(engine.state.epoch, current_iter, lr,
                                                                              engine.state.metrics['avg_loss']))
            pbar.update(log_per_iter)
            print("avg_loss",engine.state.metrics['avg_loss'])
            writer.add_scalar('loss', engine.state.metrics['avg_loss'], current_iter)

    # lr_scheduler Warm Up
    @trainer.on(Events.ITERATION_COMPLETED)
    def lr_scheduler_iteration(engine):
        scheduler.ITERATION_COMPLETED()  # 优化器: lr 更新模型参数
        current_iter = (engine.state.iteration - 1) % len_train_loader + 1 + (
                    engine.state.epoch - 1) * len_train_loader  # 计算当前 iter
        length = cfg['warm_up']['length']
        min_lr = cfg['warm_up']['min_lr']
        max_lr = cfg['warm_up']['max_lr']
        froze_num_layers = cfg['warm_up']['froze_num_lyers']
        if current_iter < length:
            """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
            lr = (max_lr - min_lr) / length * current_iter
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # pbar.write("lr: {}".format(lr))
            writer.add_scalar('lr',lr, current_iter)
            print("train_lr:",lr)
        if current_iter == length:
            if 'mode' in cfg and cfg['mode'] == "Finetuning":
                pass
            else:  # Normal 模式下,Warm Up结束解冻
                pass
                if cfg['multi_gpu']:
                    freeze_layers(model.module,froze_num_layers)
                else:
                    freeze_layers(model,froze_num_layers)
                #
                for param_group in optimizer.param_groups:
                    param_group['lr'] = cfg['optimizer']['lr']
                writer.add_scalar('lr', param_group['lr'], current_iter)
                print("train_lr_reback:", param_group['lr'])
    ##########################################################################################
    ##################               Events.EPOCH_COMPLETED                    ###############
    ##########################################################################################

    # 优化器: lr 更新模型参数
    @trainer.on(Events.EPOCH_COMPLETED)
    def lr_scheduler_epoch(engine):
        scheduler.EPOCH_COMPLETED()

    @trainer.on(Events.EPOCH_COMPLETED)
    def save_temp_epoch(engine):
        save_dir = cfg['save_dir']
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        epoch = engine.state.epoch
        if epoch % 1 == 0:
            model_name = os.path.join(save_dir, cfg['tag'] + "_temp.pth")
            # import pdb; pdb.set_trace()

            if cfg['multi_gpu']:
                save_pth = {'model': model.module.state_dict(), 'cfg': cfg}
                torch.save(save_pth, model_name)
            else:
                save_pth = {'model': model.state_dict(), 'cfg': cfg}
                torch.save(save_pth, model_name)

        if epoch % 10 == 0:
            model_name = os.path.join(save_dir, cfg['tag'] + "_" + str(epoch) + ".pth")
            if cfg['multi_gpu']:
                save_pth = {'model': model.module.state_dict(), 'cfg': cfg}
                torch.save(save_pth, model_name)
            else:
                save_pth = {'model': model.state_dict(), 'cfg': cfg}
                torch.save(save_pth, model_name)

    @trainer.on(Events.EPOCH_COMPLETED)
    def calu_acc(engine):
        epoch = engine.state.epoch
        if epoch % 10 == 0:
            model.eval()
            num_correct = 0
            num_example = 0
            torch.cuda.empty_cache()
            with torch.no_grad():
                for image, target in tqdm(train_loader):
                    image, target = image.to(master_device), target.to(master_device)
                    pred_logit_dict = model(image.float(), target.float())
                    pred_logit = [value for value in pred_logit_dict.values() if value is not None]

                    pred_logit = pred_logit[0]
                    indices = torch.max(pred_logit, dim=1)[1]
                    correct = torch.eq(indices, target).view(-1)
                    num_correct += torch.sum(correct).item()
                    num_example += correct.shape[0]

            acc = (num_correct / num_example)
            pbar.write("Acc: {}".format(acc))
            writer.add_scalar("Acc", acc, epoch)
            torch.cuda.empty_cache()
            model.train()


    @trainer.on(Events.EPOCH_COMPLETED)
    def reset_pbar(engine):
        pbar.reset()

    # @trainer.on(Events.EPOCH_COMPLETED)
    # def reset_dataset(engine): # 仅针对jr写的train_dataset,手动shuffle
    #     if hasattr(train_loader.dataset,'shuffle'):
    #         pbar.write("shuffle train_dataloader")
    #         train_loader.dataset.shuffle()

    max_epochs = cfg['max_epochs']
    trainer.run(train_loader, max_epochs=max_epochs)
    pbar.close()

if __name__ == "__main__":
    pass
# tensorboard --logdir=./features/log/log_20220211_tag_example