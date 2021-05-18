import os
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import networks
import trainer
import logging
import numpy as np
from utils import get_config,set_env,set_logger,set_outdir
from utils import get_dataloader
from utils import get_train_setting,load_checkpoint,get_proc,save_checkpoint
import os
from datasets.ct import *
from ct_test import *

def initialize():
    train_params = {'batch_size': 64,
                    'shuffle': True,
                    'num_workers': 3}

    test_params = {'batch_size': 32,
                   'shuffle': False,
                   'num_workers': 2}

    dataset = r'../data'

    train_loader = COVID_CT_Dataset(mode='train', n_classes=3, dataset_path=dataset,
                                    dim=(224, 224))
    test_loader = COVID_CT_Dataset( mode='test', n_classes=3, dataset_path=dataset,
                                   dim=(224, 224))
    training_generator = DataLoader(train_loader, **train_params)
    test_generator = DataLoader(test_loader, **test_params)
    return training_generator, test_generator


def main(conf):

    warnings.filterwarnings("ignore")
    best_score = 0.
    val_score = 0
    val_loss = 0
    epoch_start = 0


    # dataloader
    #train_loader,val_loader = get_dataloader(conf)
    train_loader , val_loader = initialize()
    # model
    model = networks.get_model(conf)
    #print(model)
    #	exit(-1)
    model = nn.DataParallel(model).cuda()

    if conf.weightfile is not None:
        wmodel = networks.get_model(conf)
        wmodel = nn.DataParallel(wmodel).cuda()
        checkpoint_dict = load_checkpoint(wmodel, conf.weightfile)
        if 'best_score' in checkpoint_dict:
            print('best score: {}'.format(best_score))
    else:
        wmodel = model

    # training setting
    criterion,optimizer,scheduler = get_train_setting(model,conf)

    # training and evaluate process for each epoch
    train,validate = get_proc(conf)

    if conf.resume:
        checkpoint_dict = load_checkpoint(model, conf.resume)
        epoch_start = checkpoint_dict['epoch']
        if 'best_score' in checkpoint_dict:
            best_score = checkpoint_dict['best_score']
            print('best score: {}'.format(best_score))
        print('Resuming training process from epoch {}...'.format(epoch_start))
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
        print('Resuming lr scheduler')
        print(checkpoint_dict['scheduler'])

    if conf.evaluate:
        print( validate(val_loader, model,criterion, conf))
        return

    detach_epoch = conf.epochs + 1
    if 'detach_epoch' in conf:
        detach_epoch = conf.detach_epoch

    start_eval = 0
    if 'start_eval' in conf:
        start_eval = conf.start_eval


    ## ------main loop-----
    for epoch in range(epoch_start, conf.epochs):
        lr = optimizer.param_groups[0]['lr']
        logging.info("Epoch: [{} | {} LR: {}".format(epoch+1,conf.epochs,lr))

        if epoch == detach_epoch:
            model.module.set_detach(False)
#        test(model , train_loader)
        tmp_loss  = train(train_loader, model, criterion, optimizer, conf,wmodel)
        infostr = {'Epoch:  {}   train_loss: {}'.format(epoch+1,tmp_loss)}
        logging.info(infostr)
        scheduler.step()

        if True:
            with torch.no_grad():
                val_score,val_loss,mscore,ascore = validate(val_loader, model,criterion, conf)
                comscore = val_score
                if 'midlevel' in conf:
                    if conf.midlevel:
                        comscore = ascore
                is_best = comscore > best_score
                best_score = max(comscore,best_score)
                infostr = {'Epoch:  {:.4f}   loss: {:.4f},gs: {:.4f},bs:{:.4f} ,ms:{:.4f},as:{:.4f}'.format(epoch+1,val_loss,val_score,best_score,mscore,ascore)}
                logging.info(infostr)
                save_checkpoint(
                        {'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                        'best_score': best_score
                        }, is_best, outdir=conf['outdir'])
        test(model , train_loader)
    print('Best val acc: {}'.format(best_score))
    return 0


if __name__ == '__main__':

    # get configs and set envs
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)



