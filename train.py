import os
import numpy as np, argparse, time, pickle, random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from dataloader import IEMOCAPDataset, MELDDataset
from model import MaskedNLLLoss, LSTMModel, GRUModel, Model, MaskedMSELoss, FocalLoss, MMDFN_FocalLoss, PCGNet_FocalLoss
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
import pandas as pd
import pickle as pk
import datetime
import wandb
import yaml
import random
from types import SimpleNamespace

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def _init_fn(worker_id):
    np.random.seed(worker_id)

def get_train_valid_sampler(trainset, valid=0.1, dataset='IEMOCAP'):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])


def get_MELD_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = MELDDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid, 'MELD')

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = MELDDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


def get_IEMOCAP_loaders(batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = IEMOCAPDataset()
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory, worker_init_fn=_init_fn)

    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPDataset(train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory, worker_init_fn=_init_fn)

    return train_loader, valid_loader, test_loader

def train_or_eval_graph_model(args, model, loss_function, dataloader, epoch, cuda, modals, optimizer=None, train=False, dataset='IEMOCAP'):
    losses, preds, labels = [], [], []
    scores, vids = [], []
    ei, et, en, el = torch.empty(0).type(torch.LongTensor), torch.empty(0).type(torch.LongTensor), torch.empty(0), []
    if cuda:
        ei, et, en = ei.cuda(), et.cuda(), en.cuda()
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    seed_everything(args.seed)
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        textf1,textf2,textf3,textf4, visuf, acouf, qmask, umask, label = [d.cuda() for d in data[:-1]] if cuda else data[:-1]
        lengths = [(umask[j] == 1).nonzero(as_tuple=False).tolist()[-1][0] + 1 for j in range(len(umask))]
        label = torch.cat([label[j][:lengths[j]] for j in range(len(label))])
        log_prob, e_i, e_n, e_t, e_l, penalty_weight_loss, link_loss = model([textf1,textf2,textf3,textf4], qmask, umask, lengths, acouf, visuf, epoch, label=label, train=train)        
        cross_entropy_loss = loss_function(log_prob, label)
        penalty_weight_loss = args.penalty_weight_coff * penalty_weight_loss
        if epoch >= args.link_loss_epoch:
            loss = cross_entropy_loss + penalty_weight_loss
        else:
            link_loss = args.link_loss_coff * max((args.link_loss_epoch-epoch)/args.link_loss_epoch, 0) * link_loss
            loss = cross_entropy_loss + penalty_weight_loss - link_loss
        preds.append(torch.argmax(log_prob, 1).cpu().numpy())
        labels.append(label.cpu().numpy())
        losses.append(loss.item())
        if train:
            loss.backward()
            optimizer.step()
    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], float('nan'), [], [], [], [], []

    vids += data[-1]
    ei = ei.data.cpu().numpy()
    et = et.data.cpu().numpy()
    en = en.data.cpu().numpy()
    el = np.array(el)
    labels = np.array(labels)
    preds = np.array(preds)
    vids = np.array(vids)

    avg_loss = round(np.sum(losses)/len(losses), 4)
    avg_accuracy = round(accuracy_score(labels, preds)*100, 2)
    avg_fscore = round(f1_score(labels,preds, average='weighted')*100, 2)

    return avg_loss, avg_accuracy, labels, preds, avg_fscore, vids, ei, et, en, el


def train_args(args, run_id=None):
    print(args)
    cuda = torch.cuda.is_available()
    feat2dim = {'IS10':1582,'3DCNN':512,'textCNN':100,'bert':768,'denseface':342,'MELD_text':600,'MELD_audio':300}
    D_audio = feat2dim['IS10'] if args.Dataset=='IEMOCAP' else feat2dim['MELD_audio']
    D_visual = feat2dim['denseface']
    seed_everything(args.seed)

    model = Model(args.base_model, 
                    D_m=1024, 
                    D_g=args.hidden_dim, 
                    D_m_v = D_visual, 
                    D_m_a = D_audio, 
                    n_speakers=9 if args.Dataset=='MELD' else 2, 
                    max_seq_len=200, 
                    window_past=args.windowp, 
                    window_future=args.windowf,
                    n_classes=7 if args.Dataset=='MELD' else 6, 
                    dropout=args.dropout,
                    use_residue=args.use_residue,
                    modals=args.modals, 
                    av_using_lstm=args.av_using_lstm, 
                    dataset=args.Dataset,
                    backbone=args.backbone,
                    use_speaker=args.use_speaker, norm = args.norm,
                    # denoise
                    num_L = args.num_L, use_residue_denoise=args.use_residue_denoise,  denoise_dropout=args.denoise_dropout, 
                    gamma = args.gamma, zeta=args.zeta, temperature=args.temperature,
                    # nodeformer
                    use_gumbel=args.use_gumbel, num_K = args.num_K, gumbel_k=args.gumbel_k, nodeformer_heads=args.nodeformer_heads, nb_features_dim=args.nb_features_dim, 
                    tau=args.tau, nodeformer_dropout=args.nodeformer_dropout, use_residue_nodeformer=args.use_residue_nodeformer, use_jk_nodeformer=args.use_jk_nodeformer
                    )
    print ('Graph NN with', args.base_model, 'as base model.')

    if cuda:
        model.cuda()
    if args.Dataset == 'IEMOCAP':
        loss_weights = torch.FloatTensor([1/0.086747,
                                        1/0.144406,
                                        1/0.227883,
                                        1/0.160585,
                                        1/0.127711,
                                        1/0.252668])
        loss_function  = nn.NLLLoss(loss_weights.cuda() if cuda else loss_weights)
    else:
        loss_weights = torch.FloatTensor([1.0 / 0.466750766,
                                           1.0 / 0.122094071,
                                           1.0 / 0.027752748,
                                           1.0 / 0.071544422,
                                           1.0 / 0.171742656,
                                           1.0 / 0.026401153,
                                           1.0 / 0.113714183])
        # loss_function  = MMDFN_FocalLoss(gamma=0.5, alpha=loss_weights if args.class_weight else None)
        # loss_function = FocalLoss()
        loss_function = PCGNet_FocalLoss(gamma=0.5, alpha=loss_weights)
    # params_to_optimize = [
    #     {'params': [param for name, param in model.named_parameters() if name != 'smax_fc.weight' and name != 'smax_fc.bias'], 'lr': args.lr},
    #     {'params': model.smax_fc.parameters(), 'lr': args.lr/10}
    # ]
    # optimizer = optim.Adam(params_to_optimize, weight_decay=args.l2)     

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)    
    if args.Dataset == 'MELD':
        train_loader, valid_loader, test_loader = get_MELD_loaders(valid=0.0, batch_size=args.batch_size, num_workers=2)
    elif args.Dataset == 'IEMOCAP':
        train_loader, valid_loader, test_loader = get_IEMOCAP_loaders(valid=0.0, batch_size=args.batch_size, num_workers=2)
    best_fscore, best_loss, best_label, best_pred, best_mask, best_epoch, best_acc = None, None, None, None, None, None, None
    all_fscore, all_acc, all_loss = [], [], []
    if args.testing:
        state = torch.load("best_model.pth.tar")
        model.load_state_dict(state)
        print('testing loaded model')
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(args, model, loss_function, test_loader, 0, cuda, args.modals, dataset=args.Dataset)
        print('test_acc:',test_acc,'test_fscore:',test_fscore)
    for e in range(args.epochs):
        start_time = time.time()
        train_loss, train_acc, _, _, train_fscore, _, _, _, _, _ = train_or_eval_graph_model(args, model, loss_function, train_loader, e, cuda, args.modals, optimizer, True, dataset=args.Dataset)
        print("train time = ", time.time() - start_time)
        valid_loss, valid_acc, _, _, valid_fscore, _, _, _, _, _ = train_or_eval_graph_model(args, model, loss_function, valid_loader, e, cuda, args.modals, dataset=args.Dataset)
        inference_start_time = time.time()
        test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(args, model, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)
        print("inference time = ", time.time() - inference_start_time)
        all_fscore.append(test_fscore)
        if best_loss == None or best_loss > test_loss:
            best_loss = test_loss
            # best_loss, best_label, best_pred = test_loss, test_label, test_pred
        if best_fscore == None or best_fscore < test_fscore:
            best_epoch = e + 1
            best_fscore = test_fscore
            best_label, best_pred, best_acc = test_label, test_pred, test_acc
            #test_loss, test_acc, test_label, test_pred, test_fscore, _, _, _, _, _ = train_or_eval_graph_model(model, loss_function, test_loader, e, cuda, args.modals, dataset=args.Dataset)
        if args.use_wandb:
            wandb.log({'epoch':e+1, 'best_fscore': best_fscore, 'best_acc': best_acc, 'test_acc': test_acc, 'test_fscore': test_fscore, 'test_loss': test_loss, 'train_loss': train_loss, 'train_fscore': train_fscore, 'seed': args.seed, 'time': round(time.time()-start_time, 2)})
            
        print('epoch: {}, train_loss: {}, train_acc: {}, train_fscore: {}, test_loss: {}, test_acc: {}, test_fscore: {}, time: {} sec'.\
                format(e+1, train_loss, train_acc, train_fscore, test_loss, test_acc, test_fscore, round(time.time()-start_time, 2)))
        if (e+1)%10 == 0:
            print ('----------best F-Score:', max(all_fscore))
            print(classification_report(best_label, best_pred, sample_weight=best_mask,digits=4))
            print(confusion_matrix(best_label,best_pred,sample_weight=best_mask))
    print("best epoch: {}, best_fscore: {}".format(best_epoch, best_fscore))






def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default='LSTM', help='base recurrent model, must be one of DialogRNN/LSTM/GRU')
    parser.add_argument('--windowp', type=int, default=10, help='context window size for constructing edges in graph model for past utterances')
    parser.add_argument('--windowf', type=int, default=10, help='context window size for constructing edges in graph model for future utterances')
    parser.add_argument('--batch_size', type=int, default=32, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E', help='number of epochs')
    parser.add_argument('--class_weight', type=str2bool, default=True, help='use class weights')
    parser.add_argument('--use_residue', type=str2bool, default=False, help='whether to use residue information or not')
    parser.add_argument('--modals', default='avl', help='modals to fusion')
    parser.add_argument('--Dataset', default='IEMOCAP', help='dataset to train and test')
    parser.add_argument('--testing', type=str2bool, default=False, help='testing')
    # base model
    parser.add_argument('--backbone', default='GCN', type=str, choices=['GCN', 'GAT', 'M3Net', 'MMGCN'])
    parser.add_argument('--norm', default='LN2', help='NORM type')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00003, metavar='L2', help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='dropout', help='dropout rate')
    parser.add_argument('--use_speaker', type=str2bool, default=True, help='whether to use speaker embedding')
    parser.add_argument('--av_using_lstm', type=str2bool, default=False, help='whether to use lstm in acoustic and visual modality')
    parser.add_argument('--hidden_dim', type=int, default=512, help='hidden_dim')
    # denoise
    # parser.add_argument('--denoise_type', default='hard', choices=['hard', 'soft'], help='denoise type')
    parser.add_argument('--num_L', type=int, default=4, help='num_denoise')
    parser.add_argument('--use_residue_denoise', type=str2bool, default=False, help='whether to use residue information or not')
    parser.add_argument('--denoise_dropout', type=float, default=0.4, help='denoise_dropout')
    parser.add_argument('--penalty_weight_coff', type=float, default=0.0001)
    parser.add_argument('--gamma', type=float, default=-0.95)
    parser.add_argument('--zeta', type=float, default=1.05)
    parser.add_argument('--temperature', type=float, default=2)
    # nodeformer
    parser.add_argument('--use_gumbel', type=str2bool, default=True, help='whether to use gumbel softmax or not')
    parser.add_argument('--num_K', type=int, default=4, help='num_nodeformer')
    parser.add_argument('--gumbel_k', type=int, default=30, help='sample_time_for_nodeformer')
    parser.add_argument('--nodeformer_heads', type=int, default=4, help='nodeformer_heads')
    parser.add_argument('--nb_features_dim', type=int, default=256)
    parser.add_argument('--tau', type=float, default=0.25, help='tau for gumbel softmax')
    parser.add_argument('--nodeformer_dropout', type=float, default=0.4, help='nodeformer_dropout')
    parser.add_argument('--use_residue_nodeformer', type=str2bool, default=False, help='whether to use residue information or not')
    parser.add_argument('--use_jk_nodeformer', type=str2bool, default=False, help='whether to use jk in nodeformer')
    parser.add_argument('--link_loss_coff', type=float, default=0.05)
    parser.add_argument('--link_loss_epoch', type=int, default=10)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--random_seed', type=str2bool, default=False, help='whether to use random seed or not')
    # wandb
    parser.add_argument('--use_wandb', type=str2bool, default=False, help='whether to use wandb or not')
    args = parser.parse_args()

    if args.random_seed:
        args.seed = random.randint(0, 100000)
    return args





if __name__ == '__main__':



    args = parse_arguments()
    if args.use_wandb:
        run = wandb.init(config=args, name = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), save_code=True)
        print("real_config", run.config)
    try:
        train_args(args, run.id if args.use_wandb else None)
    except RuntimeError as e:
        if args.use_wandb:
            if 'out of memory' in str(e):
                print('OOM happened!')
                wandb.run.notes = 'OOM error occurred.'
                wandb.run.tags += tuple(['OOM'])
                torch.cuda.empty_cache()
        else:
            raise e
        
    
    if args.use_wandb:
        run.finish()
