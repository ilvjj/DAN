from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import gvb_loss
from data_loader import *
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net, Advers_net,calc_coeff, weights_init_classifier#, domaincl
from utils import *
from loss import OriTripletLoss
from tensorboardX import SummaryWriter
import copy
from memory_table import MemoryTable
from metricloss import Metric_loss
#from center_loss import Centerloss
import tqdm
parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='tw-to-regdb', help='dataset name]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
# parser.add_argument('--resume', '-r', default='', type=str,
#                     help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='./save_model/', type=str,
                    help='model save path')
parser.add_argument('--table', '-tb', default='', type=str,help='table from checkpoint')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='./log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='./log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=4, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=32, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='base', type=str,
                    metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=3, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='1', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--table_path', default='./save_table/', type=str,
                    help='model save path')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/media/cqu/D/XYG/Cross-domain/Datasets/SYSU-MM01/'
    noise_path = '/media/cqu/D/XYG/Cross-domain/Datasets/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = '/media/cqu/D/XYG/Cross-domain/Datasets/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    noise_path = '/media/cqu/D/XYG/Cross-domain/Datasets/RegDB_noise/'
    aug_path = '/media/cqu/D/XYG/Cross-domain/Datasets/RegDB_fre/'
    test_mode = [2, 1]  # visible to thermal
elif dataset == 'regdb-to-tw':
    data_path = '/media/cqu/D/XYG/Cross-domain/Datasets/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    noise_path = '/media/cqu/D/XYG/Cross-domain/Datasets/RegDB_noise/'
    aug_path = '/media/cqu/D/XYG/Cross-domain/Datasets/RegDB_fre/'
    test_mode = [2, 1]  # visible to thermal
checkpoint_path = args.model_path
table_path = args.table_path + args.dataset + '/' + 'trial_{}'.format(args.trial)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)
if not os.path.isdir(table_path):
    os.makedirs(table_path)

suffix = dataset
if args.method=='agw':
    suffix = suffix + '_agw_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
else:
    suffix = suffix + '_base_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)


if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

if dataset == 'tw':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_osall.txt')

vis_log_dir = args.vis_log_path + suffix + '/'
suffix += 'allmethodorilea'
if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    trainset_target = SYSUDataastarget(noise_path, transform=transform_train)
    trainset_all = ALLSYSU(noise_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    trainset_target = RegDBData_aug(aug_path, args.trial, transform=transform_train)
    trainset_all = ALL_RegDBData_aug(aug_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb_aug(aug_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb_aug(aug_path, trial=args.trial, modal='thermal')
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))


elif dataset == 'regdb-to-tw':
    # training set

    trainset = RegDBData_source(data_path, args.trial, transform=transform_train)

    trainset_target = ThermalWorld_target(transform=transform_train)
    trainset_all = ALL_reg_to_tw(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_thermworld(modal='visible')
    gall_img, gall_label = process_test_thermworld(modal='thermal')
    gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
    queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))


# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
base_net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch=args.arch)
adv_net = Advers_net(2048, 1024)

base_net.to(device)
adv_net.to(device)

cudnn.benchmark = True

if len(args.resume) > 0:
    #model_path = checkpoint_path + args.resume
    model_path =args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
       # start_epoch = checkpoint['epoch']
        base_net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()

if args.method == 'agw':
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = TripletLoss_WRT()

else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)	
    metric_loss = Metric_loss(batch_size=loader_batch)
    # centers_loss = Centerloss(batch_size=loader_batch)

criterion_id.to(device)
criterion_tri.to(device)
metric_loss.to(device)
# centers_loss.to(device)
if args.optim == 'sgd':
    ignored_params = list(map(id, base_net.bottleneck.parameters())) + list(map(id, base_net.classifier.parameters()))       
    base_params = filter(lambda p: id(p) not in ignored_params, base_net.parameters())
    adv_params = adv_net.parameters()
    optimizer = optim.SGD([
        {'params': base_params, 'lr':  0.1*args.lr},
        {'params': adv_params, 'lr':  args.lr},
        {'params': base_net.bottleneck.parameters(), 'lr': args.lr},
        {'params': base_net.classifier.parameters(), 'lr': args.lr},],
        weight_decay=5e-4, momentum=0.9, nesterov=True)



def adjust_learning_rate(optimizer, epoch):
    lr = args.lr
    if epoch < 10:
        lr = args.lr #* (epoch + 1) / 5
    elif epoch >= 10 and epoch < 15:
        lr = args.lr* 0.1


    optimizer.param_groups[0]['lr'] = 0.1 * args.lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = args.lr

    return lr

def loss_fn(outputs, targets, label, memory_tb, t=1, ld=0.4, modal=0):
    if modal==0:
        prob = -ld * torch.log(memory_tb.vis_probability(outputs, targets, t))
        reliables = torch.LongTensor(memory_tb.vis_reliables[targets]).T.tolist()
        reliables_prob = torch.stack(list(map(lambda r: memory_tb.vis_reliables_probability(outputs, r,label, t), reliables))).T
        reliables_prob = -((1 - ld) / len(reliables)) * (reliables_prob.sum(dim=1))
    else:
        prob = -ld * torch.log(memory_tb.nir_probability(outputs, targets, t))
        reliables = torch.LongTensor(memory_tb.nir_reliables[targets]).T.tolist()
        reliables_prob = torch.stack(list(map(lambda r: memory_tb.nir_reliables_probability(outputs, r,label, t), reliables))).T
        reliables_prob = -((1 - ld) / len(reliables)) * (reliables_prob.sum(dim=1))
    return (prob + reliables_prob).mean()

classifiert = nn.Linear(2048, 206, bias=False).cuda()
classifiert.apply(weights_init_classifier)

def train(epoch,trainloader,memory):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    Adv_loss = AverageMeter()
    entropy_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    correct2 = 0
    total = 0
    total2 = 0
    totald = 0
    correctd = 0
    # switch to train mode
    base_net.train()
    adv_net.train()

    end = time.time()

    for batch_idx, (input1, input2, label1, label2,input3, input4, label3, label4, indexs) in enumerate(trainloader):

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        input3 = Variable(input3.cuda())
        input4 = Variable(input4.cuda())

        label1 = Variable(label1.cuda())
        label2 = Variable(label2.cuda())
        label3 = Variable(label3.cuda())
        label4 = Variable(label4.cuda())

        data_time.update(time.time() - end)

        labels = torch.cat((label1,label2), 0)
        labels2 = torch.cat((label3, label4), 0)
        loss_mean = 0
        loss_va = 0

        feat, norm_feat, out = base_net(input1, input2, modal=0, domain=0)
        feat2, norm_feat2, out2 = base_net(input3, input4, modal=0, domain=1)
        out2 = classifiert(out2)
        l3 = label3.cpu().detach().numpy()
        l3u = np.unique(l3)

        l4 = label4.cpu().detach().numpy()
        l4u = np.unique(l4)

        for i in l3u:
            if i in l4u:
                apre = torch.tensor(np.full((label3.size()), i) == l3 + 0)
                apre = apre.view(label3.size(0),1)
                apre = apre.expand(norm_feat2[:label3.size(0)].size()).cuda()
                bpre = torch.tensor(np.full((label4.size()), i) == l4 + 0)
                bpre = bpre.view(label4.size(0), 1)
                bpre = bpre.expand(norm_feat2[:label3.size(0)].size()).cuda()

                a = (apre * norm_feat2[:label3.size(0)]).mean(0)

                b = (bpre * norm_feat2[:label3.size(0)]).mean(0)
                c = (apre * norm_feat2[:label3.size(0)]).std(0)
                d = (bpre * norm_feat2[:label3.size(0)]).std(0)

                loss_mean += torch.norm(a - b)
                loss_va += torch.norm(c - d)

        softmax_out1 = nn.Softmax(dim=1)(out)
        softmax_out2 = nn.Softmax(dim=1)(out2) 




        #原来的迁移损失
        vis_out = torch.cat((softmax_out1[0:input1.size(0),:],softmax_out2[0:input3.size(0),:]),0)
        nir_out =  torch.cat((softmax_out1[input1.size(0):,:],softmax_out2[input3.size(0):,:]),0)


        vis_feat = torch.cat((norm_feat[0:input1.size(0),:],norm_feat2[0:input3.size(0),:]),0)
        nir_feat = torch.cat((norm_feat[input1.size(0):,:],norm_feat2[input3.size(0):,:]),0)
        transfer_loss1, mean_entropy = gvb_loss.GVB([vis_out,vis_feat], adv_net, calc_coeff(epoch), GVBD=1)
        transfer_loss2, mean_entropy= gvb_loss.GVB([nir_out,nir_feat], adv_net, calc_coeff(epoch), GVBD=1)

        loss_id = criterion_id(out, labels)
        loss_tri, _ = criterion_tri(norm_feat, labels)


        loss_softid1 = loss_fn(norm_feat2[0:input3.size(0), :], indexs, label3, memory, modal=0)
        loss_softid2 = loss_fn(norm_feat2[input3.size(0):, :], indexs, label4, memory, modal=1)

        loss = loss_id + loss_tri + transfer_loss1 + transfer_loss2 + 0.1 * loss_softid1 + 0.1 * loss_softid2 + loss_mean + loss_va
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        _, predicted = out.max(1)
        correct += (predicted.eq(labels).sum().item())

        _, predicted2 = out2.max(1)
        correct2 += (predicted2.eq(labels2).sum().item())


        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 *input1.size(0))
        Adv_loss.update(transfer_loss1.item(), 2 * input1.size(0))
        total += labels.size(0)
        total2 += labels2.size(0)
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 500 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'adv_loss: {Adv_loss.val:.4f} ({Adv_loss.avg:.4f}) '
                  'Accu1: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr, 100. * correct / total,
                batch_time=batch_time,train_loss=train_loss,
                id_loss=id_loss, Adv_loss=Adv_loss))


    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    # update memory



def test(epoch):
    # switch to evaluation mode
    base_net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    gall_feat2 = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat2 = base_net(input, input, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat2[ptr:ptr + batch_num, :] = feat2.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    base_net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    query_feat2 = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat2 = base_net(input, input, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat2[ptr:ptr + batch_num, :] = feat2.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat2 = np.matmul(query_feat2, np.transpose(gall_feat2))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
        cmc2, mAP2, mINP2      = eval_regdb(-distmat2, query_label, gall_label)
    elif dataset == 'sysu':

        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc2, mAP2, mINP2      = eval_sysu(-distmat2, query_label, gall_label, query_cam, gall_cam)
    elif dataset == 'regdb-to-tw':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
        cmc2, mAP2, mINP2 = eval_regdb(-distmat2, query_label, gall_label)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc2[0], epoch)
    writer.add_scalar('mAP_att', mAP2, epoch)
    writer.add_scalar('mINP_att', mINP2, epoch)
    return cmc, mAP, mINP, cmc2, mAP2, mINP2

# source dataload
trainset_source = copy.deepcopy(trainset)
sampler_s = IdentitySampler1(trainset_source.train_color_label,trainset_source.train_thermal_label, color_pos, thermal_pos, 4, 3, 0)
trainset_source.cIndex = sampler_s.index1  # color index
trainset_source.tIndex = sampler_s.index2  # thermal index

trainloader_s = data.DataLoader(trainset_source, batch_size=12,sampler=sampler_s, num_workers=args.workers, drop_last=False)

# target dataload

trainloader_t = data.DataLoader(trainset_target, batch_size=12,num_workers=args.workers, drop_last=False)
print('==> Start Training...')
for epoch in range(start_epoch, 16 - start_epoch):
    memory = MemoryTable()
    RGB_output = []
    NIR_output = []
    RGB_center = torch.zeros(206, 2048).cuda()
    NIR_center = torch.zeros(206, 2048).cuda()


    pseudo_RGB = []
    pseudo_NIR = []
    base_net.eval()

    with torch.no_grad():
        print('==> update souce memory center table...')
        for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader_s):
            label_num = len(label1.unique())

            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())
            label1 = Variable(label1.cuda())

            feat1, _ = base_net(input1,input1,1)
            feat2, _ = base_net(input2,input2,2)
            feat1 = feat1.chunk(label_num, 0)
            feat2 = feat2.chunk(label_num, 0)

            for j in range(label_num):
                ##regdb
                RGB_center[label1[j * 4]] = (RGB_center[label1[j * 4]] + torch.mean(feat1[j], dim=0).unsqueeze(0)) / 2
                NIR_center[label2[j * 4]] = (NIR_center[label2[j * 4]] + torch.mean(feat2[j], dim=0).unsqueeze(0)) / 2



        print('==> update target pseudo label...')
        for batch_idx, (input1, input2, label1, _,indexs) in enumerate(trainloader_t):
            label_num = len(label1.unique())    
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())
            label1 = Variable(label1.cuda())


            feat1, _ = base_net(input1,input1,1)
            feat2, _ = base_net(input2,input2,2)
            RGB_dist = torch.cdist(feat1, RGB_center)
            NIR_dist = torch.cdist(feat2, NIR_center)

            RGB_label = torch.argmax(-RGB_dist, dim=1)
            NIR_label = torch.argmax(-NIR_dist, dim=1)
            pseudo_RGB.append(RGB_label)
            pseudo_NIR.append(NIR_label)

        RGB_pseudo = torch.cat(pseudo_RGB)
        NIR_pseudo = torch.cat(pseudo_NIR)

    rgb_list = RGB_pseudo.cpu().numpy().tolist()
    nir_list = NIR_pseudo.cpu().numpy().tolist()
    print (np.unique(rgb_list))
    print(np.unique(nir_list))
    print('==> Preparing Data Loader...')
    color_pos_t, thermal_pos_t = GenIdx2(rgb_list, nir_list)

    # identity sampler
    sampler_all = IdentitySampler3(rgb_list, nir_list, color_pos_t, thermal_pos_t, 4, 3, epoch)

    trainset_all.cIndexs = sampler_s.index1  # color index
    trainset_all.tIndexs = sampler_s.index2  # thermal index

    trainset_all.cIndext = sampler_all.index1  # color index
    trainset_all.tIndext = sampler_all.index2  # thermal index


    trainset_all.pseudo_color_label = rgb_list  # color index
    trainset_all.pseudo_thermal_label = nir_list  # thermal index

    trainloader = data.DataLoader(trainset_all, batch_size=loader_batch,sampler=sampler_s,num_workers=0, drop_last=True)


    print('==> update memory table...')
    RGB_output = []
    GRAY_output = []
    NIR_output = []
    label_output = []
    label_output2 = []
    base_net.eval()
    with torch.no_grad():
        for batch_idx, (_, _, _, _, input1, input2, label1, label2, indexs) in enumerate(trainloader):
            input1 = Variable(input1.cuda())
            input2 = Variable(input2.cuda())
            label1 = Variable(label1.cuda())
            feat1, _, = base_net(input1, input1, 1)
            feat2, _, = base_net(input2, input2, 2)
            RGB_output.append(feat1)
            NIR_output.append(feat2)
            label_output.append(label1)
            label_output2.append(label2)

        RGB_vectors = torch.cat(RGB_output)
        NIR_vectors = torch.cat(NIR_output)


        label_table = torch.cat(label_output)
        label_table2 = torch.cat(label_output2)
        memory.update(RGB_vectors[:2064], NIR_vectors[:2064], label_table[:2064], label_table2[:2064])
        #memory.update(RGB_vectors, NIR_vectors, label_table, label_table2)

        memory.update_reliables()

    # training

    train(epoch, trainloader, memory)


    if epoch % 5 == 0 and epoch > 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP,cmc2, mAP2, mINP2 = test(epoch)
        # save model
        if cmc[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc[0]
            best_epoch = epoch
            state = {
                'net': base_net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'mINP': mINP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        # save model
        if epoch > 10 and epoch % args.save_epoch == 0:
            state = {
                'net': base_net.state_dict(),
                'cmc': cmc2,
                'mAP': mAP2,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

        print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
            cmc2[0], cmc2[4], cmc2[9], cmc2[19], mAP2, mINP2))
        print('Best Epoch [{}]'.format(best_epoch))
