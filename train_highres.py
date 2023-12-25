import os
import argparse
from models.SR import SRModel
from Networks.ms_ssim_torch import *
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import time
import col_of_def.dataset as dataset 
import col_of_def.prepare as prepare
#from datasets import Datasets, TestKodakDataset
import numpy as np
from tensorboardX import SummaryWriter
from Meter import AverageMeter
import logging
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
torch.manual_seed(0)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark=True
torch.cuda.empty_cache()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
# gpu_num = 4
gpu_num = torch.cuda.device_count()
cur_lr = base_lr = 1e-4#  * gpu_num
train_lambda = 8192
print_freq = 50
cal_step = 40
warmup_step = 0#  // gpu_num
batch_size = 8
tot_epoch = 150000
tot_step = 150000
decay_interval  = 55000
decay_interval2 = 125000
lr_decay = 0.1
lr_decay2 = 0.01
image_size = 256
logger = logging.getLogger("ImageUpscaling")
tb_logger = None
global_step = 0
save_model_freq = 5000

train_root = './Train_Image'
test_root = './Val_Image'

parser = argparse.ArgumentParser(description='Pytorch implement for ImageUpscaling')

parser.add_argument('-n', '--name', default='',
        help='output training details')
parser.add_argument('-p', '--pretrain', default = '',
        help='load pretrain model')
parser.add_argument('--test', action='store_true')

parser.add_argument('--seed', default=234, type=int, help='seed for random functions, and network initialization')

def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))

def save_model_train(model, iter, name):
    previous_iter = iter - 2000
    
    check_directory = os.path.join(name,f"iter_{previous_iter}.pth.tar")
    
    if iter > 149900:
        torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))
    else:
        if os.path.isfile(check_directory):
            os.remove(check_directory)
    
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))

def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0




def adjust_learning_rate(optimizer, global_step, decay_lr):
    global cur_lr
    global warmup_step

    # lr = base_lr * (lr_decay ** (global_step // decay_interval))
    lr = base_lr * decay_lr
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    cur_lr = lr
lpips = LearnedPerceptualImagePatchSimilarity(net_type='squeeze',normalize=True).to(device)
scaler = torch.cuda.amp.GradScaler() 
def train(epoch, global_step):
    logger.info("Epoch {} begin".format(epoch))
    net.train()
    global optimizer
    elapsed, losses, psnrs, bpps, bpp_features, bpp_zs, mse_losses = [AverageMeter(print_freq) for _ in range(7)]
    # model_time = 0
    # compute_time = 0
    # log_time = 0
    #for batch_idx, (input,_) in enumerate(train_loader,1):
    for batch_idx, (image,_) in enumerate(train_loader,1):
        start_time = time.time()
        global_step += 1
        
        image = image.to(device)
        train_image = F.interpolate(image, scale_factor=0.5, mode='bicubic', align_corners=False)
        output = net(train_image)
        image = torch.clamp(image, 0,1)
        output = torch.clamp(output, 0,1)
        mse_loss = torch.mean((output - image).pow(2))
        
        if global_step < 1250:
            distortion = mse_loss
            #distortion =  (1-lpips(image,output)) + 3 * (1 - ms_ssim(image.detach(), output, data_range=1.0, size_average=True))
        else:
            distortion =  (lpips(image,output)) + 2*(1 - ms_ssim(image.detach(), output, data_range=1.0, size_average=True))
            #distortion =  (1-lpips(image,output))

        if global_step == 500 or global_step == 550:
            torchvision.utils.save_image(output, "output_image/"+str(global_step)+"high_res.jpg")
        
        """
        elif global_step < 55000:
            image = torch.clamp(image, 0, 1)
            output = torch.clamp(output, 0, 1)
            distortion =  5 * (1 - ms_ssim(image.detach(), output, data_range=1.0, size_average=True))
        else:
            image = torch.clamp(image, 0, 1)
            output = torch.clamp(output, 0, 1)
            distortion =  (1-lpips(image,output)) + 3 * (1 - ms_ssim(image.detach(), output, data_range=1.0, size_average=True))
        """
        #with torch.cuda.amp.autocast(): 
            
        rd_loss = distortion
        optimizer.zero_grad()
        #scaler.scale(rd_loss).backward()
        rd_loss.backward()
        def clip_gradient(optimizer, grad_clip):
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None:
                        param.grad.data.clamp_(-grad_clip, grad_clip)
        clip_gradient(optimizer, 5)
        
        #scaler.step(optimizer) 
        optimizer.step()
        # model_time += (time.time()-start_time)
        if (global_step % cal_step) == 0:
            # t0 = time.time()
            if mse_loss.item() > 0:
                psnr = 10 * (torch.log(1 * 1 / mse_loss) / np.log(10))
                psnrs.update(psnr.item())
            else:
                psnrs.update(100)

            # t1 = time.time()
            elapsed.update(time.time() - start_time)
            losses.update(rd_loss.item())
            
            # t2 = time.time()
            # compute_time += (t2 - t0)
        if (global_step % print_freq) == 0:
            # begin = time.time()
            tb_logger.add_scalar('lr', cur_lr, global_step)
            tb_logger.add_scalar('loss', losses.avg, global_step)
            process = global_step / tot_step * 100.0
            log = (' | '.join([
                f'Step [{global_step}/{tot_step}={process:.2f}%]',
                f'Epoch {epoch}',
                f'Time {elapsed.val:.3f} ({elapsed.avg:.3f})',
                f'Lr {cur_lr}',
                f'Total Loss {losses.val:.6f} ({losses.avg:.6f})',
            ]))
            logger.info(log)
            
            # log_time = time.time() - begin
            # print("Log time", log_time)
            # print("Compute time", compute_time)
            # print("Model time", model_time)
        if global_step > decay_interval:
            adjust_learning_rate(optimizer, global_step, lr_decay)
       
        if (global_step % 1000) == 0:
            torchvision.utils.save_image(output, "output_image/"+str(global_step)+"high_res.jpg")
        if (global_step % 2000) == 0:
            save_model_train(model, global_step, save_path)
            torchvision.utils.save_image(output, "output_image/"+str(global_step)+"high_res.jpg")
          
        if (global_step % save_model_freq) == 0:
            save_model(model, global_step, save_path)
            testKodak(global_step)
            net.train()
            torch.cuda.empty_cache()
        #scaler.update()
        
    return global_step


def testKodak(step):
    with torch.no_grad():
        #test_dataset = TestKodakDataset(data_dir='./dataset/test')
        
        test_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        test_dataset = ImageFolder(root='./Val_Image', # Input image path
                           transform=test_transform) 

        test_loader = DataLoader(dataset=test_dataset, shuffle=False, batch_size=1, pin_memory=True, num_workers=1)
        
        #test_loader ,test_dataset= prepare.prepare_dataset_Kodak(batch_size=1, rootpath="./Kodak/")
        net.eval()
        sumBpp = 0
        sumPsnr = 0
        sumMsssim = 0
        sumMsssimDB = 0
        sumTime = 0
        cnt = 0
        for batch_idx, (image,_) in enumerate(test_loader,1):
            image = image.to(device)
            time_start = time.perf_counter()
            output = net(image)
            time_end = time.perf_counter()
            image = torch.clamp(image, 0, 1)
            output = torch.clamp(output, 0, 1)
            torchvision.utils.save_image(output, "outputKodak/"+str(cnt)+"highres.png",alpha=True)
            output = F.interpolate(output, scale_factor=0.5, mode='bicubic', align_corners=False)
            output = torch.clamp(output, 0, 1)
            mse_loss = 1-lpips(image,output)
            
            
            mse_loss = torch.mean(mse_loss)
            psnr = 10 * (torch.log(1. / mse_loss) / np.log(10))
            #psnr = mse_loss
            
            sumPsnr += psnr
            
            msssim = ms_ssim(image.detach(), output,data_range=1.0, size_average=True)
            msssimDB = -10 * (torch.log(1-msssim) / np.log(10))
            sumMsssimDB += msssimDB
            sumMsssim += msssim
            tim = time_end - time_start
            sumTime += tim
            logger.info("Time:{:.6f}, LPIPS:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(tim,psnr, msssim, msssimDB))
            cnt += 1

        logger.info("Test on Kodak dataset: model-{}".format(step))

        sumPsnr /= cnt
        sumMsssim /= cnt
        sumMsssimDB /= cnt
        sumTime /= cnt
        logger.info("Dataset Average result---Time:{:.6f}, LPIPS:{:.6f}, MS-SSIM:{:.6f}, MS-SSIM-DB:{:.6f}".format(sumTime,sumPsnr, sumMsssim, sumMsssimDB))
        if tb_logger !=None:
            logger.info("Add tensorboard---Step:{}".format(step))
            tb_logger.add_scalar("PSNR_Test", sumPsnr, step)
            tb_logger.add_scalar("MS-SSIM_Test", sumMsssim, step)
            tb_logger.add_scalar("MS-SSIM_DB_Test", sumMsssimDB, step)
        else:
            logger.info("No need to add tensorboard")

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(seed=args.seed)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s] %(message)s')
    formatter = logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')
    stdhandler = logging.StreamHandler()
    stdhandler.setLevel(logging.INFO)
    stdhandler.setFormatter(formatter)
    logger.addHandler(stdhandler)
    dd = 1
    save_path = os.path.join('checkpoints', args.name)
    if args.name != '':
        os.makedirs(save_path, exist_ok=True)
        filehandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        filehandler.setLevel(logging.INFO)
        filehandler.setFormatter(formatter)
        logger.addHandler(filehandler)
    logger.setLevel(logging.INFO)
    logger.info("image highresolution training")
    

    model = SRModel()
    if args.pretrain != '':
        logger.info("loading model:{}".format(args.pretrain))
        global_step = load_model(model, args.pretrain)
    net = model.to(device)
    #net = torch.nn.DataParallel(net, list(range(gpu_num)))
    parameters = net.parameters()
    if args.test:
        testKodak(global_step)
        exit(-1)
    optimizer = optim.Adam(parameters, lr=base_lr)
    # save_model(model, 0)
    global train_loader
    tb_logger = SummaryWriter(os.path.join(save_path, 'events'))
    #train_data_dir = './P3M/MASKpatches'
    #train_dataset = Datasets(train_data_dir, image_size)
    
    train_root = './openimages/train'
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(256),
        #transforms.RandomHorizontalFlip(p=0.5),
        #transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor(),
    ])
    train_dataset = ImageFolder(root=train_root, # 画像が保存されているフォルダのパス
                           transform=train_transform) # Tensorへの変換
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=2)
    
    #train_loader,train_dataset = prepare.prepare_dataset_train_P3M(batch_size=batch_size, rootpath="./P3M", height=256, width=256)
    steps_epoch = global_step // (len(train_dataset) // (batch_size))# * gpu_num))
    torch.save(net.state_dict(),f'checkpoint.pt')
    for epoch in range(steps_epoch, tot_epoch):
        if global_step >= decay_interval:
            adjust_learning_rate(optimizer, global_step, lr_decay)
        if global_step >= decay_interval2:
            adjust_learning_rate(optimizer, global_step, lr_decay2)
        if global_step > tot_step:
            save_model(model, global_step, save_path)
            break
        global_step = train(epoch, global_step)
        save_model(model, global_step, save_path)
