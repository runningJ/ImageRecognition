import torch
import torchvision

print("current torch version is {}".format(torch.__version__))
print("current torchvision version is {}".format(torchvision.__version__))

import sys
from models.resnet import *
from torchvision import datasets, transforms
import os
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import time
import copy
from torch.nn import DataParallel
import argparse

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def RunTrain(model, criterion, optimizer, scheduler, dataloaders,dataset_sizes,model_save_path,num_epochs=20):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            each_epoch_iter = 0
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            print(optimizer.state_dict()['param_groups'][0]['lr'])

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                if dist.get_rank()== 0:
                    save_path = os.path.join(model_save_path,'best_net.pt')
                    torch.save(model.module.state_dict(), save_path)
                elif dist.get_rank() == -1:
                    save_path = os.path.join(model_save_path,'best_net.pt')
                    torch.save(model.state_dict(),save_path)
        

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)
    #distribute mode or single gpu model
    if dist.get_rank()== 0:
        save_path = os.path.join(model_save_path,'best_net.pt')
        torch.save(model.module.state_dict(), save_path)
    elif dist.get_rank() == -1:
        save_path = os.path.join(model_save_path,'best_net.pt')
        torch.save(model.state_dict(),save_path)

def Train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder",type=str,default='/home/jl/datasets/oilrecognition',help='train and val folder path')
    parser.add_argument("--local_rank", type=int,default=-1,help='DDP parameter, do not modify')
    parser.add_argument("--distribute",action='store_true',help='whether using multi gpu train')
    parser.add_argument("--distribute_mode",type=str,default='DDP',help="using which mode to ")
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=64, help='total batch size for all GPUs')
    parser.add_argument("--save_path",type=str,default= "./save",help="the path used to save state_dict")
    opt = parser.parse_args()

    print("**********************************Training Parameters***************************************")
    print(opt)
    print("********************************************************************************************")
    
    if not os.path.exists(opt.image_folder):
        print("{}image folder is not exists".format(opt.image_folder))
        exit(0)
    if not os.path.exists(opt.save_path):
        print("{} image folder is not exists then create it")
        os.mkdir(opt.save_path)
    
    #data transform
    data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }

    #DPP model
    if opt.distribute and opt.local_rank != -1:
        global device
        torch.cuda.set_device(opt.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        device = torch.device('cuda', opt.local_rank)

    if opt.distribute:
        assert opt.batch_size % torch.cuda.device_count() == 0,'--batch-size must be multiple of CUDA device count'
        opt.batch_size = opt.batch_size//torch.cuda.device_count()
    
    #data loader
    data_dir = opt.image_folder
    image_datasets={}
    image_datasets['train'] = datasets.ImageFolder(os.path.join(data_dir, 'train'),data_transforms['train'])
    image_datasets['val'] = datasets.ImageFolder(os.path.join(data_dir, 'val'),data_transforms['val'])
    
    if opt.distribute and opt.local_rank != -1:
        word_size = dist.get_world_size()
        train_sampler = torch.utils.data.distributed.DistributedSampler(image_datasets['train'],num_replicas = word_size,rank = opt.local_rank)
    else:
        train_sampler = None
    
    print("batch size is : {}".format(opt.batch_size))
    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=opt.batch_size,shuffle=(train_sampler is None), num_workers=4, pin_memory=True, sampler=train_sampler)
    dataloaders['val'] =  torch.utils.data.DataLoader(image_datasets['val'], batch_size=opt.batch_size,shuffle = False,num_workers=4)
    
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print("train dataset size is {} and val dataset size is {}".format(dataset_sizes['train'],dataset_sizes['val']))
    class_names = image_datasets['train'].classes
    class_size = len(class_names)

    #get model
    model = DefaultResnet50(pretrained=True) #pretrained model
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, class_size)
    
    #DP model
    if opt.distribute and opt.local_rank == -1:
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = DataParallel(model)
    
    # DDP mode
    if opt.distribute and opt.local_rank != -1:
        model.to(device)
        model = DDP(model, device_ids=[opt.local_rank])
    else:
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.0005,nesterov=True)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    #do train
    RunTrain(model, criterion, optimizer, exp_lr_scheduler, dataloaders, dataset_sizes,opt.save_path,num_epochs=opt.epochs)


if __name__ =="__main__":
    Train()

