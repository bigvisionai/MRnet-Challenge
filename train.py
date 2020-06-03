from dataset import MRData
from models import MRnet
from config import config
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import _get_trainable_params, _run_eval, _confusion_metrics
import time
import torch.utils.data as data
from sklearn import metrics

"""Performs training of a specified model.
    
Input params:
    config_file: Takes in configurations to train with 
"""

# TODO : Add proper checks for gpu

def train(config : dict, export=True):
    """
    Function where actual training takes place

    Args:
        config (dict) : Configuration to train with
        export (Boolean) : Whether to export model to disk or not
    """
    
    print('Starting to Train Model...')

    print('Loading Train Dataset...')
    train_data = MRData(task='acl',train=True)
    train_loader = data.DataLoader(
        train_data, batch_size=1, num_workers=4, shuffle=True
    )

    print('Loading Validation Dataset...')
    val_data = MRData(task='acl',train=False)
    val_loader = data.DataLoader(
        val_data, batch_size=1, num_workers=4, shuffle=False
    )

    print('Initializing Model...')
    model = MRnet()
    model = model.cuda()

    print('Initializing Loss Method...')
    # Create critierion for Validation 
    train_weight = torch.tensor([train_data.weights[1]]).cuda()
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=train_weight)

    print('Setup the Optimizer')
    optimizer = torch.optim.Adam(_get_trainable_params(model),lr=config['lr'])

    scheduleLR=torch.optim.lr_scheduler.StepLR(optimizer,step_size=5,gamma=0.5)

    starting_epoch = config['starting_epoch']
    num_epochs = config['max_epoch']

    best_accuracy = 0.0

    print('Starting Training')

    writer = SummaryWriter(comment=f'lr={config["lr"]} dropout')

    # TODO : add tqdm with support with notebook
    for epoch in range(starting_epoch, num_epochs):

        epoch_start_time = time.time()  # timer for entire epoch

        train_iterations = len(train_loader)
        train_batch_size = config['batch_size']

        num_batch = 0

        # loss for the epoch
        total_loss = 0.0

        model.train()

        y_probs = []
        y_ground = []

        # TODO : add tqdm here as well ? or time remaining ?
        for i,batch in enumerate (train_loader):

            images, label, _ = batch

            # Send to GPU
            images = [x.cuda() for x in images]
            label = label.cuda()
            label = label.float()
            label = label.unsqueeze(dim = 0)

            # zero out all grads
            optimizer.zero_grad()

            output = model(images)

            # add to ROC data
            output = torch.sigmoid(output)
            y_probs.append(output.item())
            y_ground.append(label.item())

            # Calculate Loss cross Entropy
            loss = criterion(output, label)

            # add loss to epoch loss
            total_loss += loss.item()

            # Do backprop
            loss.backward()

            # Change wieghts
            optimizer.step()
            
            # Write to tensorboard
            for name,params in model.named_parameters():
                if params.requires_grad:
                    writer.add_histogram(name,params,i)
            
            # Flush to disk
            writer.flush()

            # Log some info
            if num_batch % config['log_freq'] == 0:
                print('{}/{} Epoch : {}/{} Batch Iter : Batch Loss {:.4f}'.format(
                    epoch+1, num_epochs, num_batch+1, len(train_loader), loss.item()
                ))

                # print('GT : ', label)
                # print('Pred : ',output)
                # print('-'*10)
            
            num_batch += 1

        # Updating LR
        scheduleLR.step()

        # spit train loss details
        average_train_loss = total_loss / len(train_loader)
        print('Average Train Loss at Epoch {} : {:.4f}'.format(epoch+1, average_train_loss))
        writer.add_scalar("Train/Avg Loss",average_train_loss,epoch)
        writer.add_scalar("Train/Total Loss",total_loss,epoch)

        train_auc = metrics.roc_auc_score(y_ground, y_probs)
        writer.add_scalar("Train/AUC",train_auc,epoch)
        print('Train AUC at Epoch {} : {:.4f}'.format(epoch+1, train_auc))

        # Create critierion for Validation 
        val_weight = torch.tensor([val_data.weights[1]]).cuda()
        val_criterion = torch.nn.BCEWithLogitsLoss(pos_weight=val_weight)
        
        validation_loss, y_probs, y_ground = _run_eval(model, val_loader, val_criterion, config)
        val_auc = metrics.roc_auc_score(y_ground, y_probs)
        writer.add_scalar("Val/Loss",validation_loss,epoch)
        writer.add_scalar("Val/AUC",val_auc,epoch)

        print('Average Validation Loss at Epoch {} : {:.4f}'.format(epoch+1, validation_loss))
        print('Val AUC at Epoch {} : {:.4f}'.format(epoch+1, val_auc))

        if best_accuracy < val_auc :
            best_accuracy = val_auc
            # Save this model
            if export:
                model._save_model(val_auc, config, epoch)

        total_loss = 0.0
        print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch+1, num_epochs, time.time() - epoch_start_time))
        
    writer.close()

if __name__ == '__main__':

    print('Training Configuration')
    print(config)

    train(config=config)

    print('Training Ended...')
