from dataset import MRData
from models import MRnet
from config import config
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import _get_trainable_params, _run_eval, _confusion_metrics
import time
import torch.utils.data as data

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
    train_data = MRData(task='abnormal',train=True)
    train_loader = data.DataLoader(
        train_data, batch_size=1, num_workers=4, shuffle=True
    )

    print('Loading Validation Dataset...')
    val_data = MRData(task='abnormal',train=False)
    val_loader = data.DataLoader(
        val_data, batch_size=1, num_workers=4, shuffle=False
    )

    print('Initializing Model...')
    model = MRnet()
    model = model.cuda()

    print('Initializing Loss Method...')
    # TODO : maybe take a wiegthed loss
    criterion = F.cross_entropy

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

        y_preds = []
        y_ground = []

        # TODO : add tqdm here as well ? or time remaining ?
        for i,batch in enumerate (train_loader):

            images, label, weights = batch

            # Send to GPU
            images = [img.cuda() for img in images]
            label = label.cuda()
            weights = weights.cuda()

            # squash the first dim due to dataloader
            weights = weights.squeeze(dim = 0)

            # zero out all grads
            optimizer.zero_grad()

            output = model(images)

            # add to confusion matrix data
            y_preds.append(torch.argmax(output,dim=1).item())
            y_ground.append(label.item())

            # Calculate Loss cross Entropy
            loss = criterion(output, label, weights)

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
        precision, recall, f1_score = _confusion_metrics(y_preds, y_ground)

        writer.add_scalar("Train/F1_Score",f1_score,epoch)
        writer.add_scalar("Train/Recall",recall,epoch)
        writer.add_scalar("Train/Precision",precision,epoch)

        
        validation_loss, precision, recall, f1_score = _run_eval(model, val_loader, criterion, config)
        writer.add_scalar("Val/Loss",validation_loss,epoch)
        writer.add_scalar("Val/F1_Score",f1_score,epoch)
        writer.add_scalar("Val/Recall",recall,epoch)
        writer.add_scalar("Val/Precision",precision,epoch)

        print('Average Validation Loss at Epoch {} : {:.4f}'.format(epoch+1, validation_loss))
        print('Val Precision at Epoch {} : {:.4f}'.format(epoch+1, precision))
        print('Val Recall at Epoch {} : {:.4f}'.format(epoch+1, recall))
        print('Val F1-Score at Epoch {} : {:.4f}'.format(epoch+1, f1_score))

        if best_accuracy < f1_score :
            best_accuracy = f1_score
            # Save this model
            if export:
                model._save_model(f1_score, config, epoch)

        total_loss = 0.0
        print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch+1, num_epochs, time.time() - epoch_start_time))
        
    writer.close()

if __name__ == '__main__':

    print('Training Configuration')
    print(config)

    train(config=config)

    print('Training Ended...')
