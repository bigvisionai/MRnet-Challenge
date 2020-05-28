from dataset import MRData
from models import MRnet
from config import config
import torch
from utils import _get_trainable_params, _run_eval
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
    model.cuda()

    print('Initializing Loss Method...')
    # TODO : maybe take a wiegthed loss
    criterion = torch.nn.CrossEntropyLoss()
    criterion.cuda()

    print('Setup the Optimizer')
    # TODO : Add other hyperparams as well
    optimizer = torch.optim.Adam(_get_trainable_params(model), lr=config['lr'])

    starting_epoch = config['starting_epoch']
    num_epochs = config['max_epoch']

    best_accuracy = 0.0

    print('Starting Training')

    # TODO : add tqdm with support with notebook
    for epoch in range(starting_epoch, num_epochs):

        epoch_start_time = time.time()  # timer for entire epoch

        train_iterations = len(train_loader)
        train_batch_size = config['batch_size']

        num_batch = 0

        # loss for the epoch
        total_loss = 0.0

        model.train()

        # TODO : add tqdm here as well ? or time remaining ?
        for batch in train_loader:

            images, label = batch

            images = [img.cuda() for img in images]
            label = label.cuda()

            # TODO: Add some visualiser maybe

            output = model(images)

            # Calculate Loss cross Entropy
            loss = criterion(output, label)
            # TODO : Add loss in TensorBoard

            # add loss to epoch loss
            total_loss += loss.item()

            # Do backpropogation
            loss.backward()

            # Change wieghts
            optimizer.step()

            # zero out all grads
            criterion.zero_grad()
            optimizer.zero_grad()

            # Log some info, TODO : add some graphs after some interval
            if num_batch % config['log_freq'] == 0:
                print('{}/{} Epoch : {}/{} Batch Iter : Batch Loss {:.4f}'.format(
                    epoch+1, num_epochs, num_batch+1, len(train_loader), loss.item()
                ))

            num_batch += 1


        # spit train loss details
        average_train_loss = total_loss / len(train_loader)
        print('Average Train Loss at Epoch {} : {:.4f}'.format(epoch+1, average_train_loss))

        # Calc validation results    
        # Print details about end of epoch
        validation_loss, accuracy = _run_eval(model, val_loader, criterion, config)

        print('Average Validation Loss at Epoch {} : {:.4f}'.format(epoch+1, validation_loss))
        print('Validation Accuracy at Epoch {} : {:.4f}'.format(epoch+1, accuracy))

        # TODO : Print details about end of epoch and add it to tensorboard
        # Accuracy, Train Loss, Val Loss, Learning Rate

        if best_accuracy < accuracy :
            best_accuracy = accuracy
            # Save this model
            if export:
                model._save_model(criterion, optimizer, best_accuracy, config, epoch)
        
        # TODO : Change LR depending upon epoch, LR

        total_loss = 0.0
        print('End of epoch {0} / {1} \t Time Taken: {2} sec'.format(epoch+1, num_epochs, time.time() - epoch_start_time))


if __name__ == '__main__':

    print('Training Configuration')
    print(config)

    train(config=config)

    print('Training Ended...')
