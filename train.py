from dataset import MRData
from models import MRnet
from config import config
import torch
from torch.utils.tensorboard import SummaryWriter
from utils import _train_model, _evaluate_model, _get_lr
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

    train_loader, val_loader, train_wts, val_wts = load_data(config['task'])

    print('Initializing Model...')
    model = MRnet()
    if torch.cuda.is_available():
        model = model.cuda()

    print('Initializing Loss Method...')
    criterion = torch.nn.BCEWithLogitsLoss(weight=train_wts)

    print('Setup the Optimizer')
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=.3, threshold=1e-4, verbose=True)
    
    starting_epoch = config['starting_epoch']
    num_epochs = config['max_epoch']
    patience = config['patience']
    log_train = config['log_train']
    log_val = config['log_val']
    iteration_change_loss = 0

    best_val_loss = float('inf')
    best_val_auc = float(0)

    print('Starting Training')

    writer = SummaryWriter(comment=f'lr={config['lr']} task={config['task']}')

    for epoch in range(starting_epoch, num_epochs):

        epoch_start_time = time.time()  # timer for entire epoch

        train_loss, train_auc = _train_model(
            model, train_loader, epoch, num_epochs, optimizer, writer, current_lr, log_train)

        val_loss, val_auc = _evaluate_model(
            model, val_loader, epoch, num_epochs, writer, current_lr, log_val)

        scheduler.step(val_loss)

        t_end = time.time()
        delta = t_end - epoch_start_time

        print("train loss : {0} | train auc {1} | val loss {2} | val auc {3} | elapsed time {4} s".format(
            train_loss, train_auc, val_loss, val_auc, delta))

        iteration_change_loss += 1
        print('-' * 30)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            if bool(config['save_model']):
                file_name = f'model_{config['exp_name']}_{config['task']}_val_auc_{val_auc:0.4f}_train_auc_{train_auc:0.4f}_epoch_{epoch+1}.pth'
                for f in os.listdir('./weights/'):
                    if (config['task'] in f) and (config['exp_name'] in f):
                        os.remove(f'./weights/{f}')
                torch.save({
                    'model_state_dict': model.state_dict()
                }, f'./weights/{file_name}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            print('Early stopping after {0} iterations without the decrease of the val loss'.
                  format(iteration_change_loss))
            break

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')

if __name__ == '__main__':

    print('Training Configuration')
    print(config)

    train(config=config)

    print('Training Ended...')
