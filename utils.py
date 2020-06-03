import torch
import torch.nn.functional as F

def _get_trainable_params(model):
    """Get Parameters with `requires.grad` set to `True`"""
    trainable_params = []
    for x in model.parameters():
        if x.requires_grad:
            trainable_params.append(x)
    return trainable_params

def _run_eval(model, validation_loader, criterion, config : dict):
    """Runs model over val dataset and returns accuracy and avg val loss"""

    print('Running Validation of Model...')

    model.eval()

    correct_cases = 0
    validation_loss = 0.0

    batch_iter = 0

    y_ground = []
    y_probs = []

    for images, label, _ in validation_loader:

        images = [x.cuda() for x in images]
        label = label.cuda()
        label = label.float()
        label = label.unsqueeze(dim = 0)

        with torch.no_grad():
            output = model(images)

            # Calc loss
            loss = criterion(output, label)

            validation_loss += loss.item()

            output = torch.sigmoid(output)
            y_probs.append(output.item())
            y_ground.append(label.item())

            
            if batch_iter % config['val_log_interval'] == 0:
                print('Val Loss at iter {}/{} : {:.4f}'.format( batch_iter+1, len(validation_loader), loss.item()))
            
        batch_iter += 1

    # Calculate Loss per patient
    average_val_loss = validation_loss / len(validation_loader)

    # Find precision/ recall
    # preds = torch.stack([torch.tensor(y_preds), torch.tensor(y_ground)], dim=1)

    # conf_matrix = torch.zeros(2,2, dtype=torch.int64)

    # for pred in preds:
    #     i,j = pred.tolist()
    #     conf_matrix[i,j] += 1
    
    # conf_matrix = conf_matrix.float()
    # precision = conf_matrix[1, 1].item() / (conf_matrix[1, 0].item() + conf_matrix[1, 1].item() + 1e-12)
    # recall = conf_matrix[1, 1].item() / (conf_matrix[0, 1].item() + conf_matrix[1, 1].item() + 1e-12)
    # f1_score = (2.0 * precision * recall) / (precision + recall + 1e-12)

    # print('Confusion Matrix : ',conf_matrix)

    # return average_val_loss, precision, recall, f1_score, y_preds, y_ground, y_probs
    return average_val_loss, y_probs, y_ground

def _confusion_metrics(y_preds, y_ground):

    # Find precision/ recall
    preds = torch.stack([torch.tensor(y_preds), torch.tensor(y_ground)], dim=1)

    conf_matrix = torch.zeros(2,2, dtype=torch.int64)

    for pred in preds:
        i,j = pred.tolist()
        conf_matrix[i,j] += 1
    
    conf_matrix = conf_matrix.float()
    precision = conf_matrix[1, 1].item() / (conf_matrix[1, 0].item() + conf_matrix[1, 1].item() + 1e-12)
    recall = conf_matrix[1, 1].item() / (conf_matrix[0, 1].item() + conf_matrix[1, 1].item() + 1e-12)
    f1_score = (2.0 * precision * recall) / (precision + recall + 1e-12)

    print('Confusion Matrix : ',conf_matrix)

    return precision, recall, f1_score
