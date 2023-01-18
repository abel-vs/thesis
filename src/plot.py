import matplotlib.pyplot as plt
from tqdm import tqdm
import src.metrics as metrics

HEADER_LENGTH = 80

# Method that prints the progress of the training/testing
def print_progress(title, epoch, batch_id, data, data_loader, loss, score=None):
    tqdm.write('{} Epoch: {} [{}/{} ({:.0f}%)]\t Loss: {:.6f} {}'.format(
        title, epoch, batch_id * len(data), len(data_loader.dataset),
        100. * batch_id / len(data_loader), loss.item(),
        '\tScore: ' + str(score) if score is not None else ''))


# Method that prints the average loss, optional metric score and the elapsed time
def print_performance(title, loss, duration, batch_duration, metric=None, score=None):
    title = (" " + title + " PERFORMANCE ").upper()
    half_header = int((HEADER_LENGTH - len(title))/2)
    print("="*half_header + title + "="*half_header)
    print('Average loss = {:.4f}'.format(loss))
    if metric is not None:
        print('{} = {:.4f}'.format(metrics.NAMES[metric], score))
    print('Elapsed time = {:.2f} milliseconds ({:.2f} per batch)'.format(
        duration, batch_duration))
    print("="*HEADER_LENGTH)


# Method that plots the loss
def plot_loss(loss, it, it_per_epoch, smooth_loss=[], base_name='', title=''):
    fig = plt.figure(figsize=(8, 4), dpi=100)
    plt.plot(loss)
    plt.plot(smooth_loss)
    epochs = [i * int(it_per_epoch) for i in range(int(it / it_per_epoch) + 1)]
    plt.plot(epochs, [loss[i] for i in epochs], linestyle='', marker='o')
    # if len(epochs) > 1: print(smooth_loss[epochs[-2]] -  smooth_loss[epochs[-1]] )
    plt.title(title)
    plt.ylabel('Loss')
    plt.xlabel('Iteration')
    # plt.ylim([0, 3])
    if base_name != '':
        fig.savefig(base_name + '.png')
        # pickle.dump([loss, smooth_loss, it], open(base_name + '-' + str(it) + '.p', 'wb'))
        #print(it)
    else:
        plt.show()
    plt.close("all")