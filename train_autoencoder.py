import os
import time
from opts.train_opts import TrainOptions
from data import create_dataset, create_trn_val_tst_dataset
from models import create_model
from utils.logger import get_logger

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval(model, dataloader):
    total_loss = []
    for i, data in enumerate(dataset):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.forward()
        losses = model.get_current_losses()
        loss = losses['MSE']
        total_loss.append(loss)

    return sum(total_loss) / len(total_loss)

def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join('checkpoints', expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch)+'_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))
    
if __name__ == '__main__':
    opt = TrainOptions().parse()   # get training options
    logger_path = os.path.join(opt.log_dir, opt.name) # get logger path
    suffix = '_'.join([opt.model, opt.dataset_mode])    # get logger suffix
    logger = get_logger(logger_path, suffix)            # get logger
    dataset, val_dataset, tst_dataset = create_trn_val_tst_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    total_iters = 0                # the total number of training iterations
    best_eval_loss = float('inf')  # record the best eval loss
    best_eval_epoch = -1           # record the best eval epoch

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            total_iters += 1 # opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch)   # calculate loss functions, get gradients, update network weights
                
            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' + 
                        ' '.join(map(lambda x:'{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                logger.info('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate(logger)                     # update learning rates at the end of every epoch.
        eval_loss = eval(model, val_dataset)
        logger.info('Val result of epoch %d / %d with loss %f' % (epoch, opt.niter + opt.niter_decay, eval_loss))
        if eval_loss < best_eval_loss:
            best_eval_epoch = epoch
            best_eval_loss = eval_loss
    # print best eval result
    logger.info('Best eval epoch %d found with loss %f' % (best_eval_epoch, best_eval_loss))
    # test
    logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
    model.load_networks(best_eval_epoch)
    tst_loss = eval(model, tst_dataset)
    logger.info('Tst result mean avg loss: %.4f ' % tst_loss)
    clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)
