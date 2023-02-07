import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from models import GRNVAE
from evaluate import top_k_filter, get_epr
from tqdm import tqdm
from logger import LightLogger

def runGRNVAE(exp_array, configs, 
              ground_truth=None, logger=None):
    '''
    Initialize and Train a GRNVAE model with configs
    
    Parameters
    ----------
    exp_array: np.array
        Expression data with cells on rows and genes on columns. 
    configs: dict
        A dictionary defining various hyperparameters of the 
        model. See Hyperparameters include `train_split`, 
        `train_split_seed`, `batch_size`, `hidden_dim`, `z_dim`,
        `train_on_non_zero`, `dropout_augmentation`, `cuda`,
        `alpha`, `beta`, `delayed_steps_on_sparse`, `n_epochs`, 
        `eval_on_n_steps`, `lr_nn`, `lr_adj`, `K1`, and `K2`. 
    ground_truth: tuple or None
        (Optional, only for BEELINE evaluation) You don't need 
        to define this parameter when you execute GRNVAE on real 
        datasets when the ground truth network is unknown. For 
        evaluations on BEELINE, 
        BEELINE ground truth object exported by 
        data.load_beeline_ground_truth. It's a tuple with the 
        first element being truth_edges and second element being
        evaluate_mask.
    benchmark_setting: str
        Benchmark settings. Choose among `500_STRING`, 
        `1000_STRING`, `500_Non-ChIP`, `1000_Non-ChIP`, 
        `500_ChIP-seq`, `1000_ChIP-seq`, `500_lofgof`,
        and `1000_lofgof`. If either of the `lofgof` settings
        is choosed, only `mESC` data is available. 
        
    Returns
    -------
    tuple
        First element is a scanpy data with cells on rows and 
        genes on columns. Second element is the corresponding 
        BEELINE ground truth data 
    '''
    n_obs, n_gene = exp_array.shape
    
    # Logger -------------------------------------------------------------------
    if logger is None:
        logger = LightLogger()
    logger.set_configs(configs)
    note_id = logger.start()

    # Global Mean/Std ----------------------------------------------------------
    global_mean = torch.FloatTensor(exp_array.mean(0))
    global_std = torch.FloatTensor(exp_array.std(0))

    # Train/Test split if requested --------------------------------------------
    assert configs['train_split']>0 and configs['train_split']<=1, \
        f'Expect 0<configs["train_split"]<=1'
    has_train_test_split = (configs['train_split'] != 1.0)
    
    if configs['train_split_seed'] is None:
        train_mask = np.random.rand(n_obs)
    else:
        rs = np.random.RandomState(seed=configs['train_split_seed'])
        train_mask = rs.rand(n_obs)
        
    train_dt = TensorDataset(
        torch.FloatTensor(exp_array[train_mask <= configs['train_split'], ])
    )
    train_loader = DataLoader(
        train_dt, batch_size=configs['batch_size'], shuffle=True)
    if has_train_test_split:
        val_dt = TensorDataset(
            torch.FloatTensor(exp_array[train_mask > configs['train_split'], ])
        )
        val_loader = DataLoader(
            val_dt, batch_size=configs['batch_size'], shuffle=True)

    # Defining Model -----------------------------------------------------------
    vae = GRNVAE(
        n_gene = n_gene, 
        hidden_dim=configs['hidden_dim'], z_dim=configs['z_dim'], 
        train_on_non_zero=configs['train_on_non_zero'], 
        dropout_augmentation=configs['dropout_augmentation']
    )
    
    all_but_adj = [p for i, p in enumerate(vae.parameters()) if i != 0]
    opt_nn = torch.optim.RMSprop(all_but_adj, lr=configs['lr_nn'])
    opt_adj = torch.optim.RMSprop([vae.adj_A], lr=configs['lr_adj'])

    # Move things to cuda if necessary -----------------------------------------
    if configs['cuda']:
        global_mean = global_mean.cuda()
        global_std = global_std.cuda()
        vae = vae.cuda()
        
    # Training loops -----------------------------------------------------------
    for epoch in tqdm(range(configs['n_epochs'])):
        vae.train(True)
        iteration_for_A = epoch%(configs['K1']+configs['K2'])>=configs['K1']
        vae.adj_A.requires_grad = iteration_for_A
        evaluation_turn = (epoch % configs['eval_on_n_steps'] == 0)
        
        # go through training samples 
        eval_log = {
            'train_loss_rec': 0, 'train_loss_kl': 0, 'train_loss_sparse': 0
        }
        for batch in train_loader:
            x = batch[0]
            if configs['cuda']:
                x = x.cuda()
                
            if iteration_for_A:
                opt_adj.zero_grad()
                out = vae(x, global_mean, global_std, 
                          use_dropout_augmentation=False)
            else:
                opt_nn.zero_grad()
                out = vae(x, global_mean, global_std, 
                          use_dropout_augmentation=True)
            loss = out['loss_rec'] + configs['beta'] * out['loss_kl'] 
            if epoch >= configs['delayed_steps_on_sparse']:
                loss += configs['alpha'] * out['loss_sparse']
            loss.backward()
            if iteration_for_A:
                opt_adj.step()
            else:
                opt_nn.step()
            if evaluation_turn:
                eval_log['train_loss_rec'] += out['loss_rec'].detach().cpu().item()
                eval_log['train_loss_kl'] += out['loss_kl'].detach().cpu().item()
                eval_log['train_loss_sparse'] += out['loss_sparse'].detach().cpu().item()
        
        # go through val samples
        if evaluation_turn:
            if ground_truth is not None:
                epoch_epr = get_epr(vae.get_adj(), ground_truth)
                eval_log['ep'] = epoch_epr[0]
                eval_log['epr'] = epoch_epr[1]
            
            if has_train_test_split:
                eval_log['val_loss_rec'] = 0
                eval_log['val_loss_kl'] = 0
                eval_log['val_loss_sparse'] = 0
                vae.train(False)
                for batch in val_loader:
                    x = batch[0]
                    if configs['cuda']:
                        x = x.cuda()
                    out = vae(x, global_mean, global_std, 
                              use_dropout_augmentation=False)
                    eval_log['val_loss_rec'] += out['loss_rec'].detach().cpu().item()
                    eval_log['val_loss_kl'] += out['loss_kl'].detach().cpu().item()
                    eval_log['val_loss_sparse'] += out['loss_sparse'].detach().cpu().item()

            logger.log(eval_log)
    logger.finish()
    return vae.cpu()