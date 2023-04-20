import numpy as np
import torch, wandb, os, sys
from models.kpconv.architecture import p2p_fitting_regularizer
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR, StepLR, CosineAnnealingLR, ExponentialLR
from .loss import MixLovaszCrossEntropy, CrossEntropy
from sklearn.metrics import confusion_matrix

def fast_hist(pred, label, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + pred[k].astype(int), minlength=n**2).reshape(n, n)


def per_class_iu(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, config):

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu") 

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config
        self.config_model = model.model_config

        #extract info depending on dataset
        self.n_class = self.train_dataloader.dataset.n_label
        self.class_names = self.train_dataloader.dataset.get_class_names()
        self.model = model 
        self.w =  self.train_dataloader.dataset.w
        self.w =  torch.from_numpy(self.w).float().to(self.device)


        self.save_path = config.logger.save_path 
        os.makedirs(self.save_path,exist_ok=True)
        self.model_name = config.logger.model_name

        if config.trainer.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD
        else:
            raise  NameError('Optimizer not supported')
        self.init_optimizer()


        if 'CE' in config.trainer.criterion:
            if 'weighted' in config.trainer.criterion:
                self.criterion_out = CrossEntropy(weight = self.train_dataloader.w, ignore_index=-1)
            else:
                self.criterion_out = CrossEntropy(ignore_index=-1)
        if 'Lovasz' in config.trainer.criterion:
            if 'weighted' in config.trainer.criterion:
                self.criterion_out = MixLovaszCrossEntropy(ignore_index=-1, weight = self.w)
            else:
                self.criterion_out = MixLovaszCrossEntropy(ignore_index=-1)        
        else:
            raise  NameError('Criterion not supported')
        self.init_criterion()

        if config.trainer.scheduler == "CosineAnnealing":
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.trainer.epoch_lr, eta_min=config.trainer.min_lr)
        elif config.trainer.scheduler == "Step":
            self.scheduler = StepLR(self.optimizer, step_size=config.trainer.epoch_lr, gamma=config.trainer.min_lr)
        elif config.trainer.scheduler == "Exponential":
            self.scheduler = ExponentialLR(self.optimizer, gamma=config.trainer.min_lr)
        else:
            raise  NameError('Scheduler not supported')
        self.epochs = config.trainer.epoch 

        self.print_timing = config.trainer.step_size
        self.model.model.to(self.device)
        if self.config.trainer.resume > 0:
            checkpoint = torch.load(os.path.join(self.save_path,self.model_name))
            self.model.model.load_state_dict(checkpoint)
            for _ in range(self.config.trainer.resume):
                self.scheduler.step()

        self.init_wandb()

    def init_optimizer(self):
        # Optimizer with specific learning rate for deformable KPConv
        if self.config.architecture.model == "KPCONV":
            print("KPConv optimizer loaded")
            deform_params = [v for k, v in self.model.model.named_parameters() if 'offset' in k]
            other_params = [v for k, v in self.model.model.named_parameters() if 'offset' not in k]
            deform_lr = self.config.trainer.lr * self.config_model.deform_lr_factor
            self.optimizer = self.optimizer([{'params': other_params},
                                            {'params': deform_params, 'lr': deform_lr}],
                                            lr=self.config.trainer.lr,
                                            momentum=self.config.trainer.momentum,
                                            weight_decay=self.config.trainer.weight_decay)
        else:
            self.optimizer = self.optimizer(self.model.model.parameters(),
                                    lr=self.config.trainer.lr,
                                    momentum=self.config.trainer.momentum,
                                    weight_decay=self.config.trainer.weight_decay)

    def init_criterion(self):
        if self.config.architecture.model == "KPCONV":
            def criterion(outputs, labels, neighbors=None):
                target = - torch.ones_like(labels)
                for i, c in enumerate(self.model.model.valid_labels):
                    target[labels == c] = i
                outputs = torch.transpose(outputs, 0, 1)
                outputs = outputs.unsqueeze(0)
                target = target.unsqueeze(0)
                return self.criterion_out(outputs, target, neighbors) + p2p_fitting_regularizer(self.model.model)
        else:
            def criterion(outputs, labels, neighbors=None):
                return self.criterion_out(outputs, labels, neighbors)
        self.criterion = criterion

    def save_model(self,final = False):
        if final:
            torch.save(self.model.model.state_dict(), os.path.join(self.save_path,'final_'+self.model_name))
        else:
            torch.save(self.model.model.state_dict(), os.path.join(self.save_path,self.model_name))

    def print(self, epoch):
        print("Epoch {}/{} : Current Val Loss {}, Current Val mIoU {}".format(epoch+1, self.epochs, self.ev_loss, np.nanmean(self.cluster_mIoU)))
        print("Best results at epoch {}, with best mIoU {}".format(self.best_epoch+1, np.nanmean(self.best_mIoU)))

    def log(self):
        log_dic = {'learning_rate': self.scheduler.get_last_lr(), 
        'val_loss': self.ev_loss, 
        'cluster_mIoU':np.nanmean(self.cluster_mIoU),
        }
        for i in range(self.n_class):
            log_dic[self.class_names[i]] =  self.cluster_mIoU[i] if not np.isnan(self.cluster_mIoU[i]) else 0
        wandb.log(log_dic)

    def it_log(self,loss):
        log_dic = {'learning_rate': self.scheduler.get_last_lr(), 'train_loss': loss}
        wandb.log(log_dic)

    def init_wandb(self):
        wandb.init(**self.parse_config_for_args())

    def parse_config_for_args(self):
        return {'project':"label_propagation", 'config':self.config_extraction(), 'entity':'caor', 'name':self.config.architecture.model+'_'+self.config.source+'_'+self.config.trainer.criterion.split('_')[0]+'_'+self.config.logger.run_name}

    def config_extraction(self):
        return {
            "batch_size":self.config.trainer.batch_size,
            "num_epochs":self.config.trainer.epoch,
            "learning_rate":self.config.trainer.lr,
            "optimizer":self.config.trainer.optimizer,
            "criterion":self.config.trainer.criterion,
            "cluster_param":str(self.config.cluster.n_centroids)+'_'+str(self.config.cluster.voxel_size),
            "prop_param":str(self.config.sequence.limit_GT)+'_'+str(self.config.sequence.voxel_size)+'_'+str(self.config.sequence.subsample),
            "model_name":self.config.logger.model_name,
        }

    def iteration_train(self):
        self.best_mIoU = [0 for _ in range(self.n_class)] 
        self.best_epoch = 0
        self.step_loss = 0
        self.model.model.train()
        for iter_count in tqdm(range(self.config.trainer.resume,self.epochs)):
            batch = self.train_dataloader.__next__()
            self.optimizer.zero_grad()
            batch.to(self.device)
            outputs = self.model.model(batch, self.config_model)
            loss = self.criterion(outputs, batch.labels, batch.neighbors[0])
            self.step_loss += loss.cpu().detach().numpy()
            loss.backward()
            if self.config_model.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_value_(self.model.model.parameters(), self.config_model.grad_clip_norm)
            self.optimizer.step()
            del loss
            del batch
            del outputs
            torch.cuda.empty_cache()
            self.scheduler.step()
            if (iter_count+1)%self.config.trainer.evaluate_timing == 0 or iter_count == self.epochs-1:
                if self.config.architecture.model == "KPCONV":
                    self.evaluate_kp()
                elif self.config.architecture.model == "SPVCNN":
                    self.evaluate_spv()
                if np.nanmean(self.cluster_mIoU) > np.nanmean(self.best_mIoU):
                    self.best_mIoU = self.cluster_mIoU
                    self.best_epoch = iter_count
                    self.save_model()
                self.log()
                self.print(iter_count)
                self.model.model.train()
            if (iter_count+1)%self.print_timing == 0:
                self.it_log(self.step_loss/(self.print_timing*self.config.trainer.batch_size))
                self.step_loss = 0
        self.save_model(True)


    def evaluate_kp(self):
        self.ev_loss = 0
        self.cluster_mIoU = np.zeros((self.n_class, self.n_class))
        self.model.model.eval()
        dataiter = iter(self.val_dataloader)
        for _ in tqdm(range(len(self.val_dataloader))):
            batch, cluster, r_inds = next(dataiter)
            batch.to(self.device)
            outputs = self.model.model(batch, self.config_model)
            loss = self.criterion(outputs, batch.labels, batch.neighbors[0])
            self.ev_loss += loss.cpu().detach().numpy()
            cluster_outputs = infer_all_pc(outputs, cluster, r_inds, batch.lengths[0].cpu().numpy())
            del batch
            del loss
            del outputs
            self.cluster_mIoU += compute_mIoU(cluster, cluster_outputs, self.n_class)          
            torch.cuda.empty_cache()
        self.cluster_mIoU = per_class_iu(self.cluster_mIoU)*100
        self.ev_loss = self.ev_loss/len(self.val_dataloader)

    def evaluate_spv(self):
        self.ev_loss = 0
        self.cluster_mIoU = np.zeros((self.n_class, self.n_class))
        self.model.eval()
        dataiter = iter(self.val_dataloader)
        for _ in tqdm(range(len(self.val_dataloader))):
            batch, cluster, inputs_invs = next(dataiter)
            batch.to(self.device)
            outputs = self.model.model(batch, self.config_model)
            loss = self.criterion(outputs, batch.labels)
            self.ev_loss += loss.cpu().detach().numpy()
            _outputs = []
            for idx in range(inputs_invs.C[:, -1].max() + 1):
                cur_scene_pts = (batch["lidar"].C[:, -1] == idx).cpu().numpy()
                cur_inv = inputs_invs.F[inputs_invs.C[:, -1] == idx].cpu().numpy()
                outputs_mapped = outputs[cur_scene_pts][cur_inv]
                _outputs.append(outputs_mapped)
            cluster_outputs += [np.argmax(o.detach().cpu().numpy(),axis=1) for o in _outputs]
            del batch
            del loss
            del outputs
            self.cluster_mIoU += compute_mIoU(cluster, cluster_outputs, self.n_class)          
            torch.cuda.empty_cache()
        self.cluster_mIoU = per_class_iu(self.cluster_mIoU)*100
        self.ev_loss = self.ev_loss/len(self.val_dataloader)

def infer_all_pc(outputs, sub_clouds, masks_infer, lengths, proba=False):
    l=0
    if not proba:
        pred = np.argmax(outputs.detach().cpu().numpy(),axis=1).reshape(-1,1)
    else:
        pred = outputs.detach().cpu().numpy()
    preds = []
    for i in range(len(sub_clouds)):
        L = lengths[i]
        local_cloud = pred[l:l+L]
        if not proba:
            preds.append(local_cloud[masks_infer[i]].reshape(-1,1))
        else:
            preds.append(local_cloud[masks_infer[i]])
        l += L
    return preds

def compute_mIoU(true_labels, pred_labels, n_labels):
    conf_mat = np.zeros((n_labels,n_labels))
    true_label_stack = np.concatenate(true_labels)
    pred_label_stack = np.concatenate(pred_labels)
    conf_mat = confusion_matrix(true_label_stack[:,4],pred_label_stack,labels=np.arange(0,n_labels))
    return conf_mat