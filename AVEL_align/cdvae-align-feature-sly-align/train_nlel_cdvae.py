import os
import time
import logging
from pathlib import Path
from importlib import import_module

import numpy as np

import torch
from torch.utils.data import DataLoader

from model.radam import RAdam 

from util.dataset_loader import FeatDataset, WavDataset

class Trainer(object):
    def __init__(self, train_config, model_config):
        learning_rate  = train_config.get('learning_rate', 1e-4)
        model_type     = train_config.get('model_type', 'vae')
        self.opt_param = train_config.get('optimize_param', {
                                'optim_type': 'RAdam',
                                'learning_rate': 1e-4,
                                'max_grad_norm': 10,
                                'lr_scheduler':{
                                    'step_size': 100000,
                                    'gamma': 0.5,
                                    'last_epoch': -1
                                }
                            })        

        module = import_module('model.{}'.format(model_type), package=None)
        MODEL = getattr(module, 'Model')
        # model = MODEL(model_config).cuda()
        model = MODEL(model_config)

        self.model = model.cuda()
        self.learning_rate = learning_rate

        if self.opt_param['optim_type'].upper() == 'RADAM':
            self.optimizer = RAdam( self.model.parameters(), 
                                    lr=self.opt_param['learning_rate'],
                                    betas=(0.5,0.999),
                                    weight_decay=0.0)
        else:
            self.optimizer = torch.optim.Adam( self.model.parameters(),
                                               lr=self.opt_param['learning_rate'],
                                               betas=(0.5,0.999),
                                               weight_decay=0.0)

        if 'lr_scheduler' in self.opt_param.keys():
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                                optimizer=self.optimizer,
                                **self.opt_param['lr_scheduler']
                            )
        else:
            self.scheduler = None


        print(f'lr is: {self.opt_param["learning_rate"]}')
        self.iteration = 0
        self.model.train()

    def step(self, input, iteration=None):
        assert self.model.training
        self.model.zero_grad()

        input = (x.cuda() for x in input)
        loss, loss_detail = self.model(input)

        loss.backward()
        if self.opt_param['max_grad_norm'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.opt_param['max_grad_norm'])
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        if iteration is None:
            self.iteration += 1
        else:
            self.iteration = iteration

        return loss_detail


    def save_checkpoint(self, checkpoint_path):
        torch.save( {
                'model': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'iteration': self.iteration,
            }, checkpoint_path)
        print("Saved state dict. to {}".format(checkpoint_path))


    def load_checkpoint(self, checkpoint_path):
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint_data['model'])
        # self.optimizer.load_state_dict(checkpoint_data['optimizer'])
        return checkpoint_data['iteration']

def train(train_config): 
    # Initial
    output_directory     = train_config.get('output_directory', '')
    max_iter             = train_config.get('max_iter', 100000)
    batch_size           = train_config.get('batch_size', 16)
    crop_length          = train_config.get('crop_length', 128)
    iters_per_checkpoint = train_config.get('iters_per_checkpoint', 10000)
    iters_per_log        = train_config.get('iters_per_log', 1000)
    seed                 = train_config.get('seed', 1234)
    checkpoint_path      = train_config.get('checkpoint_path', '')
    trainer_type         = train_config.get('trainer_type', 'basic')

    # Setup
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Initial trainer
    if trainer_type == 'basic':
        trainer = Trainer( train_config, model_config)
        collate_fn = None
    else:
        module = import_module('model.{}'.format(trainer_type), package=None)
        TRAINER = getattr(module, 'Trainer')
        trainer = TRAINER(train_config, model_config)
        try:
            collate_fn = getattr(module, 'collate')
        except:
            collate_fn = None

    # Load checkpoint if the path is given 
    iteration = 1
    if checkpoint_path != "":
        iteration = trainer.load_checkpoint( checkpoint_path)
        iteration += 1  # next iteration is iteration + 1

    # Load training data
    if os.path.exists(os.path.join(data_config['training_dir'],'feats.scp')):
        trainset = FeatDataset( data_config['training_dir'], 
                                crop_length,
                                data_config)
        print("exist")  
            
    else:
        trainset = WavDataset(  data_config['training_dir'], 
                                crop_length, 
                                data_config)
        print("not exist")  
        


    train_loader = DataLoader(trainset, num_workers=8, shuffle=True,
                              batch_size=batch_size,
                              pin_memory=True,
                              drop_last=True,
                              collate_fn=collate_fn)

    # Get shared output_directory ready
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    
    # Prepare logger
    logger = logging.getLogger("logger")
    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(filename=str(output_directory/'Stat'))
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s %(message)s",
                                  datefmt="%m-%d %H:%M:%S")
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger.addHandler(handler1)
    logger.addHandler(handler2)

    logger.info("Output directory: {}".format(output_directory))
    logger.info("Training utterances: {}".format(len(trainset)))

    # ================ MAIN TRAINNIG LOOP! ===================
    
    logger.info("Start traininig...")

    loss_log = dict()

    # ================ reset iteration for nlel cdvae ! ===================

    iteration = 1
    print("iteration")
    print(iteration)
    print("max_iter")
    print(max_iter)
    while iteration <= max_iter:
        for i, batch in enumerate(train_loader):
            print("batch , batch.shape , batch_type , a_type")
            print(batch)
            print(len(a) for a in batch)
            breakpoint()
            loss_detail = trainer.step(batch, iteration=iteration)

            # Keep Loss detail
            for key,val in loss_detail.items():
                if key not in loss_log.keys():
                    loss_log[key] = list()
                loss_log[key].append(val)
            
            # Save model per N iterations
            if iteration % iters_per_checkpoint == 0:
                checkpoint_path =  output_directory / "{}_{}".format(time.strftime("%m-%d_%H-%M", time.localtime()),iteration)
                trainer.save_checkpoint( checkpoint_path)

            # Show log per M iterations
            if iteration % iters_per_log == 0:
                mseg = 'Iter {}:'.format( iteration)
                for key,val in loss_log.items():
                    mseg += '  {}: {:.6f}'.format(key,np.mean(val))
                logger.info(mseg)
                loss_log = dict()

            if iteration > max_iter:
                break

            iteration += 1

    else:
        print('Finished')
        

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='conf/config_vae_vc.json',
                        help='JSON file for configuration')
    parser.add_argument('-o', '--output_directory', type=str, default=None,
                        help='Directory for checkpoint output')
    parser.add_argument('-p', '--checkpoint_path', type=str, default=None,
                        help='checkpoint path to keep training')
    parser.add_argument('-T', '--training_dir', type=str, default=None,
                        help='Traininig dictionary path')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help='Using gpu #')
    parser.add_argument('-lr', '--learning_rate', type=float, default=None,
                        help='Using learning rate')
    args = parser.parse_args()

    # Parse configs.  Globals nicer in this case
    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global model_config
    model_config = config["model_config"]

    if args.output_directory is not None:
        train_config['output_directory'] = args.output_directory
    if args.checkpoint_path is not None:
        train_config['checkpoint_path'] = args.checkpoint_path
    if args.training_dir is not None:
        data_config['training_dir'] = args.training_dir
    if args.learning_rate is not None:
        train_config['optimize_param']['learning_rate'] = args.learning_rate

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False

    train(train_config)
