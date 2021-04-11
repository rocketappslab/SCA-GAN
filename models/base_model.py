import os
import torch
import torch.nn as nn
import torchvision.models.vgg as models
import pickle
import util.util
class BaseModel(nn.Module):

    def __init__(self):
        super(BaseModel, self).__init__()

    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.vgg_path = os.path.join('fashion_data/vgg19-dcbb9e9d.pth')
        self.edgeprior_path = os.path.join('checkpoints/scagan_pctnet/latest_netG.pth')

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids, epoch, total_steps):
        save_filename = '%s_%s.pth' % (epoch_label, network_label)
        save_infoname = '%s.pkl' % (epoch_label)
        save_path = os.path.join(self.save_dir, save_filename)
        save_infoname = os.path.join(self.save_dir, save_infoname)
        torch.save(network.cpu().state_dict(), save_path)
        network.cuda()

        info = {'epoch': epoch, 'total_steps': total_steps}
        filehandler = open(save_infoname, "wb")
        pickle.dump(info, filehandler)
        filehandler.close()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            # network.load_state_dict(torch.load(save_path))
            util.util.multi_gpus_load_dict(network, save_path)
            print("Found checkpoints. Network loaded.")
        else:
            print("Not found checkpoints. Network from scratch.")

    def load_VGG(self, network):
        # pretrained_dict = torch.load(self.vgg_path)

        # pretrained_model = models.vgg19(pretrained=True).features
        vgg19 = models.vgg19(pretrained=False)
        vgg19.load_state_dict(torch.load(self.vgg_path))
        pretrained_model = vgg19.features

        pretrained_dict = pretrained_model.state_dict()

        model_dict = network.state_dict()

        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # load the new state dict
        network.load_state_dict(model_dict)

    def load_EdgePriorNetwork(self, network):
        from copy import deepcopy
        state_dict = network.state_dict().copy()
        state_dict_old = torch.load(self.edgeprior_path)
        # state_dict_old = torch.load(saved_state_dict)['state_dict']
        for key, nkey in zip(state_dict_old.keys(), state_dict.keys()):
            if key != nkey:
                # remove the 'module.' in the 'key'
                state_dict[key[7:]] = deepcopy(state_dict_old[key])
            else:
                state_dict[key] = deepcopy(state_dict_old[key])

        network.load_state_dict(state_dict)

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
