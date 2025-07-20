import os
import torch
import numpy as np
from tqdm import tqdm
from config import args
from torchvision import transforms
from dataset import Imagefolder_normal
import random
from models.utils import get_backbone_and_load_dict
from models.losses import prototype_loss
from models.tsa import reset_architecture, tsa
"""
define path
"""
PROJECT_ROOT = '\\'.join(os.path.realpath(__file__).split('\\')[:-1])
modelpath = os.path.join(PROJECT_ROOT, 'models')


def setup_seed(seed):
    """
    set up the random seeds, to get the same result every time
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
def get_score(acc_list):
    """
    analyse the mean and interval
    """
    mean = np.mean(acc_list)
    interval = 1.96 * np.sqrt(np.var(acc_list)/len(acc_list))
    return mean, interval


def stf(x, beta):
    """
    use simple transform to rectify the channel values, see details in article "https://arxiv.org/abs/2206.08126"
    """
    zero_tensor = torch.zeros_like(x)
    x_pos = torch.maximum(x, zero_tensor)
    x_neg = torch.minimum(x, zero_tensor)
    x_pos = 1 / torch.pow(torch.log(1 / (x_pos + 1e-5) + 1), beta)
    x_neg = -1 / torch.pow(torch.log(1 / (-x_neg + 1e-5) + 1), beta)
    return x_pos + x_neg

def lla(x, k):
    """
    use lla(lifted Laplacian activation) to rectify the channel values, see details in article " "
    """
    abs_x = torch.abs(x)
    y = x * (1 + k * torch.exp(-abs_x))
    return y


def train():
    setup_seed(3407)
    if os.path.exists('target_loss.txt'):
        os.remove('target_loss.txt')

    # define transforms
    data_transforms = {
        'test': transforms.Compose([
            transforms.RandomResizedCrop(84),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # define loader
    testset_1 = Imagefolder_normal(
        data_dir=args['data.dataset_dir'], mode=args['data.mode'], image_size=args['data.imageSize'], transform=data_transforms['test'],
        episode_num=args['data.episode_train_num'], way_num=args['data.way_num'], shot_num=args['data.shot_num'], query_num=args['data.query_num'],
        total_file=args['data.total_file'], test_start=args['data.test_start'], test_end=args['data.test_end']
    )
    test_loader_1 = torch.utils.data.DataLoader(
        testset_1, batch_size=args['data.episodeSize'], shuffle=True,
        num_workers=int(args['data.workers']), drop_last=True, pin_memory=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # define some hyper-parameters
    if args['data.shot_num'] == 5 or args['data.shot_num'] == 4 or args['data.shot_num'] == 3:
        lr_o, lr_beta_o = 0.05, 0.1
    elif args['data.shot_num'] == 1 or args['data.shot_num'] == 2:
        lr_o, lr_beta_o = 0.5, 1

    # get model and load dict
    model = get_backbone_and_load_dict(args['model.name'], args['model.weight_path'], modelpath, device)

    # reset the architecture to add the rectification components
    if args['test.use_rsa']:
        model = reset_architecture(model, args['model.name'], args['test.use_lightFiLM'])
        model.reset()

    model.cuda()

    test_loader_11 = iter(test_loader_1)

    val_acc, val_loss = [], []
    val_accs, val_losses = [], []

    model.eval()
    for j in tqdm(range(2000)):
        if args['test.use_rsa']:
            # initialize task-specific adapters and pre-classifier alignment for each task
            model.reset()
        # load images and manage them
        try:
            (query_images, query_targets, support_images, support_targets) = next(test_loader_11)
        except StopIteration:
            del test_loader_11
            test_loader_11 = iter(test_loader_1)
            (query_images, query_targets, support_images, support_targets) = next(test_loader_11)
        query_images = torch.cat(query_images, 0).cuda()
        input_var2 = []
        for support_image in range(len(support_images)):
            temp_support = support_images[support_image]
            temp_support = torch.cat(temp_support, 0)
            temp_support = temp_support.cuda()
            input_var2.append(temp_support)
        input_var2 = torch.stack(input_var2)
        support_images = input_var2
        support_images = torch.reshape(support_images, (-1, 3, 84, 84))
        query_targets = torch.cat(query_targets, 0)
        support_targets = torch.cat(support_targets, 0)
        context_labels = support_targets.cuda()
        target_labels = query_targets.cuda()
        # optimize task-specific adapters and/or pre-classifier alignment
        if args['test.use_rsa']:
            tsa(support_images, context_labels, model, max_iter=20, lr=lr_o, lr_beta=lr_beta_o, rectify=args['test.rectify'], k=args['test.lla_k'])
        with torch.no_grad():
            context_features = model.embed(support_images)
            target_features = model.embed(query_images)
            if args['test.use_rsa']:
                context_features = model.beta(context_features)
                target_features = model.beta(target_features)
            # rectify channel values by using stf/lla
            if args['test.rectify'] == 'stf':
                context_features = stf(context_features, 1.3)
                target_features = stf(target_features, 1.3)
            elif args['test.rectify'] == 'lla':
                context_features = lla(context_features, args['test.lla_k'])
                target_features = lla(target_features, args['test.lla_k'])
            else:
                pass
        # use prototype to classify the features
        loss, stats_dict, _ = prototype_loss(context_features, context_labels,
                                             target_features, target_labels)
        # record the results
        ep_loss = stats_dict['loss']
        ep_acc = stats_dict['acc']
        val_loss.append(ep_loss)
        val_acc.append(ep_acc)
        # report the results every 50 tasks
        if (j + 1) % 50 == 0:
            val_acc, val_loss = np.mean(val_acc) * 100, np.mean(val_loss)
            val_accs.append(val_acc)
            val_losses.append(val_loss)
            temp_test_write = f'''val_acc {val_acc:.2f}%, val_loss {val_loss:.3f},episode {j + 1};'''
            print(temp_test_write)
            with open('target_loss.txt', 'a+') as f:
                f.write('========================>' + '\n')
                f.write(temp_test_write + '\n')
                f.close
            val_acc, val_loss = [], []
    # report the average result
    avg_val_loss = np.mean(val_losses)
    mean, interval = get_score(val_accs)
    temp_test_write = f'''avg_val_acc {mean:.2f}%,interval {interval:.2f} ,avg_val_loss_1 {avg_val_loss:.3f},episode {j + 1};'''
    print(temp_test_write)
    with open('target_loss.txt', 'a+') as f:
        f.write('========================>' + '\n')
        f.write(temp_test_write + '\n')
        f.close

if __name__ == '__main__':
    train()
