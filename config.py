import argparse

parser = argparse.ArgumentParser()

# data args
parser.add_argument('--data.dataset_dir', default=r'/home/fetac/data/l_AID', help='l_AID|l_PatternNet|I_RSI-CB128|112_GID|outdoor(the path of the test dataset)')
parser.add_argument('--data.mode', default='train', help='train|val|test')
parser.add_argument('--data.workers', type=int, default=32)
parser.add_argument('--data.imageSize', type=int, default=84)
parser.add_argument('--data.episodeSize', type=int, default=1, help='the mini-batch size of training')
parser.add_argument('--data.episode_train_num', type=int, default=1000, help='the total number of training episodes')
parser.add_argument('--data.way_num', type=int, default=5, help='the number of way/class')
parser.add_argument('--data.shot_num', type=int, default=1, help='the number of shot')
parser.add_argument('--data.query_num', type=int, default=15, help='the number of queries')
parser.add_argument('--data.total_file', type=int, default=4420, help='4420 for AID|16000 for PatternNet|31211 for CB128|31839|7800 for GID for outdoor (the number of the test dataset images)')
parser.add_argument('--data.test_start', type=int, default=13,
                    help='13 for AID|20 for PatternNet|37 for CB128|13 for GID|25 for outdoor (the number of the test dataset classes)')
parser.add_argument('--data.test_end', type=int, default=13, help='the number of the class where the test dataset end')

# test args
parser.add_argument('--test.use_lightFiLM', type=bool, default=True, help='whether use lightFiLM')
parser.add_argument('--test.use_rsa', type=bool, default=True, help='whether use the rsa')
parser.add_argument('--test.rectify', default='lla', help='lla|stf|bb(bb means no rectification operator to use, use what types of rectification)')
parser.add_argument('--test.lla_k', type=int, default=2, help='the k of lla')

#model args
parser.add_argument('--model.name', default='conv4', help='resnet18|conv4(use what types of backbones)')
parser.add_argument('--model.weight_path', default='42conv4.pth.tar', help='42conv4.pth.tar|45conv4.pth.tar|'
                                                                              '42resnet18.pth.tar|45resnet18.pth.tar(the path of pretrained backbone)')
args = vars(parser.parse_args())
