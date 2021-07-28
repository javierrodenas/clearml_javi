"""
ImageNet Validation Script
Adapted from https://github.com/rwightman/pytorch-image-models
The script is further extend to evaluate VOLO
"""
import argparse
import os
import csv
import glob
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_legacy
import models
import numpy as np
from tqdm import tqdm
has_apex = False
import pandas as pd
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass


has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

class_to_oht_index = {'0': 0, '1': 1, '10': 2, '100': 3, '101': 4, '102': 5, '103': 6, '104': 7, '105': 8, '106': 9, '107': 10, '108': 11, '109': 12, '11': 13, '110': 14, '111': 15, '112': 16, '113': 17, '114': 18, '115': 19, '116': 20, '117': 21, '118': 22, '119': 23, '12': 24, '120': 25, '121': 26, '122': 27, '123': 28, '124': 29, '125': 30, '126': 31, '127': 32, '128': 33, '129': 34, '13': 35, '130': 36, '131': 37, '132': 38, '133': 39, '134': 40, '135': 41, '136': 42, '137': 43, '138': 44, '139': 45, '14': 46, '140': 47, '141': 48, '142': 49, '143': 50, '144': 51, '145': 52, '146': 53, '147': 54, '148': 55, '149': 56, '15': 57, '150': 58, '151': 59, '152': 60, '153': 61, '154': 62, '155': 63, '156': 64, '157': 65, '158': 66, '159': 67, '16': 68, '160': 69, '161': 70, '162': 71, '163': 72, '164': 73, '165': 74, '166': 75, '167': 76, '168': 77, '169': 78, '17': 79, '170': 80, '171': 81, '172': 82, '173': 83, '174': 84, '175': 85, '176': 86, '177': 87, '178': 88, '179': 89, '18': 90, '180': 91, '181': 92, '182': 93, '183': 94, '184': 95, '185': 96, '186': 97, '187': 98, '188': 99, '189': 100, '19': 101, '190': 102, '191': 103, '192': 104, '193': 105, '194': 106, '195': 107, '196': 108, '197': 109, '198': 110, '199': 111, '2': 112, '20': 113, '200': 114, '201': 115, '202': 116, '203': 117, '204': 118, '205': 119, '206': 120, '207': 121, '208': 122, '209': 123, '21': 124, '210': 125, '211': 126, '212': 127, '213': 128, '214': 129, '215': 130, '216': 131, '217': 132, '218': 133, '219': 134, '22': 135, '220': 136, '221': 137, '222': 138, '223': 139, '224': 140, '225': 141, '226': 142, '227': 143, '228': 144, '229': 145, '23': 146, '230': 147, '231': 148, '232': 149, '233': 150, '234': 151, '235': 152, '236': 153, '237': 154, '238': 155, '239': 156, '24': 157, '240': 158, '241': 159, '242': 160, '243': 161, '244': 162, '245': 163, '246': 164, '247': 165, '248': 166, '249': 167, '25': 168, '250': 169, '251': 170, '252': 171, '253': 172, '254': 173, '255': 174, '256': 175, '257': 176, '258': 177, '259': 178, '26': 179, '260': 180, '261': 181, '262': 182, '263': 183, '264': 184, '265': 185, '266': 186, '267': 187, '268': 188, '269': 189, '27': 190, '270': 191, '271': 192, '272': 193, '273': 194, '274': 195, '275': 196, '276': 197, '277': 198, '278': 199, '279': 200, '28': 201, '280': 202, '281': 203, '282': 204, '283': 205, '284': 206, '285': 207, '286': 208, '287': 209, '288': 210, '289': 211, '29': 212, '290': 213, '291': 214, '292': 215, '293': 216, '294': 217, '295': 218, '296': 219, '297': 220, '298': 221, '299': 222, '3': 223, '30': 224, '300': 225, '301': 226, '302': 227, '303': 228, '304': 229, '305': 230, '306': 231, '307': 232, '308': 233, '309': 234, '31': 235, '310': 236, '311': 237, '312': 238, '313': 239, '314': 240, '315': 241, '316': 242, '317': 243, '318': 244, '319': 245, '32': 246, '320': 247, '321': 248, '322': 249, '323': 250, '324': 251, '325': 252, '326': 253, '327': 254, '328': 255, '329': 256, '33': 257, '330': 258, '331': 259, '332': 260, '333': 261, '334': 262, '335': 263, '336': 264, '337': 265, '338': 266, '339': 267, '34': 268, '340': 269, '341': 270, '342': 271, '343': 272, '344': 273, '345': 274, '346': 275, '347': 276, '348': 277, '349': 278, '35': 279, '350': 280, '351': 281, '352': 282, '353': 283, '354': 284, '355': 285, '356': 286, '357': 287, '358': 288, '359': 289, '36': 290, '360': 291, '361': 292, '362': 293, '363': 294, '364': 295, '365': 296, '366': 297, '367': 298, '368': 299, '369': 300, '37': 301, '370': 302, '371': 303, '372': 304, '373': 305, '374': 306, '375': 307, '376': 308, '377': 309, '378': 310, '379': 311, '38': 312, '380': 313, '381': 314, '382': 315, '383': 316, '384': 317, '385': 318, '386': 319, '387': 320, '388': 321, '389': 322, '39': 323, '390': 324, '391': 325, '392': 326, '393': 327, '394': 328, '395': 329, '396': 330, '397': 331, '398': 332, '399': 333, '4': 334, '40': 335, '400': 336, '401': 337, '402': 338, '403': 339, '404': 340, '405': 341, '406': 342, '407': 343, '408': 344, '409': 345, '41': 346, '410': 347, '411': 348, '412': 349, '413': 350, '414': 351, '415': 352, '416': 353, '417': 354, '418': 355, '419': 356, '42': 357, '420': 358, '421': 359, '422': 360, '423': 361, '424': 362, '425': 363, '426': 364, '427': 365, '428': 366, '429': 367, '43': 368, '430': 369, '431': 370, '432': 371, '433': 372, '434': 373, '435': 374, '436': 375, '437': 376, '438': 377, '439': 378, '44': 379, '440': 380, '441': 381, '442': 382, '443': 383, '444': 384, '445': 385, '446': 386, '447': 387, '448': 388, '449': 389, '45': 390, '450': 391, '451': 392, '452': 393, '453': 394, '454': 395, '455': 396, '456': 397, '457': 398, '458': 399, '459': 400, '46': 401, '460': 402, '461': 403, '462': 404, '463': 405, '464': 406, '465': 407, '466': 408, '467': 409, '468': 410, '469': 411, '47': 412, '470': 413, '471': 414, '472': 415, '473': 416, '474': 417, '475': 418, '476': 419, '477': 420, '478': 421, '479': 422, '48': 423, '480': 424, '481': 425, '482': 426, '483': 427, '484': 428, '485': 429, '486': 430, '487': 431, '488': 432, '489': 433, '49': 434, '490': 435, '491': 436, '492': 437, '493': 438, '494': 439, '495': 440, '496': 441, '497': 442, '498': 443, '499': 444, '5': 445, '50': 446, '500': 447, '501': 448, '502': 449, '503': 450, '504': 451, '505': 452, '506': 453, '507': 454, '508': 455, '509': 456, '51': 457, '510': 458, '511': 459, '512': 460, '513': 461, '514': 462, '515': 463, '516': 464, '517': 465, '518': 466, '519': 467, '52': 468, '520': 469, '521': 470, '522': 471, '523': 472, '524': 473, '525': 474, '526': 475, '527': 476, '528': 477, '529': 478, '53': 479, '530': 480, '531': 481, '532': 482, '533': 483, '534': 484, '535': 485, '536': 486, '537': 487, '538': 488, '539': 489, '54': 490, '540': 491, '541': 492, '542': 493, '543': 494, '544': 495, '545': 496, '546': 497, '547': 498, '548': 499, '549': 500, '55': 501, '550': 502, '551': 503, '552': 504, '553': 505, '554': 506, '555': 507, '556': 508, '557': 509, '558': 510, '559': 511, '56': 512, '560': 513, '561': 514, '562': 515, '563': 516, '564': 517, '565': 518, '566': 519, '567': 520, '568': 521, '569': 522, '57': 523, '570': 524, '571': 525, '572': 526, '573': 527, '574': 528, '575': 529, '576': 530, '577': 531, '578': 532, '579': 533, '58': 534, '580': 535, '581': 536, '582': 537, '583': 538, '584': 539, '585': 540, '586': 541, '587': 542, '588': 543, '589': 544, '59': 545, '590': 546, '591': 547, '592': 548, '593': 549, '594': 550, '595': 551, '596': 552, '597': 553, '598': 554, '599': 555, '6': 556, '60': 557, '600': 558, '601': 559, '602': 560, '603': 561, '604': 562, '605': 563, '606': 564, '607': 565, '608': 566, '609': 567, '61': 568, '610': 569, '611': 570, '612': 571, '613': 572, '614': 573, '615': 574, '616': 575, '617': 576, '618': 577, '619': 578, '62': 579, '620': 580, '621': 581, '622': 582, '623': 583, '624': 584, '625': 585, '626': 586, '627': 587, '628': 588, '629': 589, '63': 590, '630': 591, '631': 592, '632': 593, '633': 594, '634': 595, '635': 596, '636': 597, '637': 598, '638': 599, '639': 600, '64': 601, '640': 602, '641': 603, '642': 604, '643': 605, '644': 606, '645': 607, '646': 608, '647': 609, '648': 610, '649': 611, '65': 612, '650': 613, '651': 614, '652': 615, '653': 616, '654': 617, '655': 618, '656': 619, '657': 620, '658': 621, '659': 622, '66': 623, '660': 624, '661': 625, '662': 626, '663': 627, '664': 628, '665': 629, '666': 630, '667': 631, '668': 632, '669': 633, '67': 634, '670': 635, '671': 636, '672': 637, '673': 638, '674': 639, '675': 640, '676': 641, '677': 642, '678': 643, '679': 644, '68': 645, '680': 646, '681': 647, '682': 648, '683': 649, '684': 650, '685': 651, '686': 652, '687': 653, '688': 654, '689': 655, '69': 656, '690': 657, '691': 658, '692': 659, '693': 660, '694': 661, '695': 662, '696': 663, '697': 664, '698': 665, '699': 666, '7': 667, '70': 668, '700': 669, '701': 670, '702': 671, '703': 672, '704': 673, '705': 674, '706': 675, '707': 676, '708': 677, '709': 678, '71': 679, '710': 680, '711': 681, '712': 682, '713': 683, '714': 684, '715': 685, '716': 686, '717': 687, '718': 688, '719': 689, '72': 690, '720': 691, '721': 692, '722': 693, '723': 694, '724': 695, '725': 696, '726': 697, '727': 698, '728': 699, '729': 700, '73': 701, '730': 702, '731': 703, '732': 704, '733': 705, '734': 706, '735': 707, '736': 708, '737': 709, '738': 710, '739': 711, '74': 712, '740': 713, '741': 714, '742': 715, '743': 716, '744': 717, '745': 718, '746': 719, '747': 720, '748': 721, '749': 722, '75': 723, '750': 724, '751': 725, '752': 726, '753': 727, '754': 728, '755': 729, '756': 730, '757': 731, '758': 732, '759': 733, '76': 734, '760': 735, '761': 736, '762': 737, '763': 738, '764': 739, '765': 740, '766': 741, '767': 742, '768': 743, '769': 744, '77': 745, '770': 746, '771': 747, '772': 748, '773': 749, '774': 750, '775': 751, '776': 752, '777': 753, '778': 754, '779': 755, '78': 756, '780': 757, '781': 758, '782': 759, '783': 760, '784': 761, '785': 762, '786': 763, '787': 764, '788': 765, '789': 766, '79': 767, '790': 768, '791': 769, '792': 770, '793': 771, '794': 772, '795': 773, '796': 774, '797': 775, '798': 776, '799': 777, '8': 778, '80': 779, '800': 780, '801': 781, '802': 782, '803': 783, '804': 784, '805': 785, '806': 786, '807': 787, '808': 788, '809': 789, '81': 790, '810': 791, '811': 792, '812': 793, '813': 794, '814': 795, '815': 796, '816': 797, '817': 798, '818': 799, '819': 800, '82': 801, '820': 802, '821': 803, '822': 804, '823': 805, '824': 806, '825': 807, '826': 808, '827': 809, '828': 810, '829': 811, '83': 812, '830': 813, '831': 814, '832': 815, '833': 816, '834': 817, '835': 818, '836': 819, '837': 820, '838': 821, '839': 822, '84': 823, '840': 824, '841': 825, '842': 826, '843': 827, '844': 828, '845': 829, '846': 830, '847': 831, '848': 832, '849': 833, '85': 834, '850': 835, '851': 836, '852': 837, '853': 838, '854': 839, '855': 840, '856': 841, '857': 842, '858': 843, '859': 844, '86': 845, '860': 846, '861': 847, '862': 848, '863': 849, '864': 850, '865': 851, '866': 852, '867': 853, '868': 854, '869': 855, '87': 856, '870': 857, '871': 858, '872': 859, '873': 860, '874': 861, '875': 862, '876': 863, '877': 864, '878': 865, '879': 866, '88': 867, '880': 868, '881': 869, '882': 870, '883': 871, '884': 872, '885': 873, '886': 874, '887': 875, '888': 876, '889': 877, '89': 878, '890': 879, '891': 880, '892': 881, '893': 882, '894': 883, '895': 884, '896': 885, '897': 886, '898': 887, '899': 888, '9': 889, '90': 890, '900': 891, '901': 892, '902': 893, '903': 894, '904': 895, '905': 896, '906': 897, '907': 898, '908': 899, '909': 900, '91': 901, '910': 902, '911': 903, '912': 904, '913': 905, '914': 906, '915': 907, '916': 908, '917': 909, '918': 910, '919': 911, '92': 912, '920': 913, '921': 914, '922': 915, '923': 916, '924': 917, '925': 918, '926': 919, '927': 920, '928': 921, '929': 922, '93': 923, '930': 924, '931': 925, '932': 926, '933': 927, '934': 928, '935': 929, '936': 930, '937': 931, '938': 932, '939': 933, '94': 934, '940': 935, '941': 936, '942': 937, '943': 938, '944': 939, '945': 940, '946': 941, '947': 942, '948': 943, '949': 944, '95': 945, '950': 946, '951': 947, '952': 948, '953': 949, '954': 950, '955': 951, '956': 952, '957': 953, '958': 954, '959': 955, '96': 956, '960': 957, '961': 958, '962': 959, '963': 960, '964': 961, '965': 962, '966': 963, '967': 964, '968': 965, '969': 966, '97': 967, '970': 968, '971': 969, '972': 970, '973': 971, '974': 972, '975': 973, '976': 974, '977': 975, '978': 976, '979': 977, '98': 978, '980': 979, '981': 980, '982': 981, '983': 982, '984': 983, '985': 984, '986': 985, '987': 986, '988': 987, '989': 988, '99': 989, '990': 990, '991': 991, '992': 992, '993': 993, '994': 994, '995': 995, '996': 996, '997': 997, '998': 998, '999': 999}
oht_index_to_class = {v: k for k, v in class_to_oht_index.items()}



torch.backends.cudnn.benchmark = True
_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                    help='dataset type (default: ImageFolder/ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224),'
                                          ' uses model default if empty')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=50, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-test-pool', dest='no_test_pool', action='store_true',
                    help='disable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=True,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--legacy-jit', dest='legacy_jit', action='store_true',
                    help='use legacy jit mode for pytorch 1.5/1.5.1/1.6 to get back fusion performance')
parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--mode_validation', default='test')



def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.legacy_jit:
        set_jit_legacy()

    # create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript,
        img_size=args.img_size)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema, strict=False)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model, use_test_size=True)
    test_time_pool = False
    if not args.no_test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=True)

    if args.torchscript:
        torch.jit.optimized_execution(True)
        model = torch.jit.script(model)

    model = model.cuda()
    if args.apex_amp:
        model = amp.initialize(model, opt_level='O1')

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().cuda()

    dataset = create_dataset(
        root=args.data, name=args.dataset, split=args.split,
        load_bytes=args.tf_preprocessing, class_map=args.class_map)

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = {int(line.rstrip()) for line in f}
            valid_labels = [i in valid_labels for i in range(args.num_classes)]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        pin_memory=args.pin_mem,
        tf_preprocessing=args.tf_preprocessing)

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()
    img_filenames = []
    img_preds = []
    all_predictions_by_image = []
    with torch.inference_mode():
        for idx, item in tqdm(enumerate(loader), total=len(loader)):
            img_filename = loader.sampler.data_source.parser.samples[idx][0].split('/')[-1]
            img, img_id = item
            img = img.cuda()
            out = model(img)

            out_all_predictions = out.cpu().detach().numpy()
            all_predictions_by_image.append(out_all_predictions)

            pred = torch.argmax(out, 1).item()
            img_filenames += [img_filename]
            img_preds += [pred]

        arr_predictions = np.array(all_predictions_by_image)
        arr_filenames = np.array(img_filenames)
        np.savez_compressed(args.mode_validation, scores_array=arr_predictions.astype(np.float16),
                            file_array=arr_filenames)

    df = pd.DataFrame({
        "id": img_filenames,
        "predicted": img_preds
    })

    df.to_csv("iccv_preds.csv", index=False)

    """
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + data_config['input_size']).cuda()
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        model(input)
        end = time.time()

        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.cuda()
                input = input.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                output = model(input)
                class_index = torch.argmax(output)
            if isinstance(output, (tuple, list)):
                output = output[0]
            if valid_labels is not None:
                output = output[:, valid_labels]
            loss = criterion(output, target)



            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        batch_idx, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses, top1=top1, top5=top5))

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        cropt_pct=crop_pct,
        interpolation=data_config['interpolation'])

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err']))
    
    """

    return df


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(pretrained=True, exclude_filters=['*in21k'])
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(args.model)
            model_cfgs = [(n, '') for n in model_names]

    if len(model_cfgs):
        results_file = args.results_file or './results-all.csv'
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            start_batch_size = args.batch_size
            for m, c in model_cfgs:
                batch_size = start_batch_size
                args.model = m
                args.checkpoint = c
                result = OrderedDict(model=args.model)
                r = {}
                while not r and batch_size >= args.num_gpu:
                    torch.cuda.empty_cache()
                    try:
                        args.batch_size = batch_size
                        print('Validating with batch size: %d' % args.batch_size)
                        r = validate(args)
                    except RuntimeError as e:
                        if batch_size <= args.num_gpu:
                            print("Validation failed with no ability to reduce batch size. Exiting.")
                            raise e
                        batch_size = max(batch_size // 2, args.num_gpu)
                        print("Validation failed, reducing batch size by 50%")
                result.update(r)
                if args.checkpoint:
                    result['checkpoint'] = args.checkpoint
                results.append(result)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
        if len(results):
            write_results(results_file, results)
    else:
        validate(args)

def write_results(results_file, results):
    with open(results_file, mode='w') as cf:
        dw = csv.DictWriter(cf, fieldnames=results[0].keys())
        dw.writeheader()
        for r in results:
            dw.writerow(r)
        cf.flush()

if __name__ == '__main__':
    main()
