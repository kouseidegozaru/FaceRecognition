import os
import pathlib

resource_info_path = os.path.abspath(__file__)
PROJECT_PATH = pathlib.Path(__file__).parent.parent.resolve()

PERSON_NAMES = ['kento','ryou','syou']
TRAIN_IMAGE_IDS = [0,1,2]
TRAIN_IMAGE_NAMES = ['yamazaki_kento.jpg','yoshizawa_ryou.jpg','hirano_syou.jpg']
TRAIN_IMAGE_DIR_PATH = os.path.join(PROJECT_PATH,'images','train')