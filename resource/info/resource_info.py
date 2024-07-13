import os
import pathlib

resource_info_path = os.path.abspath(__file__)
RESOURCE_PATH = pathlib.Path(__file__).parent.parent.resolve()

PERSON_NAMES = {
    0:'kento',
    1:'ryou',
    2:'syou'
}

CURRENT_MODEL_PATH = os.path.join(RESOURCE_PATH,'trained_models','nearest_neighbors.joblib')

TRAIN_IMAGE_NAMES = ['yamazaki_kento.jpg','yoshizawa_ryou.jpg','hirano_syou.jpg']
TRAIN_IMAGE_DIR_PATH = os.path.join(RESOURCE_PATH,'images','train')

TEST_IMAGE_NAMES = ['yamazaki_kento.jpg','yoshizawa_ryou.jpg','hirano_syou.jpg']
TEST_IMAGE_DIR_PATH = os.path.join(RESOURCE_PATH,'images','test')