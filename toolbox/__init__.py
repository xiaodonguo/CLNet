from .metrics import averageMeter, runningScore
from .log import get_logger
from .optim.AdamW import AdamW
from .optim.Lookahead import Lookahead
from .optim.RAdam import RAdam
from .optim.Ranger import Ranger


from .utils import ClassWeight, save_ckpt, load_ckpt, class_to_RGB, \
    compute_speed, setup_seed, group_weight_decay


def get_dataset(cfg):
    assert cfg['dataset'] in ['nyuv2', 'sunrgbd', 'cityscapes', 'camvid', 'irseg', 'pst900', "glassrgbt", "mirrorrgbd", 'glassrgbt_merged']

    if cfg['dataset'] == 'irseg':
        from .datasets.irseg import IRSeg
        # return IRSeg(cfg, mode='trainval'), IRSeg(cfg, mode='test')
        return IRSeg(cfg, mode='train'), IRSeg(cfg, mode='val'), IRSeg(cfg, mode='test')

def get_model(cfg):

    ############# bbbmodel ################
    if cfg['model_name'] == 'MMSMCNet':
        from Semantic_Segmentation_Street_Scenes.model_others import nation
        return nation()

    if cfg['model_name'] == 'TLDNet':
        from Semantic_Segmentation_Street_Scenes.model_others.TLDNet.TLDNet import Teacher
        return Teacher()

    if cfg['model_name'] == 'HAINet':
        from Semantic_Segmentation_Street_Scenes.model_others.HAINet.HAINet import HAIMNet_VGG
        return HAIMNet_VGG()

    if cfg['model_name'] == 'C2DFNet':
        from Semantic_Segmentation_Street_Scenes.model_others.C2DFNet.DualFastnet_res import DualFastnet
        return DualFastnet()

    if cfg['model_name'] == 'BBSNet':
        from Semantic_Segmentation_Street_Scenes.model_others.BBS.BBSNet_model import BBSNet
        return BBSNet(channel=32)

   ## student

    if cfg['model_name'] == 'model5_b0':
        from Semantic_Segmentation_Street_Scenes.proposed.student.model5_b0 import Model
        return Model()

    if cfg['model_name'] == 'model5_b1':
        from Semantic_Segmentation_Street_Scenes.proposed.student.model5_b1 import Model
        return Model()

    if cfg['model_name'] == 'model5_b2':
        from Semantic_Segmentation_Street_Scenes.proposed.student.model5_b2 import Model
        return Model()

    if cfg['model_name'] == 'model5_b2(share)':
        from Semantic_Segmentation_Street_Scenes.proposed.student.model5_b2_share import Model
        return Model()

    if cfg['model_name'] == 'pts900_student':
        from Semantic_Segmentation_Street_Scenes.proposed.student.model5_b2_share_pts900 import Model
        return Model()

    ## teacher
    if cfg['model_name'] == 'model5_b3':
        from Semantic_Segmentation_Street_Scenes.proposed.teacher.b3.model5 import Teacher
        return Teacher()

    if cfg['model_name'] == 'model5_b4':
        from Semantic_Segmentation_Street_Scenes.proposed.teacher.b4.model5 import Teacher
        return Teacher()

    if cfg['model_name'] == 'model5_b5':
        from Semantic_Segmentation_Street_Scenes.proposed.teacher.b5.model5 import Teacher
        return Teacher()

    if cfg['model_name'] == 'pts900_teacher':
        from Semantic_Segmentation_Street_Scenes.proposed.teacher.b4.model5_pst900 import Teacher
        return Teacher()



#### KD
    if cfg['model_name'] == 'KD_Model1_b0':
        from Semantic_Segmentation_Street_Scenes.proposed.distillation.student.b0.KD_Model1 import Student
        return Student(distillation=True)

    if cfg['model_name'] == 'KD_Model2_b0':
        from Semantic_Segmentation_Street_Scenes.proposed.distillation.student.b0.KD_Model2 import Student
        return Student(distillation=True)

    if cfg['model_name'] == 'KD_Model1_b1':
        from Semantic_Segmentation_Street_Scenes.proposed.distillation.student.b1.KD_Model1 import Student
        return Student(distillation=True)

    if cfg['model_name'] == 'KD_b1_model1':
        from Semantic_Segmentation_Street_Scenes.proposed.distillation.student.b1.KD_Model2 import Student
        return Student(distillation=True)

    if cfg['model_name'] == 'KD_b2_model1':
        from Semantic_Segmentation_Street_Scenes.proposed.distillation.student.b2.student_b2_model1 import Student
        return Student(distillation=True)

    if cfg['model_name'] == 'KD_b2_share_model5':
        from Semantic_Segmentation_Street_Scenes.proposed.distillation.student.b2_share.student_b2_share_irseg import Student
        return Student(distillation=True)

    if cfg['model_name'] == 'pst900_KD':
        from Semantic_Segmentation_Street_Scenes.proposed.distillation.student.b2_share.student_b2_share_pst900 import Student
        return Student(distillation=True)

    ## visualize
    if cfg['model_name'] == 'model5_b4_visualize':
        from Semantic_Segmentation_Street_Scenes.proposed.teacher.b4.model5_visualize import Teacher
        return Teacher()

    ## ablation1

    if cfg['model_name'] == 'withoutMHFI':
        from Semantic_Segmentation_Street_Scenes.proposed.teacher.b4.wo_MHFI import Teacher
        return Teacher()

    if cfg['model_name'] == 'withoutDGSD':
        from Semantic_Segmentation_Street_Scenes.proposed.teacher.b4.wo_DGSD import Teacher
        return Teacher()

    if cfg['model_name'] == 'base':
        from Semantic_Segmentation_Street_Scenes.proposed.teacher.b4.base import Teacher
        return Teacher()












