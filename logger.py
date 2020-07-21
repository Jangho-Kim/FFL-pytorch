from torch.utils.tensorboard import SummaryWriter


class SummaryLogger(SummaryWriter):

    def __init__(self, path):
        super().__init__()
        file_path = 'logs/' + path
        self.logger = SummaryWriter(file_path)

    def add_scalar_group(self, main_tag, tag_scalar_dict, global_step):
        for sub_tag, scalar in tag_scalar_dict.items():
            self.logger.add_scalar(main_tag+'/{}'.format(sub_tag), scalar, global_step)


####################################################################
"""
Convenience function for logging multiple scalars of the same group
Example Below: """
# logger = SummaryLogger('logs')
#
# import numpy as np
# scalar_dict = {'Loss1': np.random.random(),
#                'Loss2': np.random.random(),
#                'Loss3.': np.random.random()}
# main_tag ='Train'
# for i in range(100):
#     global_step = i
#     logger.add_scalar_group(main_tag, scalar_dict, global_step)
#
# logger.add_scalar('tag',5,5)
#######################################################################