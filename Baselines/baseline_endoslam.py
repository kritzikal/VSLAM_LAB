import os.path

from huggingface_hub import hf_hub_download

from utilities import print_msg
from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class ENDOSLAM_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='endoslam', baseline_folder='EndoSLAM'):

        checkpoints_dir = os.path.join(VSLAMLAB_BASELINES, baseline_folder, 'checkpoints')

        default_parameters = {
            'verbose': 1,
            'mode': 'mono',
            'pretrained_posenet': os.path.join(checkpoints_dir, 'exp_pose_model_best.pth.tar'),
            'pretrained_dispnet': os.path.join(checkpoints_dir, 'dispnet_model_best.pth.tar')
        }

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'purple'
        self.modes = ['mono']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)

    def git_clone(self):
        super().git_clone()
        self.endoslam_download_weights()

    def is_installed(self):
        return (True, 'is installed') if self.is_cloned() else (False, 'not installed (conda package available)')

    def endoslam_download_weights(self):
        """Download pretrained PoseNet and DispNet weights"""
        checkpoints_dir = os.path.join(self.baseline_path, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        files = [
            os.path.join(checkpoints_dir, "dispnet_model_best.pth.tar"),
            os.path.join(checkpoints_dir, "exp_pose_model_best.pth.tar")
        ]

        for file in files:
            file_name = os.path.basename(file)
            if not os.path.exists(file):
                print_msg(f"\n{SCRIPT_LABEL}", f"Download weights: {file}", 'info')
                _ = hf_hub_download(repo_id='kritzikal/endoslam_weights', filename=file_name,
                                    repo_type='model', local_dir=checkpoints_dir)
