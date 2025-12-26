import os.path
import subprocess
import zipfile

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
        """Download pretrained PoseNet and DispNet weights from Dropbox"""
        checkpoints_dir = os.path.join(self.baseline_path, 'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        zip_path = os.path.join(checkpoints_dir, "endoslam_weights.zip")

        # Check if weights already exist
        posenet_path = os.path.join(checkpoints_dir, "exp_pose_model_best.pth.tar")
        dispnet_path = os.path.join(checkpoints_dir, "dispnet_model_best.pth.tar")

        if os.path.exists(posenet_path) and os.path.exists(dispnet_path):
            return

        # Dropbox URL (dl=1 for direct download)
        dropbox_url = "https://www.dropbox.com/scl/fi/nnpzsv6at0mw84axgwp2f/08-13-00-00.zip?rlkey=n93sfqomi53sb6z99f823h8sy&dl=1"

        print_msg(f"\n{SCRIPT_LABEL}", f"Downloading weights from Dropbox...", 'info')

        # Download using wget
        subprocess.run(["wget", "-O", zip_path, dropbox_url], check=True)

        # Extract zip file
        print_msg(f"\n{SCRIPT_LABEL}", f"Extracting weights to {checkpoints_dir}", 'info')
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(checkpoints_dir)

        # Move files from subfolder to checkpoints_dir (zip extracts to 08-13-00:00/)
        subfolder = os.path.join(checkpoints_dir, "08-13-00:00")
        if os.path.isdir(subfolder):
            for filename in os.listdir(subfolder):
                src = os.path.join(subfolder, filename)
                dst = os.path.join(checkpoints_dir, filename)
                os.rename(src, dst)
            os.rmdir(subfolder)

        # Clean up zip file
        os.remove(zip_path)
