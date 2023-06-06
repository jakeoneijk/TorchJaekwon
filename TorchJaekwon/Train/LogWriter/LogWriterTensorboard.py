from tensorboardX import SummaryWriter

from TorchJaekwon.Train.LogWriter.LogWriter import LogWriter
from TorchJaekwon.DataProcess.Util.UtilAudioSTFT import UtilAudioSTFT

class LogWriterTensorboard(LogWriter):
    def __init__(self) -> None:
        super().__init__()
        self.tensorboard_writer = SummaryWriter(log_dir=self.log_path["visualizer"])
    
    def visualizer_log(
        self,
        x_axis_name:str,
        x_axis_value:float,
        y_axis_name:str,
        y_axis_value:float) -> None:
        self.tensorboard_writer.add_scalar(y_axis_name,y_axis_value,x_axis_value)
    
    def plot_wav(self, name:str, wav, global_step):
        self.tensorboard_writer.add_audio(name, wav, sample_rate=self.h_params.preprocess.sample_rate, global_step=global_step)
    
    def plot_mel(self, name:str, spec, vmin=-6.0, vmax=1.5,transposed=False, global_step=0):
        figure = UtilAudioSTFT.spec_to_figure(spec, vmin=vmin, vmax=vmax,transposed=transposed)
        self.tensorboard_writer.add_figure(name,figure,global_step=global_step)


