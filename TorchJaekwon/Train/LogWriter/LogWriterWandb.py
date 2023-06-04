from TorchJAEKWON.Train.LogWriter.LogWriter import LogWriter

class LogWriterWandb(LogWriter):
    def __init__(self,model) -> None:
        super().__init__()