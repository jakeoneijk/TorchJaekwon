from HParams import HParams
HParams()
from TorchJaekwon.Controller import Controller

if __name__ == '__main__':
    controller = Controller()
    controller.set_argparse()
    controller.run()