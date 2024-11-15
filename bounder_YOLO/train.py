from ultralytics import YOLO
from yaml import load, FullLoader
from utils import aggregate_run_results
from augment_dataset import augment_dataset


class manga109_YOLO_trainer:
    def __init__(self, test=False):
        self.hyperparameters = None
        self.augmentations = None
        self.p = None
        self.testing = test

    def set_params(self, epochs, batch, imgsz):
        # Sets params for training/testing based on the self.testing flag
        if self.testing:
            return {
                'epochs': 1,
                'batch': 12,
                'imgsz': 256,
                'fraction': 0.0001
            }
        else:
            return {
                'epochs': epochs,
                'batch': batch,
                'imgsz': imgsz,
            }

    def __train0__(self):
        # 16 epochs on clean dataset with high image resolution
        model = YOLO("YOLOv8s.pt").to('cuda')
        params = self.set_params(epochs=16, batch=12, imgsz=1024)
        model.train(
            data='yaml_files/manga109.yaml',
            nbs=64,
            val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
            **self.hyperparameters,
            **params
        )

    def __train1__(self):
        # 16 epochs on clean dataset at lower image size
        model = YOLO("runs/detect/train/weights/last.pt").to('cuda')
        params = self.set_params(epochs=16, batch=16, imgsz=512)
        model.train(
            data='yaml_files/manga109.yaml',
            nbs=64,
            val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
            **self.hyperparameters,
            **params
        )

    def __train2__(self):
        # 8 epochs with progressive augmentation
        for i in range(1, 5):
            pass
            # self.p += .0625
            # augment_dataset(self.augmentations, self.p)

            model = YOLO(f"runs/detect/train{i + 1}/weights/last.pt").to('cuda')
            params = self.set_params(epochs=2, batch=12, imgsz=1024)
            model.train(
                # data='yaml_files/manga109_aug.yaml',
                data='yaml_files/manga109.yaml',
                nbs=64,
                val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
                augment=True,
                **self.hyperparameters,
                **params
            )

    def __train3__(self):
        # 8 epochs with decreasing augmentation and increasing dropout
        for i in range(8):
            if i % 2 == 0:
                pass
                # self.p = self.p - .0625
                # augment_dataset(self.augmentations, self.p)

            model = YOLO(f"runs/detect/train{i + 5}/weights/last.pt").to('cuda')
            self.hyperparameters['dropout'] += .01
            params = self.set_params(epochs=2, batch=12, imgsz=1024)
            model.train(
                # data='yaml_files/manga109_aug.yaml',
                data='yaml_files/manga109.yaml',
                nbs=64,
                val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
                augment=True,
                **self.hyperparameters,
                **params
            )

    def __train4__(self):
        # 8 epochs with decreasing dropout and increasing frozen layers
        for i in range(8):
            self.hyperparameters['dropout'] -= .01
            self.hyperparameters['freeze'] += 1
            model = YOLO(f"runs/detect/train{i + 13}/weights/last.pt").to('cuda')
            params = self.set_params(epochs=2, batch=12, imgsz=1024)
            model.train(
                data='yaml_files/manga109.yaml',
                nbs=64,
                val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
                **self.hyperparameters,
                **params
            )

    def __train5__(self):
        # 8 epochs with decreasing frozen layers
        for i in range(8):
            self.hyperparameters['freeze'] -= 1
            model = YOLO(f"runs/detect/train{i + 21}/weights/last.pt").to('cuda')
            params = self.set_params(epochs=2, batch=12, imgsz=1024)
            model.train(
                data='yaml_files/manga109.yaml',
                nbs=64,
                val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
                **self.hyperparameters,
                **params
            )

    def __train6__(self):
        # 8 epochs with YOLO augmented dataset and best hyperparameters
        model = YOLO("runs/detect/train29/weights/last.pt").to('cuda')
        params = self.set_params(epochs=8, batch=12, imgsz=1024)
        model.train(
            data='yaml_files/manga109.yaml',
            nbs=64,
            val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
            augment=True,
            **self.hyperparameters,
            **params
        )

    def __train7__(self):
        # 8 epochs on clean dataset with default YOLO
        model = YOLO("runs/detect/train30/weights/last.pt").to('cuda')
        params = self.set_params(epochs=8, batch=16, imgsz=1024)
        model.train(
            data='yaml_files/manga109.yaml',
            nbs=64,
            val=True, save=True, plots=True, verbose=False, device='cuda',
            **params
        )

    def train(self):
        # Custom training loop
        # with open('yaml_files/hyperparameters.yaml') as file:
        #     self.hyperparameters = load(file, Loader=FullLoader)
        self.hyperparameters = {'weight_decay': 0.0005}
        self.__train0__()
        self.__train1__()

        # with open('yaml_files/augmentations.yaml') as file:
        #     self.augmentations = load(file, Loader=FullLoader)
        # self.p = .25
        # self.__train2__()

        # self.hyperparameters['dropout'] = .01
        # self.__train3__()

        # self.hyperparameters['freeze'] = 8
        # self.__train4__()

        # self.__train5__()

        # with open('yaml_files/manga109_best.yaml') as file:
        #     self.hyperparameters = load(file, Loader=FullLoader)
        # self.__train6__()

        # self.__train7__()

        aggregate_run_results()
        return


def main():
    # test_trainer = manga109_YOLO_trainer(test=True)
    # test_trainer.train()

    trainer = manga109_YOLO_trainer()
    trainer.train()
    return


if __name__ == "__main__":
    main()
