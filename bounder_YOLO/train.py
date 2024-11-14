from ultralytics import YOLO
from yaml import load, FullLoader
from utils import aggregate_run_results
from augment_dataset import augment_dataset


class manga109_YOLO_trainer:
    def __init__(self):
        self.hyperparameters = None
        self.augmentations = None
        self.p = None

    def __train0__(self):
        # 16 epochs on clean dataset starting with high image using hyperparameters from tuning
        model = YOLO("YOLOv8m.pt").to('cuda')
        model.train(
            data='yaml_files/manga109.yaml',
            # epochs=16, # default
            epochs=1,  # for testing
            fraction=.0001,  # for testing
            batch=12,
            nbs=64,
            # imgsz=1024, # default
            imgsz=256,  # for testing
            amp=True, val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
            **self.hyperparameters
        )

    def __train1__(self):
        # 16 epochs on clean dataset at lower image size
        model = YOLO("runs/detect/train/weights/last.pt").to('cuda')
        model.train(
            data='yaml_files/manga109.yaml',
            # epochs=16, # default
            epochs=1,  # for testing
            fraction=.0001,  # for testing
            batch=16,
            nbs=64,
            # imgsz=512, # default
            imgsz=256,  # for testing
            amp=True, val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
            **self.hyperparameters
        )

    def __train2__(self):
        # start augmentation params at 25% of value increasee every 2 epochs peak at 100% at epoch 8 after peak reduce by 25% every 2 epochs until 0% at epoch 16
        # 8 epoch with progressive augmentation
        for i in range(1, 5):  # start i at 1
            self.p += .0625
            # augment_dataset(augmentations, p)

            # train model with new settings for 2 epochs
            # get new model / update path to model
            model = YOLO(f"runs/detect/train{i + 1}/weights/last.pt").to('cuda')
            model.train(
                data='yaml_files/manga109_aug.yaml',  # used augmented dataset
                # epochs=2,  # default
                epochs=1,  # for testing
                fraction=.0001,  # for testing
                batch=12,
                nbs=64,
                # imgsz=1024, # default
                imgsz=256,  # for testing
                amp=True, val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
                **self.hyperparameters
            )

    def __train3__(self):
        # start dropout at epoch 8 at 0.01 increase by 0.01 every epoch peak at 0.08 at epoch 16 and reduce by 0.01 every epoch until 0.01 at epoch 24
        # 8 epoch with decreasing augmentation and increasing dropout
        for i in range(8):
            # decrease augmentation settings every 2 epochs
            if i % 2 == 0:
                self.p = self.p - .0625
                # augment_dataset(augmentations, p)

            # increase dropout every epoch
            self.hyperparameters['dropout'] += .01

            # train model with new settings for 2 epochs
            # get new model / update path to model
            model = YOLO(f"runs/detect/train{i + 5}/weights/last.pt").to('cuda')
            model.train(
                data='yaml_files/manga109_aug.yaml',  # used augmented dataset
                # epochs=2, # default
                epochs=1,  # for testing
                fraction=.0001,  # for testing
                batch=12,
                nbs=64,
                # imgsz=1024, # default
                imgsz=256,  # for testing
                amp=True, val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
                **self.hyperparameters
            )

    def __train4__(self):
        # start frozen layers increasing by 8 layer every epoch to peak at 16 layers on epoch 24 and reduce by 1 layer every epoch until 8 layers at epoch 32
        # 8 epoch with decreasing dropout and increasing frozen layers
        for i in range(8):
            # decrease dropout every epoch
            self.hyperparameters['dropout'] -= .01

            # increase frozen layers every epoch
            self.hyperparameters['freeze'] += 1

            # train model with new settings for 2 epochs
            # get new model / update path to model
            model = YOLO(f"runs/detect/train{i + 13}/weights/last.pt").to('cuda')
            model.train(
                data='yaml_files/manga109.yaml',  # used clean dataset
                # epochs=2, # default
                epochs=1,  # for testing
                fraction=.0001,  # for testing
                batch=12,
                nbs=64,
                # imgsz=1024, # default
                imgsz=256,  # for testing
                amp=True, val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
                **self.hyperparameters
            )

    def __train5__(self):
        # desecnd pyramids
        # 8 epoch with decreasing frozen layers
        for i in range(8):
            # decrease frozen layers every epoch
            self.hyperparameters['freeze'] -= 1

            # train model with new settings for 2 epochs
            # get new model / update path to model
            model = YOLO(f"runs/detect/train{i + 21}/weights/last.pt").to('cuda')
            model.train(
                data='yaml_files/manga109.yaml',  # used clean dataset
                # epochs=2, # default
                epochs=1,  # for testing
                fraction=.0001,  # for testing
                batch=12,
                nbs=64,
                # imgsz=1024, # default
                imgsz=256,  # for testing
                amp=True, val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
                **self.hyperparameters
            )

    def __train6__(self):
        # train model for 8 epochs using augmented dataset and manga109 best hyperparameters
        model = YOLO("runs/detect/train29/weights/last.pt").to('cuda')
        model.train(
            data='yaml_files/manga109_aug.yaml',
            # epochs=8, # default
            epochs=1,  # for testing
            fraction=.0001,  # for testing
            batch=12,
            nbs=64,
            # imgsz=1024, # default
            imgsz=256,  # for testing
            amp=True, val=True, save=True, plots=True, verbose=False, device='cuda', pretrained=True,
            **self.hyperparameters
        )

    def __train7__(self):
        # train model with clean dataset for 8 epochs using default yolo
        model = YOLO("runs/detect/train30/weights/last.pt").to('cuda')
        model.train(
            data='yaml_files/manga109.yaml',
            # epochs=8, # default
            epochs=1,  # for testing
            fraction=.0001,  # for testing
            batch=16,
            nbs=64,
            amp=True, val=True, save=True, plots=True, verbose=False, device='cuda'
        )

    def train(self):
        """
        Custom training loop for fine-tuning YOLOv8n model on manga109 dataset with class weight adjustment and pyramid scheduled dataset augmentation overlapping with pyramid scheduled incremental frozen layers and dropout starting at peak of dataset augmentation and ending with clean dataset training.
        """

        with open('yaml_files/hyperparameters.yaml') as file:
            self.hyperparameters = load(file, Loader=FullLoader)
        # self.__train0__()
        # self.__train1__()

        # augmentation settings
        with open('yaml_files/augmentations.yaml') as file:
            self.augmentations = load(file, Loader=FullLoader)
        self.p = .25
        # augment_dataset(self.augmentations, p)
        # self.__train2__()

        # dropout
        self.hyperparameters['dropout'] = .01
        self.__train3__()

        # freeze
        self.hyperparameters['freeze'] = 8
        self.__train4__()

        # descend
        self.__train5__()

        # best
        with open('yaml_files/augmentations_best.yaml') as file:
            self.augmentations = load(file, Loader=FullLoader)
        p = .5
        augment_dataset(self.augmentations, p)
        with open('yaml_files/manga109_best.yaml') as file:
            self.hyperparameters = load(file, Loader=FullLoader)
        self.__train6__()

        # yolo
        self.__train7__()

        # finally
        aggregate_run_results()

        return

#TODO RATHER THAN AUGMENTING THE DATASET JUST USE YOLO AND HOPE IT WORKS AND KEEP THIS AS A BACKUP
# FIND OUT HOW TO DO PROGRESSIVE AUGMENTATION USING YOLO
# FIND OUT HOW TO DO PROGRESSIVE DROPOUT AND FREEZE USING YOLO (AS WELL AS REVERSE)
# REMEBER TO CHAMGE TO S MODEL
# FIGURE OUT HOW TO SAVE RAM WHEN RUNNING THIS (AGUMENT DATASET seems to be the main problem)

def main():
    trainer = manga109_YOLO_trainer()
    trainer.train()

    return


if __name__ == "__main__":
    main()
