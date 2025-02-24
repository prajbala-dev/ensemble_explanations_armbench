import os
import json
import argparse
import pytorch_lightning as pl
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import ImageFolder
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint


## Choose which model to train and comment the others. Be sure to change it at initialization as well

from resnet50_model import defect_resnet50
#from resnet18_model import defect_resnet18
#from resnet101_model import defect_resnet101
#from vit_model_base import defect_vit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset_path", required=True)
    parser.add_argument("-m", "--mode", default="train",
                        help="Set model to 'train' or 'test'", required=False)
    parser.add_argument("-c", "--checkpoint",
                        help="Path to checkpoint", required=False)
    parser.add_argument("-r", "--resume", action='store_true', default=False)

    args = parser.parse_args()
    mode = args.mode
    dataset_path = args.dataset_path
    train_path = os.path.join(args.dataset_path, "imagenet/train")
    val_path = os.path.join(args.dataset_path, "imagenet/test")

    output_folder = os.path.join(dataset_path, "output")

    img_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        monitor="val_loss",
        mode="min",
        save_last=True,
        dirpath=output_folder,
        filename="sample-{epoch:02d}-{val_loss:.2f}",
        every_n_epochs=5,
    )

    model = defect_resnet50()

    if mode == "train":
        train_dataset = ImageFolder(root=train_path, transform=img_transforms)
        train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                                    num_workers=8)
        val_dataset = ImageFolder(root=train_path, transform=img_transforms)
        val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                                    num_workers=8)

        trainer = pl.Trainer(accelerator='gpu',
                        default_root_dir=output_folder,
                        callbacks=[checkpoint_callback],
                        check_val_every_n_epoch=10,
                        max_epochs=200,
                        )
        

        if args.resume:
            print ("Resuming from checkpoint: ", args.checkpoint)
            trainer.fit(model,
                        train_dataloader,
                        val_dataloader,
                        ckpt_path=args.checkpoint)
        else:
            trainer.fit(model, train_dataloader, val_dataloader)
    else:
        test_dataset = ImageFolder(root=val_path, transform=img_transforms)
        test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                                    num_workers=8)

        model = defect_resnet50.load_from_checkpoint(args.checkpoint)
        trainer = pl.Trainer(accelerator='gpu',
                    default_root_dir=output_folder,
                    callbacks=[checkpoint_callback],
                    check_val_every_n_epoch=5,
                    max_epochs=100, devices=1,num_nodes=1, inference_mode="validate",
                    )

        predictions = trainer.predict(model, test_dataloader)
        pred_list = list()
        for preds in predictions:
            for p in preds:
                pred_list.append(p.cpu().detach().numpy().tolist())

        result = {'predictions': pred_list, 
                  'targets': test_dataset.targets}
                    

        results_location = os.path.join(output_folder, 'output.txt')

        with open(results_location, 'w', encoding='UTF8', newline='') as f:
            json.dump(result, f)
        
