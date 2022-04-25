from PatchCore_anomaly_detection.train import *

import pytorch_lightning as pl
import timm
import torch.nn.functional as F
from torchmetrics import F1Score
from sklearn.preprocessing import LabelEncoder

class BaseModel(pl.LightningModule):
    def __init__(self, meta_datas, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.meta_datas = meta_datas
        self.model      = timm.create_model('efficientnet_b0', pretrained=True, num_classes=self.hparams['n_classes'])
        self.F1Score    = F1Score(num_classes=self.hparams['n_classes'], average='macro')
        self.freeze()
    def forward(self, x):
        return self.model(x)
    def freeze(self):
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    def unfreeze(self):
        for param in self.model.parameters():
            param.requires_grad = True
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.hparams['lr'])
    def train_dataloader(self):
        print("> Training dataset:", len(self.meta_datas['train']))
        self.y_enc         = LabelEncoder()
        labels             = self.y_enc.fit_transform(self.meta_datas['train'][self.hparams['label']])
        self.class_weights = torch.tensor([sum(labels == c) for c in range(len(self.y_enc.classes_))], dtype=torch.float32).cuda()
        datasets           = MVTecDataset(self.meta_datas['train'], transform=get_data_transform(self.hparams['load_size'], self.hparams['crop_size'], training=True),
                                          phase='train', input_dir=self.hparams['input_dir'], labels=labels)
        return DataLoader(datasets, batch_size=self.hparams['batch_size'], shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    def val_dataloader(self):
        print("> Validation dataset:", len(self.meta_datas['val']))
        labels   = self.y_enc.transform(self.meta_datas['val'][self.hparams['label']])
        datasets = MVTecDataset(self.meta_datas['val'], transform=get_data_transform(self.hparams['load_size'], self.hparams['crop_size'], training=False),
                                phase='val', input_dir=self.hparams['input_dir'], labels=labels)
        return DataLoader(datasets, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    def predict_dataloader(self):
        print("> Test dataset:", len(self.meta_datas['test']))
        datasets = MVTecDataset(self.meta_datas['test'], transform=get_data_transform(self.hparams['load_size'], self.hparams['crop_size'], training=False),
                                phase='test', input_dir=self.hparams['input_dir'])
        return DataLoader(datasets, batch_size=1, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)
    def step(self, batch):
        X, y = batch
        pred = self(X)
        loss = F.cross_entropy(pred, y, weight=self.class_weights)
        return loss, y, pred
    def training_step(self, batch, batch_idx):
        loss, y, pred = self.step(batch)
        f1_score      = self.F1Score(y, pred.argmax(1))
        return {'loss': loss, 'f1_score': f1_score}
    def training_epoch_end(self, outputs):
        avg_loss     = torch.stack([x['loss'] for x in outputs]).mean()
        avg_f1_score = torch.stack([x['f1_score'] for x in outputs]).mean()
        print("\n  > training_loss:", avg_loss, "\t training_f1_score:", avg_f1_score)
    def validation_step(self, batch, batch_idx):
        loss, y, pred = self.step(batch)
        f1_score      = self.F1Score(y, pred.argmax(1))
        return {'loss': loss, 'y': y.detach(), 'pred': pred.detach(), 'f1_score': f1_score}
    def validation_epoch_end(self, outputs):
        avg_loss     = torch.stack([x['loss'] for x in outputs]).mean()
        avg_f1_score = torch.stack([x['f1_score'] for x in outputs]).mean()
        print("\n  > val_loss:", avg_loss, "\t val_f1_score:", avg_f1_score)
        self.log('val_loss', avg_loss);  self.log('f1_score', avg_f1_score)
        return {'val_loss': avg_loss, 'f1_score': avg_f1_score}
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        X, _ = batch
        pred = torch.argmax(self(X), dim=1).cpu()
        return self.y_enc.inverse_transform(pred)[0]
