import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vit import HeadlessVisionTransformer

class Model(nn.Module):
    def __init__(self, embed_dim, num_classes):
        super().__init__()

        self.base = HeadlessVisionTransformer()

        self.cset_fc = nn.Linear(embed_dim, num_classes)
        self.oset_fc = nn.Linear(embed_dim, self.num_classes * 2)

    def forward(self, batch):
        imgs, labels = batch['img'], batch['label']

        if not self.training:
            feats = self.base(imgs)
            feats_flatten = torch.flatten(feats, 1)
            cset_logit = self.cset_fc(feats_flatten)
            oset_logit = self.oset_fc(feats_flatten).view(imgs.size(0), 2, self.num_classes)
            oset_prob = F.softmax(oset_logit, dim=1)

            cset_pred = torch.max(cset_logit, dim=1)[1]
            oset_pred = oset_prob[torch.arange(imgs.size(0)), 1, cset_pred] > 0.5
            return cset_pred, oset_pred, feats_flatten

        half_bs = imgs.size(0) // 2
        source_imgs, source_labels = imgs[:half_bs], labels[:half_bs]
        target_imgs = imgs[half_bs:]

        source_feats = self.base(source_imgs)
        source_feats = torch.flatten(source_feats, 1)
        source_cset_logit = self.cset_fc(source_feats)
        source_oset_logit = self.oset_fc(source_feats).view(half_bs, 2, self.num_classes)
        source_oset_prob = F.softmax(source_oset_logit, dim=1)

        source_oset_pos_target = torch.zeros_like(source_cset_logit)
        source_oset_pos_target[torch.arange(half_bs), source_labels] = 1
        source_oset_neg_target = 1 - source_oset_pos_target

        source_cset_loss = F.cross_entropy(source_cset_logit, source_labels)
        source_oset_pos_loss = torch.mean(torch.sum(-source_oset_pos_target * torch.log(source_oset_prob[:,0,:] + 1e-8), dim=1))
        source_oset_neg_loss = torch.mean(torch.max(-source_oset_neg_target * torch.log(source_oset_prob[:,1,:] + 1e-8), dim=1)[0])
        source_oset_loss = source_oset_pos_loss + source_oset_neg_loss

        loss = source_cset_loss * self.loss_weights['source_cset'] + \
            source_oset_loss * self.loss_weights['source_oset']
        metric = {
            'loss': loss.item(),
            'source_cset_loss': source_cset_loss.item(),
            'source_oset_loss': source_oset_loss.item()
        }
        return loss, metric

def ViTOVANet():
    # INSTANTIATE MODEL 
    model = Model()

    # 

    return