import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math


class Metric_loss(nn.Module):
    def __init__(self,batch_size, margin=0.8):
        super(Metric_loss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
    def forward(self,inputs,targets):
        n = inputs.size(0)
        center1 = []
        center2 = []

        center_unque1 = []
        center_unque2 = []

        feat1 = inputs[0:(n//2),:]
        feat2 = inputs[(n//2):n,:]

        label1 = targets[0:(n//2)]
        label2 = targets[(n//2):n]
        label_num = len(label1.unique())
        label_uni = label1.unique()
        feat1 = feat1.chunk(label_num, 0)
        feat2 = feat2.chunk(label_num, 0)

        for j in range(label_num):
            center1.append(torch.mean(feat1[j], dim=0).unsqueeze(0))
            center1.append(torch.mean(feat1[j], dim=0).unsqueeze(0))
            center1.append(torch.mean(feat1[j], dim=0).unsqueeze(0))
            center1.append(torch.mean(feat1[j], dim=0).unsqueeze(0))

            center2.append(torch.mean(feat2[j], dim=0).unsqueeze(0))
            center2.append(torch.mean(feat2[j], dim=0).unsqueeze(0))
            center2.append(torch.mean(feat2[j], dim=0).unsqueeze(0))
            center2.append(torch.mean(feat2[j], dim=0).unsqueeze(0))

        for j in range(label_num):
            center_unque1.append(torch.mean(feat1[j], dim=0).unsqueeze(0))
            center_unque2.append(torch.mean(feat2[j], dim=0).unsqueeze(0))

        center1 = torch.cat(center1)
        center2 = torch.cat(center2)



        center =torch.cat([center1,center2],dim=0)
        label_unique =torch.cat([label_uni,label_uni],dim=0)

        center_dist =torch.pow(center,2).sum(dim=1, keepdim=True).expand(n, n)

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = center_dist + dist.t()
        dist.addmm_(1, -2, center, inputs.t())  # dist   = 1 × dist - 2 ×（inputs @ inputs_t）
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []

        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)

        loss = self.ranking_loss(dist_an, dist_ap, y)


        return loss
