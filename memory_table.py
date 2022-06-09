import torch
import torch.cuda

class MemoryTable(object):
    def __init__(self, reliable_num=5, lambda_p=0.5, lambda_c=1):
        self.reliable_num = reliable_num
        self.lambda_p = lambda_p
        self.lambda_c = lambda_c
        self.vis_reliables = []
        self.nir_reliables = []
    def __getitem__(self, index):
        return torch.stack(self.rgb_vectors[index]), torch.stack(self.nir_vectors[index]), self.vis_reliables[index], self.nir_reliables[index]

    def vis_global_distance(self):
        return torch.cdist(self.rgb_vectors, self.rgb_vectors)
    def nir_global_distance(self):
        return torch.cdist(self.nir_vectors, self.nir_vectors)


    def overall_dissimilarity(self, global_dist):
        result = global_dist 
        return result


    def update(self, rgb_vectors, nir_vectors, labels, labels2):
        self.rgb_vectors = rgb_vectors
        self.nir_vectors = nir_vectors
        self.label = labels.cuda()
        self.label2 = labels2.cuda()
        
    def update_reliables(self):
        vis_global_dist = self.vis_global_distance()
        nir_global_dist = self.nir_global_distance()

        vis_overall_diss = self.overall_dissimilarity(vis_global_dist)
        vis_overall_diss = vis_overall_diss.to('cpu')
        nir_overall_diss = self.overall_dissimilarity(nir_global_dist)
        nir_overall_diss = nir_overall_diss.to('cpu')

        self.vis_reliables = vis_overall_diss.argsort(dim=1)[:,1:self.reliable_num+1]
        self.nir_reliables = nir_overall_diss.argsort(dim=1)[:,1:self.reliable_num+1]
        
    def vis_probability(self, outputs, targets, t=0.1):

        cls_prob = torch.exp((self.nir_vectors[targets] * outputs).sum(dim=1) / t)
        sum_prob = torch.exp(outputs.matmul(self.nir_vectors.T) / t).sum(dim=1)
        return cls_prob / sum_prob

    def nir_probability(self, outputs, targets, t=0.1):

        cls_prob = torch.exp((self.rgb_vectors[targets] * outputs).sum(dim=1) / t)
        sum_prob = torch.exp(outputs.matmul(self.rgb_vectors.T) / t).sum(dim=1)
        return cls_prob / sum_prob

    def vis_reliables_probability(self, outputs, targets, labels, t=0.1):

        mask = torch.eq(self.label[targets],labels).type(torch.LongTensor)


        mask = mask.cuda()
        cls_prob = torch.exp((self.nir_vectors[targets] * outputs).sum(dim=1) / t)
        sum_prob = torch.exp(outputs.matmul(self.nir_vectors.T) / t).sum(dim=1)

        return mask*torch.log(cls_prob / sum_prob)


    def nir_reliables_probability(self, outputs, targets, labels, t=0.1):

        mask = torch.eq(self.label2[targets], labels).type(torch.LongTensor)
        mask = mask.cuda()
        cls_prob = torch.exp((self.nir_vectors[targets] * outputs).sum(dim=1) / t)
        sum_prob = torch.exp(outputs.matmul(self.nir_vectors.T) / t).sum(dim=1)
        return mask*torch.log(cls_prob / sum_prob)

