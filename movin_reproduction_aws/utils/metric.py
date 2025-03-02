import torch

def calc_MPJPE(pred_or, gt_or): # input unit: meter (m) 
    '''
    pred: (N, K, 3) 
    gt: (N, K, 3)
    '''

    _pred = pred_or.detach().clone()
    _gt = gt_or.detach().clone()
    
    # print(pred_or[:, 0, :].unsqueeze(1).shape) # [B, 1, 3]
    pred = _pred - _pred[:, 0, :].unsqueeze(1)
    gt = _gt - _gt[:, 0, :].unsqueeze(1)
    # pred = pred_or.cpu().detach().numpy()
    # gt = gt_or.cpu().detach().numpy()

    assert(pred.shape == gt.shape)
    assert(len(pred.shape) == 3)

    # N, K = pred.shape[0], pred.shape[1] # [BS, J, 3]

    err_dist = torch.sqrt(torch.sum((pred - gt) ** 2, dim=2)) * 1000 # convert to mm unit
    err_dist = torch.mean(err_dist)
    return err_dist


def calc_mAP(pred_or, gt_or, dist=0.1): # unit: meter (m)
    '''
    pred: (N, K, 3) 
    gt: (N, K, 3)
    '''

    pred = pred_or.detach().clone()
    gt = gt_or.detach().clone()

    assert(pred.shape == gt.shape)
    assert(len(pred.shape) == 3)

    N, K = pred.shape[0], pred.shape[1] # [BS, J, 3]
    err_dist = torch.sqrt(torch.sum((pred - gt)**2, dim=2)) # (N, K)

    acc_d = (err_dist < dist).sum(dim=0) / N

    return torch.mean(acc_d) * 100 # conver to percent (%)
    2