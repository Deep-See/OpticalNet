from scipy.stats import pearsonr

def pearson_correlation(gt, raw_pred):
    '''
    gt: ground truth
    raw_pred: raw prediction
    in (1,1,H,W) format
    '''
    gt = gt.squeeze(0).squeeze(0).cpu().numpy()  # (H,W)
    raw_pred = raw_pred.squeeze(0).squeeze(0).cpu().numpy()  # (H,W)
    gt = gt.flatten()
    raw_pred = raw_pred.flatten()
    return pearsonr(gt, raw_pred)[0]
