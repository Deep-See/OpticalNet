import torch

# SR : Segmentation Result
# GT : Ground Truth
threshold = 0.5


def get_accuracy(SR, GT, threshold=threshold):
    SR = SR > threshold
    GT = GT == 1
    corr = torch.sum(SR == GT)
    # tensor_size = SR.size(0) * SR.size(1) * SR.size(2) * SR.size(3)
    tensor_size = SR.numel()
    acc = float(corr) / float(tensor_size)

    return acc

def get_sensitivity_no_threshold(SR, GT):
    GT = GT == torch.max(GT)
    SR = SR.cpu().numpy().astype(int)
    GT = GT.cpu().numpy().astype(int)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR == 1).astype(int) + (GT == 1).astype(int)) == 2
    FN = ((SR == 0).astype(int) + (GT == 1).astype(int)) == 2

    TP = TP.astype(int)
    FN = FN.astype(int)

    # SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    SE = float(TP.sum()) / (float((TP.sum() + FN.sum())) + 1e-6)
    if SE > 100:
        print(f'TP: {TP.sum()}  FN: {FN.sum()} SE: {SE}')

    return SE


def get_sensitivity(SR, GT, threshold=threshold):
    # Sensitivity == Recall
    SR = SR > threshold
    GT = GT == 1
    SR = SR.cpu().numpy().astype(int)
    GT = GT.cpu().numpy().astype(int)

    # TP : True Positive
    # FN : False Negative
    TP = ((SR == 1).astype(int) + (GT == 1).astype(int)) == 2
    FN = ((SR == 0).astype(int) + (GT == 1).astype(int)) == 2

    TP = TP.astype(int)
    FN = FN.astype(int)

    # SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    SE = float(TP.sum()) / (float((TP.sum() + FN.sum())) + 1e-6)
    if SE > 100:
        print(f'TP: {TP.sum()}  FN: {FN.sum()} SE: {SE}')

    return SE


def get_specificity(SR, GT, threshold=threshold):
    SR = SR > threshold
    GT = GT == 1
    SR = SR.cpu().numpy().astype(int)
    GT = GT.cpu().numpy().astype(int)

    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0).astype(int) + (GT == 0).astype(int)) == 2
    FP = ((SR == 1).astype(int) + (GT == 0).astype(int)) == 2

    TN = TN.astype(int)
    FP = FP.astype(int)

    # SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    SP = float(TN.sum()) / (float((TN.sum() + FP.sum())) + 1e-6)

    return SP


def get_precision(SR, GT, threshold=threshold):
    SR = SR > threshold
    GT = GT == 1
    SR = SR.cpu().numpy().astype(int)
    GT = GT.cpu().numpy().astype(int)

    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1).astype(int) + (GT == 1).astype(int)) == 2
    FP = ((SR == 1).astype(int) + (GT == 0).astype(int)) == 2

    TP = TP.astype(int)
    FP = FP.astype(int)

    PC = float(TP.sum()) / (float((TP.sum() + FP.sum())) + 1e-6)

    return PC


def get_F1(SR, GT, threshold=threshold):
    # Sensitivity == Recall
    SE = get_sensitivity(SR, GT, threshold=threshold)
    PC = get_precision(SR, GT, threshold=threshold)

    F1 = 2 * SE * PC / (SE + PC + 1e-6)

    return F1


def get_JS(SR, GT, threshold=threshold):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT == 1
    SR = SR.cpu().numpy().astype(int)
    GT = GT.cpu().numpy().astype(int)

    # Inter = torch.sum((SR+GT)==2)
    # Union = torch.sum((SR+GT)>=1)
    Inter = (((SR + GT) == 2).astype(int)).sum()
    Union = (((SR + GT) >= 1).astype(int)).sum()

    JS = float(Inter) / (float(Union) + 1e-6)

    return JS


def get_DC(SR, GT, threshold=threshold):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT == 1
    SR = SR.cpu().numpy().astype(int)
    GT = GT.cpu().numpy().astype(int)

    # Inter = torch.sum((SR+GT)==2)
    # DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)
    Inter = (((SR + GT) == 2).astype(int)).sum()
    DC = float(2 * Inter) / (float(SR.sum() + GT.sum()) + 1e-6)
    return DC
