import os
import numpy as np
import torchvision
from torch import optim
import torch.nn.functional as F
from evaluation import *
from network import U_Net, R2U_Net, AttU_Net, R2AttU_Net, AE_Net, Encoder_Net, Transformer_UNet
from tqdm import tqdm
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter
from comp_pearson_corr import pearson_correlation
from Basic_Unet import UNet, CustomResNet34, CustomResNet18
# import wandb
import torch.nn as nn
from pytorch_msssim import SSIM
from torch.optim.lr_scheduler import CosineAnnealingLR


def f_beta_loss(preds, labels, beta=1, threshold=0.5):
    epsilon = 1e-7  # used to prevent division by zero
    preds = (preds > threshold).float()

    true_positives = torch.sum(preds * labels)
    false_positives = torch.sum(preds * (1 - labels))
    false_negatives = torch.sum((1 - preds) * labels)

    precision = true_positives / (true_positives + false_positives + epsilon)
    recall = true_positives / (true_positives + false_negatives + epsilon)

    beta_squared = beta ** 2
    f_beta_score = (1 + beta_squared) * (precision * recall) / \
                   ((beta_squared * precision) + recall + epsilon)

    f_beta_loss = 1. / (f_beta_score.mean() + epsilon)

    return f_beta_loss


def concat_images(image_paths, output_path):
    images = [Image.open(x) for x in image_paths]
    width, height = images[0].size
    total_width = width * len(images)
    new_image = Image.new('RGBA', (total_width, height))
    x_offset = 0

    for image in images:
        new_image.paste(image, (x_offset, 0))
        x_offset += width
    new_image.save(output_path)


class Solver(object):
    def __init__(self, config, train_loader, valid_loader, test_loader):

        self.config = config
        # Data loader
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        # Models
        self.unet = None
        self.optimizer = None
        self.img_ch = config.img_ch
        self.output_ch = config.output_ch
        self.criterion = torch.nn.BCELoss()
        self.augmentation_prob = config.augmentation_prob

        # Hyper-parameters
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        self.focus_beta = config.focus_beta

        # Training settings
        self.num_epochs = config.num_epochs
        self.num_epochs_decay = config.num_epochs_decay
        self.batch_size = config.batch_size

        # Step size
        self.log_step = config.log_step
        self.val_step = config.val_step

        # Path
        self.model_path = config.model_path + config.special_save_folder_name
        self.result_path = config.result_path
        self.mode = config.mode

        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device(f'cuda:{config.cuda_idx}')
        self.model_type = config.model_type
        self.t = config.t
        self.build_model()

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.num_epochs, eta_min=1e-6)

    def build_model(self):
        """Build generator and discriminator."""
        if self.model_type == 'U_Net':
            # self.unet = U_Net(img_ch=3, output_ch=self.config.output_ch)
            self.unet = U_Net(img_ch=self.config.img_ch, output_ch=self.config.output_ch)

            # ===================== try double ====================================== #
            # self.unet = self.unet.double()
            # ====================================================================== #
        elif self.model_type == 'AE_Net':
            self.unet = AE_Net(img_ch=self.config.img_ch, output_ch=self.config.output_ch)

        elif self.model_type == 'Encoder_Net':
            self.unet = Encoder_Net(img_ch=self.config.img_ch, output_ch=self.config.output_ch)

        elif self.model_type == 'Transformer_UNet':
            self.unet = Transformer_UNet(L=self.config.L, dim=self.config.transformer_dim,
                                         encoder_only=self.config.encoder_only,
                                         output_hw=self.config.output_hw)

        elif self.model_type == 'UNet':
            self.unet = UNet(img_ch=self.config.img_ch, output_ch=self.config.output_ch)
            self.unet.initialize_weights()

        elif self.model_type == 'CustomResNet34':
            self.unet = CustomResNet34()

        elif self.model_type == 'CustomResNet18':
            self.unet = CustomResNet18()

        elif self.model_type == 'AE_Net_step1':
            self.unet = AE_Net(img_ch=3, output_ch=self.config.output_ch)
            self.unet.load_state_dict(torch.load(
                '/home/benquan/AE_Net_step1-250-0.0000-114-0.3052.pkl'))
        elif self.model_type == 'AE_Net_step2':
            self.unet = AE_Net(img_ch=3, output_ch=self.config.output_ch)
            self.unet_mask = AE_Net(img_ch=3, output_ch=self.config.output_ch)
            self.unet_mask.load_state_dict(torch.load(
                '/home/benquan/AE_Net_step1-250-0.0000-114-0.3052.pkl'))
            self.unet_mask.to(self.device)
        elif self.model_type == 'R2U_Net':
            self.unet = R2U_Net(img_ch=3, output_ch=self.config.output_ch, t=self.t)
        elif self.model_type == 'AttU_Net':
            self.unet = AttU_Net(img_ch=3, output_ch=self.config.output_ch)
        elif self.model_type == 'R2AttU_Net':
            self.unet = R2AttU_Net(img_ch=3, output_ch=self.config.output_ch, t=self.t)

        self.optimizer = optim.Adam(list(self.unet.parameters()),
                                    self.lr, [self.beta1, self.beta2], weight_decay=self.config.wd)
        self.unet.to(self.device)

    # self.print_network(self.unet, self.model_type)

    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()
        print(model)
        print(name)
        print("The number of parameters: {}".format(num_params))

    def to_data(self, x):
        """Convert variable to tensor."""
        if torch.cuda.is_available():
            x = x.cpu()
        return x.data

    def update_lr(self, g_lr, d_lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def reset_grad(self):
        """Zero the gradient buffers."""
        self.unet.zero_grad()

    def compute_accuracy(self, SR, GT):
        SR_flat = SR.view(-1)
        GT_flat = GT.view(-1)

        acc = GT_flat.data.cpu() == (SR_flat.data.cpu() > 0.5)

    def tensor2img(self, x):
        img = (x[:, 0, :, :] > x[:, 1, :, :]).float()
        img = img * 255
        return img

    def train(self):
        # ====================================== Training ===========================================#
        # ===========================================================================================#

        special_save_name = self.config.special_save_name
        unet_path = os.path.join(self.model_path,
                                 f'{self.model_type}_epoch_{self.num_epochs}_lr_{self.lr}_'
                                 f'focus_weight_{self.config.focus_weight}_{self.config.image_type}_{special_save_name}.pkl')

        if not os.path.exists(os.path.join(self.model_path, 'train_valid_records')):
            os.makedirs(os.path.join(self.model_path, 'train_valid_records'))
        writer = SummaryWriter(os.path.join(self.model_path, 'train_valid_records',
                                            f'{self.model_type}_epoch_{self.num_epochs}_lr_{self.lr}_'
                                            f'focus_weight_{self.config.focus_weight}_{self.config.image_type}_{special_save_name}'))

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))
            print('No training is executed.')
        else:
            # Train for Encoder
            lr = self.lr
            best_unet_score = 0.

            for epoch in range(self.num_epochs):

                self.unet.train(True)
                epoch_loss = 0
                epoch_focus_loss = 0

                acc = 0.  # Accuracy
                SE = 0.  # Sensitivity (Recall)
                SP = 0.  # Specificity
                PC = 0.  # Precision
                F1 = 0.  # F1 Score
                JS = 0.  # Jaccard Similarity
                DC = 0.  # Dice Coefficient
                length = 0
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Starting epoch {epoch + 1} with learning rate: {current_lr}')

                for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.train_loader)):
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = self.unet(images)

                    SR_probs = F.sigmoid(SR)
                    SR_flat = SR_probs.view(SR_probs.size(0),
                                            -1)

                    GT_flat = GT.view(GT.size(0), -1)
                    loss = self.criterion(SR_flat, GT_flat)  # base loss
                    epoch_loss += loss.item()
                    # epoch_focus_loss += focus_loss.item()

                    # Backprop + optimize
                    self.reset_grad()
                    loss.backward()
                    self.optimizer.step()

                    acc += get_accuracy(SR_probs, GT)
                    SE += get_sensitivity(SR_probs, GT)
                    SP += get_specificity(SR_probs, GT)
                    PC += get_precision(SR_probs, GT)
                    F1 += get_F1(SR_probs, GT)
                    JS += get_JS(SR_probs, GT)
                    DC += get_DC(SR_probs, GT)
                    # length += images.size(0)
                    length += 1

                # break

                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length

                # ===================  log in wandb ============ #
                total_loss = epoch_loss / length
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f'Finished epoch {epoch + 1}, new learning rate: {current_lr}')


                # Print the log info
                print(
                    'Epoch [%d/%d], Total_Loss: %.4f, Focus_loss: %.4f, \n[Training] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                        epoch + 1, self.num_epochs, \
                        epoch_loss / length, epoch_focus_loss / length, \
                        acc, SE, SP, PC, F1, JS, DC))

                writer.add_scalar('Train_epoch_loss', epoch_loss / length, epoch)
                writer.add_scalar('Train_SE', SE, epoch)

                # ===================================== Validation ====================================#
                with torch.no_grad():
                    self.unet.train(False)
                    self.unet.eval()

                    for module in self.unet.modules():
                        if isinstance(module, nn.Dropout):
                            assert module.p == 0 or not module.training, "Dropout should be disabled in eval mode"

                    acc = 0.  # Accuracy
                    SE = 0.  # Sensitivity (Recall)
                    SP = 0.  # Specificity
                    PC = 0.  # Precision
                    F1 = 0.  # F1 Score
                    JS = 0.  # Jaccard Similarity
                    DC = 0.  # Dice Coefficient
                    length = 0
                    for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.valid_loader)):
                        images = images.to(self.device)
                        GT = GT.to(self.device)
                        SR = F.sigmoid(self.unet(images))
                        acc += get_accuracy(SR, GT)
                        SE += get_sensitivity(SR, GT)
                        SP += get_specificity(SR, GT)
                        PC += get_precision(SR, GT)
                        F1 += get_F1(SR, GT)
                        JS += get_JS(SR, GT)
                        DC += get_DC(SR, GT)

                        # length += images.size(0)
                        length += 1
                    # break

                acc = acc / length
                SE = SE / length
                SP = SP / length
                PC = PC / length
                F1 = F1 / length
                JS = JS / length
                DC = DC / length
                # unet_score = JS + DC
                # unet_score = acc
                unet_score = SE

                print(
                    '[Validation] Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f' % (
                        acc, SE, SP, PC, F1, JS, DC))

                # writer.add_scalar('Valid_acc', acc, epoch)
                writer.add_scalar('Valid_SE', SE, epoch)

                # use threshold to preprocess SR img
                threshold_075, threshold_050, threshold_030, threshold_010 = 0.75, 0.5, 0.3, 0.1

                SR_075 = torch.where(SR > threshold_075, 1., 0.)
                SR_050 = torch.where(SR > threshold_050, 1., 0.)
                SR_030 = torch.where(SR > threshold_030, 1., 0.)
                SR_010 = torch.where(SR > threshold_010, 1., 0.)

                # Save Best U-Net model
                if unet_score >= best_unet_score:
                    best_unet_score = unet_score
                    best_epoch = epoch
                    best_unet = self.unet.state_dict()
                    print('Best %s model score : %.4f' % (self.model_type, best_unet_score))
                    # unet_path = os.path.join(self.model_path,
                    #                          f'{self.model_type}_epoch_{self.num_epochs}_lr_{self.lr}_'
                    #                          f'focus_weight_{self.config.focus_weight}_{self.config.image_type}_{special_save_name}.pkl')
                    # add in epoch name in the unet path
                    unet_path = os.path.join(self.model_path,
                                             f'{self.model_type}_epoch_{epoch + 1}_lr_{self.lr}_'
                                             f'focus_weight_{self.config.focus_weight}_{self.config.image_type}_{special_save_name}.pkl')
                    torch.save(best_unet, unet_path)
        writer.close()

    def generate_test_result(self):
        """Train encoder, generator and discriminator."""

        # ====================================== Training ===========================================#
        # ===========================================================================================#

        unet_path = os.path.join(self.model_path, '%s-%d-%.4f-%d-%.4f.pkl' % (
            self.model_type, self.num_epochs, self.lr, self.num_epochs_decay,
            self.augmentation_prob))

        # U-Net Train
        if os.path.isfile(unet_path):
            # Load the pretrained Encoder
            self.unet.load_state_dict(torch.load(unet_path))
            print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))

            # =================================== generate test results one by one ==================================#
            with torch.no_grad():
                self.unet.train(False)
                self.unet.eval()

                for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.valid_loader)):
                    images = images.to(self.device)
                    GT = GT.to(self.device)
                    SR = F.sigmoid(self.unet(images))

                    # # use threshold to preprocess SR img
                    SR_050 = torch.where(SR > 0.5, 1., 0.)

                    torchvision.utils.save_image(SR_050.data.cpu(),
                                                 os.path.join(self.result_path,
                                                              f'{gt_paths[17:-4]}_test.tif'))

    def tensor_to_3x3_image(self, tensor):
        # Ensure the tensor is on CPU and detached from the computation graph
        tensor = tensor.cpu().detach()

        # Reshape the tensor to 3x3
        hw = int(np.sqrt(tensor.shape[-1]))
        img_array = tensor.view(hw, hw).numpy() * 255

        img_array = img_array.astype(np.uint8)

        # Create a PIL Image
        img = Image.fromarray(img_array, mode='L')  # 'L' mode for grayscale

        # Resize for better visibility (optional)
        img = img.resize((64, 64), Image.NEAREST)

        return img

    def test(self, pretrain_path=None):
        unet_path = pretrain_path
        self.test_result_path = os.path.join(self.result_path,
                                             f'test_result_{self.config.selected_test_fold[0]}')

        print(
            f'test on fold {self.config.selected_test_fold[0]} and save results to {self.test_result_path}')
        if not os.path.exists(self.test_result_path):
            os.makedirs(self.test_result_path)

        self.unet.load_state_dict(torch.load(unet_path))
        print(f'{self.model_type} is Successfully Loaded from {unet_path}')
        # print('%s is Successfully Loaded from %s' % (self.model_type, unet_path))

        # ===================================== Testing ====================================#
        with torch.no_grad():
            self.unet.train(False)
            self.unet.eval()

            acc = 0.  # Accuracy
            SE = 0.  # Sensitivity (Recall)
            SP = 0.  # Specificity
            PC = 0.  # Precision
            F1 = 0.  # F1 Score
            JS = 0.  # Jaccard Similarity
            DC = 0.  # Dice Coefficient
            epoch_loss = 0.
            auc_roc = 0.
            length = 0
            CR = 0.  # Pearson correlation

            accs = []
            SEs = []
            PCs = []
            F1s = []
            auc_rocs = []
            CRs = []

            # create a dictionary to save the results of this subfolder
            correlation_result_dict = {
                '2 nanoholes': [],
                '3 nanoholes': [],
                '4 nanoholes': [],
                '5 nanoholes': [],
                '6 nanoholes': [],
                '7 nanoholes': [],
                '8 nanoholes': [],
                '9 nanoholes': [],
                '10 nanoholes': [],
            }

            for i, (images, GT, image_path, GT_path) in enumerate(tqdm(self.test_loader)):
                images = images.to(self.device)
                GT = GT.to(self.device)
                predict_image = self.unet(images)
                # predict_image_flatten = predict_image.flatten()

                SR = torch.sigmoid(predict_image)

                SR_flat = SR.view(SR.size(0), -1)

                GT_flat = GT.view(GT.size(0), -1)
                loss = self.criterion(SR_flat, GT_flat)
                epoch_loss += loss.item()

                point_acc = get_accuracy(SR, GT)
                point_SE = get_sensitivity(SR, GT)
                point_SP = get_specificity(SR, GT)
                point_PC = get_precision(SR, GT)
                point_F1 = get_F1(SR, GT)
                point_JS = get_JS(SR, GT)
                point_DC = get_DC(SR, GT)
                # GT_binary = np.round(GT.flatten().cpu().numpy())
                # point_auc_roc = roc_auc_score(y_true=GT_binary.flatten(),
                #                               y_score=SR.flatten().cpu().numpy())
                # point_CR = pearson_correlation(GT_flat, SR_flat)

                acc += point_acc
                SE += point_SE
                SP += point_SP
                PC += point_PC
                F1 += point_F1
                JS += point_JS
                DC += point_DC
                # auc_roc += point_auc_roc
                # CR += point_CR

                accs.append(point_acc)
                SEs.append(point_SE)
                PCs.append(point_PC)
                F1s.append(point_F1)
                # auc_rocs.append(point_auc_roc)
                # CRs.append(point_CR)

                # length += images.size(0)
                length += 1

                # use threshold to preprocess SR img
                threshold_075, threshold_050, threshold_030, threshold_010 = 0.75, 0.5, 0.3, 0.1

                SR_075 = torch.where(SR > threshold_075, 1., 0.)
                SR_050 = torch.where(SR > threshold_050, 1., 0.)
                SR_030 = torch.where(SR > threshold_030, 1., 0.)
                SR_010 = torch.where(SR > threshold_010, 1., 0.)

                output_list = (SR_075, SR_050, SR_030, SR_010)
                output_SEs = []

                self.tensor_to_3x3_image(SR).save(os.path.join(self.test_result_path,
                                                               'test_%s_idx_%d_SR.png' % (
                                                                   self.model_type, i)))
                self.tensor_to_3x3_image(SR_075).save(os.path.join(self.test_result_path,
                                                                   'test_%s_idx_%d_SR_075.png' % (
                                                                       self.model_type, i)))
                self.tensor_to_3x3_image(SR_050).save(os.path.join(self.test_result_path,
                                                                   'test_%s_idx_%d_SR_050.png' % (
                                                                       self.model_type, i)))
                self.tensor_to_3x3_image(SR_030).save(os.path.join(self.test_result_path,
                                                                   'test_%s_idx_%d_SR_030.png' % (
                                                                       self.model_type, i)))
                self.tensor_to_3x3_image(SR_010).save(os.path.join(self.test_result_path,
                                                                   'test_%s_idx_%d_SR_010.png' % (
                                                                       self.model_type, i)))
                self.tensor_to_3x3_image(GT).save(os.path.join(self.test_result_path,
                                                               'test_%s_idx_%d_GT.png' % (
                                                                   self.model_type, i)))
                # iterate over all the output images and compute the SE
                for output in output_list:
                    se = get_sensitivity_no_threshold(output, GT)
                    output_SEs.append(se)
                image = cv2.imread(os.path.join(self.test_result_path,
                                                'test_%s_idx_%d_SR_050.png' % (self.model_type, i)))
                image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image_gray = cv2.resize(image_gray, (512, 512))
                gt = cv2.imread(GT_path[0])
                gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)

        acc = acc / length
        SE = SE / length
        SP = SP / length
        PC = PC / length
        F1 = F1 / length
        JS = JS / length
        DC = DC / length
        epoch_loss = epoch_loss / length
        auc_roc = auc_roc / length
        CR = CR / length
        # unet_score = JS + DC
        unet_score = acc

        print(
            '[Testing] BCE loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, AUROC: %.4f, Pearson: %.4f' % (
                epoch_loss, acc, SE, SP, PC, F1, JS, DC, auc_roc, CR,))

        with open(os.path.join(self.test_result_path, 'metric_result_record.txt'), 'w') as f:
            f.write(
                '[Testing] BCE loss: %.4f, Acc: %.4f, SE: %.4f, SP: %.4f, PC: %.4f, F1: %.4f, JS: %.4f, DC: %.4f, auroc: %.4f, Pearson: %.4f' % (
                    epoch_loss, acc, SE, SP, PC, F1, JS, DC, auc_roc, CR))
