import time
from functools import partial

import torch
import torch.nn as nn

from project.blocks import Transformer
from project.data import data_preparation
from project.hyperparameters import *
from project.train import test, train
from project.utils import EarlyStopping, freeze

torch.manual_seed(0)
train_loader, test_loader = data_preparation(DATA_DIR, BATCH_SIZE_TRAIN, BATCH_SIZE_TEST)

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError or ModuleNotFoundError:
    has_apex = False
has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass


class VTR(nn.Module):
    """
    The last stage of ResNet101 contains 3 bottleneck blocks. At the end of the network, we output 16 visual tokens
    to the classification head, which applies an average pooling over the tokens and use a fully-connected layer to
    predict the probability We replace them with the same number 3 of VT modules. At the end of stage-4 (before
    stage-5 max pooling)

        ResNet-{101} generate 14x14 × 1024.
        We set VT’s feature map channel size 1024
        VT block with a channel size for the output feature map as 1024, channel size for visual tokens as 1024,
        and the number of tokens as 16.
        We adopt 16 visual tokens with a channel size of 1024. Only train the transformer and the FC layers.
    """

    def __init__(self, img_size=IMAGE_SIZE, in_chans=CHANNELS, num_classes=NUM_CLASSES,
                 dim=VTR_DIM, depth=TRANSFORMER_DEPTH,
                 num_heads=TRANSFORMER_HEADS, mlp_hidden_dim=VTR_MLP_DIM,
                 dropout=VTR_DROPOUT, attn_dropout=ATTN_DROPOUT, emb_dropout=EMB_DROPOUT, backbone=BACKBONE):
        """img_size: input image size
            in_chans: number of input channels
            num_classes: number of classes for classification head
            dim :embedding dimension
            depth: depth of transformer
            num_heads : number of attention heads
            mlp_dim : mlp hidden dimension
            drop_rate : dropout rate
            attn_drop_rate : attention dropout rate
            drop_path_rate : stochastic depth rate
            backbone (nn.Module): CNN backbone to use in-place of PatchEmbed module
        """
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.img_size = (img_size, img_size)
        self.in_chans = in_chans
        backbone = nn.Sequential(*list(backbone.children())[:-3])
        freeze(module=backbone, train_bn=False)
        self.backbone = backbone
        feature_map = backbone(torch.zeros(1, in_chans, img_size, img_size))
        feature_size = feature_map.shape[-2:]
        feature_dim = feature_map.shape[1]
        num_patches = feature_size[0] * feature_size[1]
        self.patch_embed = nn.Conv2d(feature_dim, dim, 1).cuda()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))

        self.emb_dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim=dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim,
                                       dropout=dropout, attn_dropout=attn_dropout, depth=depth)
        self.norm = norm_layer(dim)
        self.fc = nn.Linear(dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.backbone(x)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        x = self.emb_dropout(x)
        x = self.transformer(x)
        x = self.norm(x)[:, 0]
        return self.fc(x)


if torch.cuda.is_available():
    use_cuda = True
    model = VTR().cuda()
else:
    model = VTR()
    use_cuda = False

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=LEARNING_RATE, eps=EPSILON, weight_decay=WEIGHT_DECAY)
if DECAY_TYPE == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_PER, gamma=GAMMA)
elif DECAY_TYPE == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=COS_T0, T_mult=COS_T_MULT)
early_stopping = EarlyStopping(patience=EARLY_STOP_PATIENCE, verbose=True, path=CHECKPOINT_PATH)
if has_apex:
    model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

# from torchsummary import summary
# summary(model, input_size=(3, 224, 224)


def main():
    total_time_left = 'Unknown'
    train_loss_history, test_loss_history, eval_loss_history = [], [], []
    for epoch in range(1, N_EPOCHS + 1):
        if epoch == 1:
            try:
                checkpoint = torch.load(CHECKPOINT_PATH)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch_c = checkpoint['epoch']
                train_loss_history = list([checkpoint['loss']])
                test_loss_history = list([checkpoint['val_loss']])
                print(f'Checkpoint loaded, continuing training from EPOCH {epoch + epoch_c}')
                cpload = checkpoint['val_loss']
            except FileNotFoundError:
                print('No checkpoint found, training from scratch')
                epoch_c = 0
                cpload = False
        current_learning_rate = optimizer.param_groups[0]['lr']
        if epoch == 1:
            print(
                f'EPOCH: [{epoch + epoch_c}/{N_EPOCHS}], Learning Rate: [{current_learning_rate}], Steps per epoch: [{len(train_loader.dataset) // BATCH_SIZE_TRAIN}]')
        else:
            print(
                f'EPOCH: [{epoch + epoch_c}/{N_EPOCHS}], Learning Rate: [{current_learning_rate}], Total time left: {total_time_left if type(total_time_left) != str else str(total_time_left)} minutes')
        start_time = time.time()
        train(model, optimizer, train_loader, train_loss_history, use_cuda=use_cuda, use_fp16=True)
        print('\n')
        test(model, test_loader, test_loss_history, use_cuda=use_cuda)
        early_stopping(val_loss=test_loss_history[-1], model=model, loss=train_loss_history, optimizer=optimizer,
                       epoch=epoch + epoch_c, cpload=cpload)
        cpload = False
        total_time_left = ((time.time() - start_time) * (N_EPOCHS - epoch - epoch_c)) // 60
        if early_stopping.early_stop:
            print('Early stopping')
            break
        scheduler.step()
        print(f'\n{(time.time() - start_time):5.2f} second(s) / EPOCH')


if __name__ == '__main__':
    main()
