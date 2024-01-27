class Options():
    """docstring for Options"""
    def __init__(self):
        pass

    def init(self, parser):        

        # global settings
        parser.add_argument('--batch_size', type=int, default=8, help='batch size')
        parser.add_argument('--nepoch', type=int, default=250, help='training epochs')
        parser.add_argument('--train_workers', type=int, default=4, help='train_dataloader workers')
        parser.add_argument('--eval_workers', type=int, default=4, help='eval_dataloader workers')
        parser.add_argument('--dataset', type=str, default ='Microscope')

        parser.add_argument('--optimizer', type=str, default ='adamw', help='optimizer for training')
        parser.add_argument('--lr_initial', type=float, default=0.0002, help='initial learning rate')
        parser.add_argument('--step_lr', type=int, default=50, help='step size for lr decay')
        parser.add_argument('--weight_decay', type=float, default=0.02, help='weight decay')
        parser.add_argument('--gpu', type=str, default='0,1', help='GPUs')
        parser.add_argument('--arch', type=str, default ='HoloFormer',  help='archtechture')
        parser.add_argument('--dd_in', type=int, default=2, help='dd_in') 

        # args for saving 
        parser.add_argument('--save_dir', type=str, default ='./logs/',  help='save dir')
        parser.add_argument('--checkpoint', type=int, default=50, help='checkpoint')
        parser.add_argument('--env', type=str, default ='_',  help='env')

        # args for Uformer
        parser.add_argument('--norm_layer', type=str, default ='nn.LayerNorm', help='normalize layer in transformer')
        parser.add_argument('--embed_dim', type=int, default=32, help='dim of emdeding features')
        parser.add_argument('--win_size', type=int, default=8, help='window size of self-attention')
        parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
        parser.add_argument('--token_mlp', type=str,default='mlp', help='ffn') 

        # args for vit
        parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
        parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
        parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
        parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
        parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
 
        # args for training
        parser.add_argument('--resume', action='store_true',default=False)
        parser.add_argument('--data_dir', type=str, default ='/home/zx/Desktop/bzq_exp/lensless/dataset_patch',  help='dir of train and val data')
        parser.add_argument('--warmup', action='store_true', default=False, help='warmup') 
        parser.add_argument('--warmup_epochs', type=int,default=3, help='epochs for warmup') 

        # contrast loss
        parser.add_argument('--is_ab', type=bool, default=False, help='whether ablation study of contrast loss')
        parser.add_argument('--w_loss_1st', type=float, default=1, help='weight of 1st loss')
        parser.add_argument('--w_loss_contrast', type=float, default=0.01, help='weight of contrast loss')

        return parser

