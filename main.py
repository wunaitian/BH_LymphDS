import os
import argparse
from collections import namedtuple
from BH_LymphDS.function.run_classification import main as clf_train
from BH_LymphDS.function.run_predict import main as clf_predict
from BH_LymphDS.function.Feature_Extract import feature_extract
from BH_LymphDS.function.GradCam import swin_gradcam


def train(model_name, data, label_dir, model_path):
    data_pattern = os.path.join(data, 'images')
    train_f = os.path.join(label_dir, 'train.txt')
    val_f = os.path.join(label_dir, 'val.txt')
    labels_f = os.path.join(label_dir, 'labels.txt')
    weights = os.path.join(model_path, 'BH_LymphDS.pth')

    params = dict(train=train_f,
                  valid=val_f,
                  labels_file=labels_f,
                  data_pattern=data_pattern,
                  j=4,
                  batch_balance=False,
                  normalize_method='imagenet',
                  model_name=model_name,
                  gpus=[0],
                  batch_size=1,
                  epochs=1,
                  init_lr=0.01,
                  optimizer='sgd',
                  model_root=r'.\Output\Train',
                  iters_start=0,
                  iters_verbose=1,
                  save_per_epoch=False,
                  weights=weights)
    # 训练模型
    Args = namedtuple("Args", params)
    clf_train(Args(**params))


def predict(model_name, data, label_dir, model_path):
    data_pattern = os.path.join(data, 'images')
    test_f = os.path.join(label_dir, 'test.txt')
    labels_f = os.path.join(label_dir, 'labels.txt')
    weights = os.path.join(model_path, 'BH_LymphDS.pth')


    params = dict(test=test_f,
                  labels_file=labels_f,
                  data_pattern=data_pattern,
                  class_num=4,
                  j=4,
                  batch_balance=False,
                  normalize_method='imagenet',
                  model_name=model_name,
                  gpus=[0],
                  batch_size=1,
                  epochs=1,
                  init_lr=0.01,
                  optimizer='sgd',
                  retrain=None,
                  model_root=r'.\Output\Patch_Predict',
                  iters_start=0,
                  iters_verbose=1,
                  save_per_epoch=False,
                  weights=weights)

    # 训练模型
    Args = namedtuple("Args", params)
    clf_predict(Args(**params))


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurations for BH_LymphDS')
    parser.add_argument('--train', action='store_true', help='Use pretrained core or not')
    parser.add_argument('--predict', action='store_true', help='Use pretrained core or not')
    parser.add_argument('--feature_extract', action='store_true', help='Use pretrained core or not')
    parser.add_argument('--gradcam', action='store_true', help='Use pretrained core or not')
    parser.add_argument('--model_name', type=str, default='SwinTransformer', help='DL backbone model')
    parser.add_argument('--image_data', type=str, default='.\Data\input_images', help='image data')
    parser.add_argument('--label_dir', type=str, default='.\Data\Label_Files', help='path point to label files')
    parser.add_argument('--model_path', type=str, default='.\Data\pretrained_Weight', help='pretrained model weight')
    args = parser.parse_args()

    if args.train:
        train(model_name=args.model_name, data=args.image_data, label_dir=args.label_dir, model_path=args.model_path)
    if args.predict:
        predict(model_name=args.model_name, data=args.image_data, label_dir=args.label_dir, model_path=args.model_path)
    if args.feature_extract:
        feature_extract(model_name=args.model_name, data=args.image_data, model_path=args.model_path)
    if args.gradcam:
        swin_gradcam(model_path=args.model_path, data=args.image_data)
