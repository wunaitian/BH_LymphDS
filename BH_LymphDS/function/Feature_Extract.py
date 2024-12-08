import os
from BH_LymphDS.custom.comp import extract, print_feature_hook, reg_hook_on_module, init_from_model


def feature_extract(model_name, data, model_path):
    model_name = model_name
    model_path = model_path
    mydir = data
    if model_name == 'SwinTransformer':
        feature_name = 'avgpool'
    if model_name == 'ConvolutionalVisionTransformer':
        feature_name = 'norm'

    model, transformer, device = init_from_model(model_path)

    save_path = r'.\Output\Patch_Features'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for entry in os.listdir(mydir):
        full_path = os.path.join(mydir, entry)
        directory = os.path.expanduser(full_path)
        test_samples = [os.path.join(directory, p) for p in os.listdir(directory) if
                        p.endswith('.png') or p.endswith('.jpg')]
        if os.path.isdir(full_path):
            csv_filename = f"{entry}.csv"
            csv_path = os.path.join(save_path, csv_filename)
            outfile = open(csv_path, 'w')
            try:
                hook = lambda module, inp, outp: print_feature_hook(module, inp, outp, outfile)
                print(hook)
                # hook = partial(print_feature_hook, fp=outfile)
                hook_handles = reg_hook_on_module(feature_name, model, hook)
                # print(find_num)
                results = extract(test_samples, model, transformer, device, fp=outfile)
                # print(results)


            finally:
                for handle in hook_handles:
                    handle.remove()
                outfile.close()







