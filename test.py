import argparse, time, os
import imageio
import numpy as np
import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset


def main():
    parser = argparse.ArgumentParser(description='Test Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)
    opt = option.dict_to_nonedict(opt)

    # initial configure
    scale = [2 ** (i + 1) for i in range(int(np.log2(opt['scale'])))] if 'ms_test' in opt.keys() and opt['ms_test'] else [
        opt['scale']]
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()
    if opt['self_ensemble']: model_name += 'plus'

    # create test dataloader
    bm_names = []
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        print('===> Test Dataset: [%s]   Number of images: [%d]' % (test_set.name(), len(test_set)))
        bm_names.append(test_set.name())

    # create solver (and load model)
    solver = create_solver(opt)
    # Test phase
    print('===> Start Test')
    print("==================================================")
    print("Method: %s || Scale: %s || Degradation: %s" % (model_name, scale, degrad))

    for bm, test_loader in zip(bm_names, test_loaders):
        print("Test set : [%s]" % bm)

        sr_list = [[] for _ in range(len(scale))]
        path_list = [[] for _ in range(len(scale))]

        total_psnr = [[] for _ in range(len(scale))]
        total_ssim = [[] for _ in range(len(scale))]
        total_time = []

        need_HR = False if test_loader.dataset.__class__.__name__.find('HR') < 0 else True

        for iter, batch in enumerate(test_loader):
            solver.feed_data(batch, need_HR=need_HR)

            # calculate forward time
            t0 = time.time()
            solver.test(scale)
            t1 = time.time()
            total_time.append((t1 - t0))

            visuals = solver.get_current_visual(need_HR=need_HR)

            for i in range(len(scale)):
                sr_list[i].append(visuals['SR'][i])

                # calculate PSNR/SSIM metrics on Python
                lr_path = batch['LR_path']
                if len(scale) > 1:
                    lr_path = batch['LR_path'][i]
                if need_HR:
                    psnr, ssim = util.calc_metrics(visuals['SR'][i], visuals['HR'][i], crop_border=scale[i])
                    total_psnr[i].append(psnr)
                    total_ssim[i].append(ssim)
                    hr_path = batch['HR_path']
                    if len(scale) > 1:
                        hr_path = batch['HR_path'][i]
                    path_list[i].append(os.path.basename(hr_path[0]).replace('HR', model_name))
                    print("[%d/%d] %s || PSNR(dB)/SSIM: %.3f/%.5f || Timer: %.4f sec ." % (iter + 1, len(test_loader),
                                                                                           os.path.basename(lr_path[0]),
                                                                                           psnr, ssim,
                                                                                           (t1 - t0)))
                else:
                    path_list[i].append(os.path.basename(lr_path[0]))
                    print("[%d/%d] %s || Timer: %.4f sec ." % (iter + 1, len(test_loader),
                                                               os.path.basename(lr_path[0]),
                                                               (t1 - t0)))

        print("---- Average Speed(s) for [%s] x%s is %.4f sec ----" % (bm, scale, sum(total_time) / len(total_time)))
        for i in range(len(scale)):
            if need_HR:
                print("---- Average PSNR(dB) /SSIM for [%s] x%d ----" % (bm, scale[i]))
                print("PSNR: %.3f      SSIM: %.5f" % (sum(total_psnr[i]) / len(total_psnr[i]),
                                                      sum(total_ssim[i]) / len(total_ssim[i])))

            # save SR results for further evaluation on MATLAB
            if need_HR:
                save_img_path = os.path.join('./results/SR/' + degrad, model_name, bm, "x%d" % scale[i])
            else:
                save_img_path = os.path.join('./results/SR/' + bm, model_name, "x%d" % scale[i])

            print("===> Saving SR images of [%s]... Save Path: [%s]\n" % (bm, save_img_path))

            if not os.path.exists(save_img_path): os.makedirs(save_img_path)
            for img, name in zip(sr_list[i], path_list[i]):
                imageio.imwrite(os.path.join(save_img_path, name), img)

    print("==================================================")
    print("===> Finished !")


if __name__ == '__main__':
    main()
