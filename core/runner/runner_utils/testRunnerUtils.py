import torch
import time
from .utils import transform_input


def testmodel(model, loader, loggers, test_freq, testset_name, last_iter):  # last_iter: for logging use
    if 'save' in testset_name:
        print('skip dataset', testset_name)
        return 0., 0
    # must be val_mode
    all_error, n_count = 0., 0
    # model = model.train()
    if hasattr(model, 'initialize_error'):
        model.initialize_error()
    if hasattr(model, 'individual_tester'):
        print('Using Invidiual Tester', flush=True)
        with torch.no_grad():
            error = model.individual_tester(loader, transform_input, test_freq)
        return error, 1.
    for it, sample in enumerate(loader):
        sample = transform_input(sample)
        with torch.no_grad():  # no tracking
            output = model(sample)
            if it == len(loader) - 1 and hasattr(model, 'final_error'):
                output = model.final_error(sample, output)
            # mutli-batch; for data-parallel-model use
            if isinstance(model, torch.nn.DataParallel):
                for key, value in output.items():
                    if 'error' in key or 'n_count' == key:
                        output[key] = torch.sum(value, dim=0)
                    # print('error', key, value.shape, output[key].shape)
            # print('error sum', output['error'])
            if it == 0:
                output['testset_name_out'] = testset_name
            output['iteration'] = [it + 1, len(loader), (it + 1) / len(loader)]
            if it == len(loader) - 1:
                output['last_iter'] = last_iter
                output['flush'] = True
            loggers.update_error(output, it % test_freq == 0 or it == len(loader) - 1)
            all_error += output['error']
            n_count += output['n_count']
    # print('testing one dataset : DONE', all_error / n_count)
    return all_error / n_count, 1.
