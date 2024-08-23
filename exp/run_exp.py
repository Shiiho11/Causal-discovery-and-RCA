from bcgllm.exp import exp_bcgllm
from exp_auto_mpg import exp_auto_mpg
from exp_gaia import exp_gaia
from exp_sachs import exp_sachs

if __name__ == '__main__':
    exp_dict = dict()
    exp_dict['exp_bcgllm'] = exp_bcgllm
    exp_dict['exp_auto_mpg'] = exp_auto_mpg
    exp_dict['exp_gaia'] = exp_gaia
    exp_dict['exp_sachs'] = exp_sachs
    for name, func in exp_dict.items():
        print('[EXP]Run:', name)
        func()
        # try:
        #     print('[EXP]Run:', name)
        #     func()
        #     print('[EXP]Success:', name)
        # except Exception as e:
        #     print('ERROR:', e)
        #     print('[EXP]Fail:', name)
