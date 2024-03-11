python test.py configs/tusimple_efv2s.py --test_model /root/Ultra-Fast-Lane-Detection-v2/logs/20240311_000856_lr_5e-02_b_4/model_best.pth --test_work_dir ./tmp
python train.py configs/tusimple_regy3_2.py --log_path logs
python test.py configs/tusimple_efv2s.py --test_model /root/Ultra-Fast-Lane-Detection-v2/logs/20240311_000856_lr_5e-02_b_4/model_best.pth --test_work_dir ./tmp

python test.py configs/tusimple_regy3_2.py --test_model /root/Ultra-Fast-Lane-Detection-v2/logs/20240311_120313_lr_5e-02_b_8/model_best.pth --test_work_dir ./tmp

python train.py configs/tusimple_regy1_6.py --log_path logs