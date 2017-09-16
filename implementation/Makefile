SUMMARY_DIR=graphs/crowdnet

train:
	nohup python -m tensorflow.tensorboard --logdir=$(SUMMARY_DIR) > tensorboard.log &
	python crowdnet_train.py --summary_dir=$(SUMMARY_DIR)

