dirs:
	mkdir -p data/CIFAR10
	mkdir -p ckpts/classifier ckpts/robust_classifier ckpts/angular ckpts/angular_classifier
	git clone https://github.com/YoongiKim/CIFAR-10-images data/CIFAR10/

requirements: dirs
	pip install -r requirements.txt
