#encoding="utf-8"
import mnist_loader
import network
# �����ݼ���ֳ��������ϣ�ѵ������������֤�������Լ�
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# �������������������ṹΪ���㣬ÿ��ڵ�������Ϊ��784, 30, 10��
net = network.Network([784, 128, 10])
# �ã�mini-batch���ݶ��½���ѵ�������磨Ȩ����ƫ�ƣ��������ɲ��Խ����
# ѵ���غ���=30, ��������ݶ��½�������С������=10��ѧϰ��=3.0
net.SGD(training_data, 30, 10, 3.0, test_data=test_data)