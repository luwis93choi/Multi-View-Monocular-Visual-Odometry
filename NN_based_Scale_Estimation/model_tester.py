from dataloader import voDataLoader

from notifier import notifier_Outlook

import torch
import torch.optim as optim
from torchvision import transforms
from torch.autograd import Variable

from torchsummaryX import summary

import datetime
import time
import numpy as np
import math
from matplotlib import pyplot as plt

class tester():

    def __init__(self, NN_model=None,
                       model_path='./',
                       use_cuda=True, cuda_num='',
                       loader_preprocess_param=transforms.Compose([]), 
                       img_dataset_path='', pose_dataset_path='',
                       test_epoch=1, test_sequence=[], test_batch=1,
                       plot_epoch=True,
                       sender_email='', sender_email_pw='', receiver_email=''):

        self.NN_model = NN_model
        self.use_cuda = use_cuda
        self.cuda_num = cuda_num

        self.img_dataset_path = img_dataset_path
        self.pose_dataset_path = pose_dataset_path
        self.model_path = model_path

        self.test_epoch = test_epoch
        self.test_sequence = test_sequence
        self.test_batch = test_batch

        self.plot_epoch = plot_epoch

        self.sender_email = sender_email
        self.sender_pw = sender_email_pw
        self.receiver_email = receiver_email

        if (use_cuda == True) and (cuda_num != ''):        
            # Load main processing unit for neural network
            self.PROCESSOR = torch.device('cuda:'+self.cuda_num if torch.cuda.is_available() else 'cpu')

        else:
            self.PROCESSOR = torch.device('cpu')

        print(str(self.PROCESSOR))

        if NN_model == None:

            sys.exit('No NN model is specified')

        else:

            self.NN_model = NN_model
            self.NN_model.to(self.PROCESSOR)
            self.model_path = './'

        self.NN_model.eval()
        self.NN_model.evaluation = True

        self.test_loader = torch.utils.data.DataLoader(voDataLoader(img_dataset_path=self.img_dataset_path,
                                                                    pose_dataset_path=self.pose_dataset_path,
                                                                    transform=loader_preprocess_param,
                                                                    sequence=test_sequence),
                                                                    batch_size=self.test_batch, shuffle=False, drop_last=True)

        self.criterion = torch.nn.L1Loss()
        
        summary(self.NN_model, Variable(torch.zeros((1, 6, 384, 1280)).to(self.PROCESSOR)))

        # Prepare Email Notifier
        self.notifier = notifier_Outlook(sender_email=self.sender_email, sender_email_pw=self.sender_pw)

    def run_test(self):

        start_time = str(datetime.datetime.now())

        estimated_x = 0.0
        estimated_y = 0.0
        estimated_z = 0.0

        current_pose_T = np.array([[0], 
                                   [0], 
                                   [0]])

        current_pose_R = np.array([[1, 0, 0],
                                   [0, 1, 0],
                                   [0, 0, 1]])
        test_loss = []

        fig = plt.figure(figsize=(20, 10))
        plt.grid(True)

        for epoch in range(self.test_epoch):

            with torch.no_grad():

                print('[EPOCH] : {}'.format(epoch))

                loss_sum = 0.0

                before_epoch = time.time()

                for batch_idx, (prev_current_img, prev_current_absoluteScale) in enumerate(self.test_loader):

                    if self.use_cuda == True:
                        prev_current_img = Variable(prev_current_img.to(self.PROCESSOR))
                        prev_current_absoluteScale = Variable(prev_current_absoluteScale.to(self.PROCESSOR))

                        estimated_scale = Variable(torch.zeros(prev_current_absoluteScale.shape))

                    estimated_scale = self.NN_model(prev_current_img)

                    loss = self.criterion(estimated_scale[0][0], prev_current_absoluteScale.float())

                    # Tensors (ex : loss, input data) have to be loaded on GPU and comsume GPU memory for training
                    # In order to conserve GPU memory usage, tensors for non-training related functions (ex : printing loss) have to be converted back from tensor
                    # As a result, for non-training related functions (ex : printing loss), use itme() or float(tensor.item()) API in order to utilize values stored in tensor
                    # Reference : https://pytorch.org/docs/stable/notes/faq.html (My model reports “cuda runtime error(2): out of memory”)
                    #           : https://stackoverflow.com/questions/61509872/resuming-pytorch-model-training-raises-error-cuda-out-of-memory

                    print('[EPOCH {}] Batch : {} / Loss : {}'.format(epoch, batch_idx, float(loss.item()))) # Use itme() in order to conserve GPU usage for printing loss

                    loss_sum += float(loss.item())  # Use itme() in order to conserve GPU usage for printing loss
                                                    # If not casted as float, this will accumulate tensor instead of single float value

                after_epoch = time.time()

                test_loss.append(loss_sum / len(self.test_loader))

                # Send the result of each epoch
                self.notifier.send(receiver_email=self.receiver_email, 
                                title='[Deep Scale : Epoch {} / {} Complete]'.format(epoch+1, self.test_epoch),
                                contents=str(loss_sum / len(self.test_loader)) + '\n' + 'Time taken : {} sec'.format(after_epoch-before_epoch))

                print('[Epoch {} Complete] Loader Reset'.format(epoch))
                self.test_loader.dataset.reset_loader()

                # Plotting average loss on each epoch
                if self.plot_epoch == True:
                    plt.clf()
                    plt.figure(figsize=(20, 8))
                    plt.plot(range(len(test_loss)), test_loss, 'bo-')
                    plt.title('DeepVO Scale Estimation Test with KITTI [Average MSE Loss]\nTest Sequence ' + str(self.test_sequence))
                    plt.xlabel('Test Length')
                    plt.ylabel('L1 Loss')
                    plt.savefig(self.model_path + 'Test Results ' + start_time + '.png')

        return test_loss