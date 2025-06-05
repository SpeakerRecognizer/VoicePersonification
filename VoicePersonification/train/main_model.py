import torch.nn as nn
import VerificationModel
class MainModel(VerificationModel):

    def __init__(self, model, trainfunc, **kwargs):
        super(MainModel, self).__init__()

        self.__S__ = model
        self.__L__ = trainfunc

    def forward(self, data, label=None):

        data = data.transpose(0,1).cuda() 
        outp = self.__S__.forward(data)

        if label == None:
            
            return outp

        else:
            outp = outp.reshape(1, -1, outp.size()[-1]).transpose(1, 0).squeeze(1)

            nloss, prec1 = self.__L__.forward(outp, label)

            return nloss, prec1