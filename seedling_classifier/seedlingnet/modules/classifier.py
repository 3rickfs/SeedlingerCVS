import sys
import warnings
import torch
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.append('classifiers')
from classifiers.linearModel import LinearModel
import cv2

class Classifier:
    def __init__(self, classifier_name, weights):

        print("classifiers in building............!!!")

        self.model = None

        # '''''''''' Warning: NoneType variable '''''''''''
        if classifier_name is None:
            warnings.warn('classifier_name is a NoneType object')
            return

        # '''''''''''''''''' linearModel '''''''''''''''''''
        elif classifier_name == 'linear':
            self.model = LinearModel(weights)

        # '''''''''' Warning: Not available model '''''''''''
        else:
            warnings.warn('Model is not in available')
            return

    @torch.no_grad()
    def predict(self, horizontalMask, verticalMask):
        if self.model is None:
            warnings.warn('self.Model is a NoneType object, please select an available model')
            return
        
        prediction = self.model.classify(horizontalMask, verticalMask)
        return prediction


if __name__=='__main__':
    horizontal_mask = cv2.imread('/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/classifiers/gallery/horizontal.jpg',0)
    vertical_mask = cv2.imread('/home/robot/seedlinger/SeedlingerCVS/seedling_classifier/seedlingnet/modules/classifiers/gallery/vertical.jpg',0)

    linear = Classifier('linear','classifiers/weights/linearModel.pt')
    length = linear.predict(horizontal_mask, vertical_mask)
    print(length)
