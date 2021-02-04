Accuracy: Accuracy is a metric that describes the performance of a model
and values range from 0-1. Accuracy is calculated by dividing the number
of available events by the number of correctly predicted events
[1,2].

Balanced dataset: A balanced dataset contains the same number of events
for each class.

Batch normalization layer: A batch normalization layer introduces two
trainable parameters which allow the model to scale all the activations
of the preceding layer. In practice, this results in a smoother
optimization function and therefore reduces the number of epochs
required for convergence [1].

Data augmentation: Data augmentation is the process of artificially
increasing the dataset by introducing slight alterations to the original
data. Data augmentation is especially common for images, as they can be
flipped or rotated without mutating the contained phenotype [1].

Dropout layer: A dropout layer randomly silences activations of nodes of
the preceding layer during training. The number of silenced activations
is defined by the dropout rate. A dropout rate of 0 and 1 corresponds to
zero or all nodes being silenced, respectively. Dropout layers can help
to avoid overfitting [1].

Epochs: Number of iterations, the model is trained. In each training
iteration, the parameters of the neural net are updated.

F1 score: The F1 score is a metric that describes the performance of a
model and values range from 0-1 [2]. It is calculated based on
recall and precision:

<img src="https://render.githubusercontent.com/render/math?math=\Large F_{1} = \frac{2}{\text{recall}^{- 1} %2B \text{precision}^{- 1}}">

Learning rate: The learning rate defines how much model parameters are
changed in each training iteration [1].

Normalization: Most machine learning models train faster if the input
data is within a certain range of values. A rule of thumb is that the
input values should be in the range 0-1. In a normalization step, the
input data is transformed to meet these requirements. For images, a
popular normalization method is a division by 255 [1].

Overfitting: After training, models should typically be employed to
predict new data. If a model is trained well, it is capable to perform
well on such unseen data. On the other hand, an overfit model will only
be able to correctly predict the exact same data it was trained on.
Methods to prevent overfitting are often called "regularization methods"
[1].

Skip connection: Purely sequential models process data in consecutive
layers. In contrast, skip connections allow data to bypass layers. While
skip connections can result in faster convergence during training, the
resulting models are often less suitable to apply transfer learning
[1].

Precision: Precision is a metric that describes the performance of a
model and values range from 0-1 [2]. Precision is computed based on
the number of true positive (TP) and false positive (FP) predictions:

<img src="https://render.githubusercontent.com/render/math?math=\Large precision = \frac{\text{TP}}{TP %2B FP}">

Precision recall curve (PR curve): The PR curve is a plot showing the
precision and recall values on the abscissa and ordinate, respectively
[2]. Similar to ROC curves, PR curves are obtained by computing
precision and recall for different probability thresholds. PR curves are
suitable to assess imbalanced datasets.

Recall: Recall (sometimes called "sensitivity" or "true positive rate")
is a metric that describes the performance of a model and values range
from 0-1 [2]. Recall is computed based on the number of true
positive (TP) and false negative (FN) predictions:

<img src="https://render.githubusercontent.com/render/math?math=\Large recall = \frac{\text{TP}}{TP %2B FN} = sensitivity = true\ positive\ rate">

Receiver operating characteristic curve (ROC curve): The ROC curve is a
plot showing the false positive rate (<img src="https://render.githubusercontent.com/render/math?math=\Large FPR = \frac{\text{FP}}{FP %2B TN}">)
and recall on the abscissa and ordinate, respectively [2]. When
using a model for classification, typically, the prediction is the class
for which the model returned the maximum probability. E.g. if there are
only two classes, the class which surpasses a probability threshold of
50% would be predicted. After performing the prediction step for the
entire dataset, only a single FPR and recall value is found. To obtain
the ROC-curve, FPR and recall values are computed for each probability
threshold between 0 and 100%. The area under the ROC curve (ROC-AUC) can
be employed as a performance metric. ROC curves are suitable to assess
balanced datasets.

Training dataset: Training of deep neural nets is performed based on an
existing labelled dataset, called training dataset ("supervised machine
learning") [1].

Validation dataset: After each training iteration ("epoch"), the model
is applied to predict images of the so called validation dataset. The
resulting performance metric (e.g. accuracy) allows to draw a conclusion
if the model is applicable to new, unseen data. Therefore, the evolution
of the validation accuracy can be used to spot when the model is
overfitting. A validation dataset should reflect data which could occur
during application of the final model. For example, if a model is
trained to detect cancer cells, the model should still work for cells
from different patients. Therefore, the validation set should ideally be
assembled using data of new patients [1].  


[1]	I. Goodfellow, Y. Bengio, A. Courville, Deep Learning, MIT Press, 2016.  
[2]	T. Fawcett, Pattern Recognit. Lett. 2006, 27, 861.  


