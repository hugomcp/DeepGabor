#######################################################################################################
# DeepGabor
#######################################################################################################

Source code for reproducing the experiments described in "DeepGabor: A Learning-Based Framework to Augment IrisCodes Permanence" 

For over three decades, the Gabor-based IrisCode approach has been acknowledged as the gold standard for iris 1 recognition, mainly due to the high entropy and binary nature sgn() of its signatures. This method is highly effective in large scale environments (e.g., national ID applications), where millions of comparisons per second are required. However, it is known that non-linear deformations in the iris texture, with fibers vanish- ing/appearing in response to pupil dilation/contraction, often flip the signatures’ coefficients, being the main cause for the increase of false rejections. 

This paper proposes a solution to this problem, describing a customised Deep Learning (DL) framework that: 1) virtually emulates the IrisCode feature encoding phase; while also 2) detects the variations in the iris texture that may lead to bit flipping, and autonomously adapts the filter configurations for such cases. The proposed DL architecture seamlessly integrates the Gabor kernels that extract the IrisCode and a multi-scale texture analyzer, from where the biometric signatures yield. In this sense, it can be seen as an adaptive encoder that is fully compatible to the IrisCode approach, while increasing the permanence of the signatures (i.e., by reducing bit flipping). 

The experiments were conducted in two well known datasets (CASIA-Iris-Lamp and CASIA-Iris-Thousand) and showed a notorious decrease of the mean/standard deviation values of the genuines distribution, at expenses of only marginal deteriorations in the impostors scores. The resulting decision environments consistently reduce the levels of false rejections with respect to the baseline for most operating levels (e.g., over 50% at 1e−3 FAR values).

######################################################################################################

- "scr_create_cvlassifier_piecewise.py": PYTHON code for creating (learning) the DeepGabor encoding network

- "configs.txt": Configuration file (with corresponding parameterizations), for running the DeepGabor encoding network

- The auxiliary data required for learning the DeepGabor encoder are available at: http://di.ubi.pt/~hugomcp/DeepGabor.tar

- "scr_correct.py": PUTHON script to test a previously learned DeepGabor model
