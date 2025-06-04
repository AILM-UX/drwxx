# <p align="center">Object Attribute Recognition Method Integrating Visual-Tactile Data and Multi-Task Learning</p>

 <p align="center">Dapeng Chen, Peng Gao, Tianyu Fan, Lina Wei, and Jia Liu</p>
  <p align="center">Nanjing University of Information Science & Technology</p>

## <p align="center">ABSTRACT</p>
Accurate object attribute recognition is crucial for intelligent robots to efficiently accomplish various tasks. However, current methods suffer from limitations such as reliance on singlemodality inputs and poor inter-task correlation across attribute recognition tasks. To address these challenges, this study proposes a multi-dimensional object attribute recognition framework by integrating visual-tactile data fusion (VTDF) with multi-task learning (MTL) techniques. First, we design a spatio-temporal(ST) attentionbased VTDF module, enabling the model to fully process, interact, and fuse cross-modal visual and tactile information, thereby significantly improving the precision of single-attribute recognition. Second, we introduce a MTL method grounded in cross-task attention queries, which facilitates efficient inter-task information sharing and interaction while ensuring task balance. This allows each task to leverage shared knowledge from other tasks while preserving its own task-specific features, ultimately enabling robots to acquire more accurate classification results across all target attributes. Experiments demonstrate that the proposed VTDF module effectively integrates features from visual and tactile modalities, substantially enhancing single-attribute recognition accuracy. The MTL module successfully achieves inter-task information interaction and balanced performance Finally, by combining these two modules, our framework achieves efficient and precise recognition of multi-dimensional object attributes, empowering intelligent robots with comprehensive perception of environmental objects.


## <p align="center"> Framework diagram of an attribute recognition method</p>
In this work, the framework we propose consists of two main component. The first part is the ST attention-based VTDF module, which enables organic interaction and fusion of input visual and tactile data, generating fused visual-tactile features that are fully interacted in terms of inter-modal and ST information. The second part is the cross-task attention-driven MTL module. The fused visual-tactile features obtained from the first part serve as shared features within the MTL module, facilitating inter-task information interaction and fusion to output recognition results for multiple object attributes.

## <p align="center">DETAILS OF IMPLEMENT</p>
This project includes scripts for requirements, model training and testing.

### Requirements

- Python 3.12
- torch 2.7.0
- CUDA 11.8

### Dataset
We use the TVL dataset as the test and validation set for the object attribute recognition framework. The TVL dataset consists of the SSVTP dataset and the HCT dataset. The visual input is RGB images from a Logitech BRIO webcam, and the tactile input is tactile images from DIGIT. The SSVTP dataset comprises 4,587 pairs of visual and tactile images, while the HCT dataset comprises 39,154 pairs. For each pair of data in SSVTP, staff manually annotated the attribute information. For each pair of data in the HCT dataset, GPT-4V annotated the attributes, resulting in a dataset containing 43,741 pairs of visual-tactile images.
For more details, see https://arxiv.org/abs/2402.13232

### Data Preparation
Modify the dataset path in `dataloador.py`
```bash
python dataloador.py
```
### Model Training
Use `MQTransformer_Uncertainty Loss_train.py` to train the model. Adjust parameters such as epoch and batch_size based on the training environment.
```bash
python MQTransformer_Uncertainty Loss_train.py
```
After training, the model will be saved in the specified directory.

## Testing
Use `result.py` to validate results on the test set.
```bash
python result.py
```

