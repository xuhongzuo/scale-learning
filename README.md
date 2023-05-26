# Deep anomaly detection with scale learning

the source code of the paper "Fascinating Supervisory Signals and Where to Find Them:
Deep Anomaly Detection with Scale Learning" accepted by ICML'23 (to appear). 

Please see our paper at https://arxiv.org/abs/2305.16114


## How to use?
Easy APIs like the sklearn style.   
We first instantiate the model class by giving the parameters
then, the instantiated model can be used to fit and predict data

```Python
from algorithms.slad import SLAD
model = SLAD()
model.fit(X_train)
score = model.decision_function(X_test)
```

## Citation

Please consider citing our paper if you find this repository useful.

H. Xu, Y. Wang, J. Wei, S. Jian, Y. Li, N. Liu, "Fascinating Supervisory Signals and Where to Find Them:
Deep Anomaly Detection with Scale Learning" 
in ICML. 2023. 

```
@inproceedings{xu2023fascinating,
  author={Xu, Hongzuo and Wang, Yijie and Wei, Juhui and Jian, Songlei and Li, Yizhou and Liu, Ning},  
  booktitle={International Conference on Machine Learning},  
  title={Fascinating Supervisory Signals and Where to Find Them: Deep Anomaly Detection with Scale Learning},   
  year={2023},
}
```
