## Earth-Engine-with-Deep-Learning (An ongoing project)

### Structure

~~~
├── README.md
├── dataPreparation
|   ├── dataPred.ipynb
|   └── dataTrain.ipynb
├── models
|   ├── pretrain
|   └── ai_platform_deploy.ipynb
|   └── models.ipynb
|   └── models.py
├── trainer
|   ├── ai_platform_package
|       └── __init__.py
|       └── config.py 
|       └── dataLoader.py
|       └── config.py
|       └── model.py
|       └── trainTask.py
|       └── ....ipynb
|   └── trainer_ai_platform_full.ipynb
|   └── trainer_ai_platform.ipynb
|   └── trainer_colab.ipynb
├── infer
|   └── infer_colab.ipynb
|   └── infer_onLine.ipynb
└── utils.ipynb
└── utils.py
~~~

### Training and prediction in [Google Colab](https://colab.research.google.com/notebooks/intro.ipynb)

- **Step 1** : Prepare training data and prediction data with Earth Engine (**Run the code files: dataPreparation**).
- **Step 2**:  Training the model with Earth Engine data (**Run the code file: trainer/trainer_colab.ipynb**).

- **Step 3**:  Prediction for the selected data from Earth Engine (**Run the code file: infer/infer_colab.ipynb**).

