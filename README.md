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
|   └── ai_platform_package
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



### Training in [Google AI Platform](https://cloud.google.com/ai-platform/docs/technical-overview)

- **Step 1**: Prepare training data and prediction data (from Earth Engine to Google Cloud Storage through **running the code files: dataPreparation**).
- **Step 2**: Implement a training task on Google AI Platform (**Run the code file: trainer/trainer_ai_platform.ipynb**).



### Deploy the trained model and on line prediction on Earth Engine.

- **Step 1**: Deploy the pre-trained model (**Run the code file: models/ai_platform_deploy.ipynb**).

- **Step 2**: on line prediction on Earth Engine (**Run the code file: infer/infer_onLine**).