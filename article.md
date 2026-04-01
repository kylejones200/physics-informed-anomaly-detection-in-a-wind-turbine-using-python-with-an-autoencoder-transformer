# Physics-informed anomaly detection in a wind turbine using Python with an autoencoder transformer The challenge we're trying to address here is to detect anomalies in the
components of a Wind Turbine. Each wind turbine has many sensors...

### Physics-informed anomaly detection in a wind turbine using Python with an autoencoder transformer
The challenge we're trying to address here is to detect anomalies in the
components of a Wind Turbine. Each wind turbine has many sensors that
reads data like:

- external temperature
- Rotor speed
- Air pressure
- Voltage (or current) in the generator
- Vibration in the GearBox, Generator, and Tower


<figcaption>Photo by <a
href="https://unsplash.com/@farber?utm_source=medium&amp;utm_medium=referral"
class="markup--anchor markup--figure-anchor"
data-href="https://unsplash.com/@farber?utm_source=medium&amp;utm_medium=referral"
rel="photo-creator noopener" target="_blank">Jonathan Farber</a> on <a
href="https://unsplash.com?utm_source=medium&amp;utm_medium=referral"
class="markup--anchor markup--figure-anchor"
data-href="https://unsplash.com?utm_source=medium&amp;utm_medium=referral"
rel="photo-source noopener" target="_blank">Unsplash</a></figcaption>


Depending on the type of the anomalies we want to detect, we need to
select one or more features and then prepare a dataset that 'explains'
the anomalies. We are interested in three types of anomalies:

- Rotor speed (when the rotor is not in an expected speed)
- Produced voltage (when the generator is not producing the expected
  voltage)
- Gearbox vibration (when the vibration of the gearbox is far from the
  expected)

All these three anomalies (or violations) depend on many variables while
the turbine is working. We use an unsupervised ML model called
[Autoencoder](https://en.wikipedia.org/wiki/Autoencoder) that correlates features. It learns the
latent representation of the dataset and tries to predict the same
tensor given as input. The strategy then is to use a dataset collected
from a normal turbine (without anomalies). The model will then learn
'normal behavior'. When the sensors readings of a malfunctioning turbine
is used as input, the model will not be able to rebuild the input,
predicting something with a high error and detected as an anomaly.

Sensors readings are time-series data and we observe a high correlation
between neighbor samples. We can explore this by reformatting the data
as a multidimensional tensor. We'll create a temporal encoding of eight
features in 10x10 steps to create a tensor with a shape of 8x10x10.

Let's start preparing our dataset.

### Helper function
We use the `wavelet_denoise()` function
to smooth out the time series data it is built on the library pywt.

```python
%maptloplib inline

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import pywt

# Helper function for wavelet denoising
def wavelet_denoise(data, wavelet, noise_sigma):
    wavelet = pywt.Wavelet(wavelet)
    levels = min(5, (np.floor(np.log2(data.shape[0]))).astype(int))
    wavelet_coeffs = pywt.wavedec(data, wavelet, level=levels)
    threshold = noise_sigma*np.sqrt(2*np.log2(data.size))
    new_wavelet_coeffs = map(lambda x: pywt.threshold(x, threshold, mode='soft'), wavelet_coeffs)
    return pywt.waverec(list(new_wavelet_coeffs), wavelet)
```

Features:

- timestamp: timestamp of the row (sensors report every 90
  seconds)
- sensorId: id of the edge device that collected the data
- long: longitude of the turbine that produced this data
- lat: latitude of the turbine that produced this data
- temp: external temperature
- pressure: air pressure
- humidity: air humidity
- altitude: altitude of the turbine that produced this data
- voltage: voltage produced by the generator in milivolts
- power: power produced by the generator in milivolts
- rpm: wind speed in Rotations Per Minute
- [status: status of turbine \['active', 'inactive'\]]
- gearbox_vibration: vibration measurement inside gearbox
- generator_vibration: vibration measurement inside generator
- tower_vibration: vibration measurement on tower gearbox
- anomaly: expert provided label for if row was anomaly

### Plotting the vibration data
Our data is noisy --- there is so much happening that we can't see the
patterns.

``` 
# Plot and save original data
plt.figure(figsize=(20,10))
df[features].plot(figsize=(20,10))
plt.title("Original Data")
plt.tight_layout()
plt.savefig('original_data.png')
plt.show()
```


### Cleaning and normalizing
So we will clear and simplify the data. The wavelet function preserves
the physics behind how these parts work.

```python
# Denoise and normalize the data
df_train = df.copy()
raw_std = df_train[features].std()
for f in features:
    df_train[f] = wavelet_denoise(df_train[f].values, 'db6', raw_std[f])
    
    
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_train = pd.DataFrame(scaler.fit_transform(df_train[features]), columns=features)


# Plot and save denoised & normalized data
plt.figure(figsize=(20,10))
df_train.plot(figsize=(20,10))
plt.title("Denoised & Normalized Data")
plt.tight_layout()
plt.savefig('denoised_normalized_data.png')
plt.show()
```


Nice, now we can see a lot more.

### Anomaly detection with isolation forest
Now we can apply the isolation forest algorith to look for anomalies.

``` 
# Create the Isolation Forest model and detect anomalies
iso_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
iso_forest.fit(df_train)
anomaly_scores = iso_forest.decision_function(df_train)

# Identify anomalies
threshold = np.percentile(anomaly_scores, 1)
anomalies = df.loc[anomaly_scores < threshold]

# Plot the results
plt.figure(figsize=(15, 8))
plt.scatter(df.index, df['voltage'], c='b', label='Normal Data')
plt.scatter(anomalies.index, anomalies['voltage'], c='r', label='Anomalies')
plt.title('Anomaly Detection in Wind Turbine Voltage')
plt.xlabel('Timestamp')
plt.ylabel('Voltage')
plt.legend()
plt.savefig('anomaly_detection.png')
plt.show()

print(f"Number of anomalies detected: {len(anomalies)}")
```


We find there were two periods associated with a lot of anomalies. From
the business side, we will want to look into these more.

``` 
# Create and save correlation heatmap
corr = df_train.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(15, 8))
sns.heatmap(corr, annot=True, mask=mask)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
plt.show()
```


The correlation map shows us the autocorrelation between features. We
see that many of these values are correlated, which makes sense since
this is sensor data from a machine.

Now we can convert the data into tensors and save the results as numpy
arrays. These are more efficient for use in the autoencoder model.

```python
def create_dataset(X, time_steps=1, step=1):
    Xs = []
    for i in range(0, len(X) - time_steps, step):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
    return np.array(Xs)

INTERVAL = 5 
TIME_STEPS = 20 * INTERVAL 
STEP = 10
n_cols = len(df_train.columns)
X = create_dataset(df_train, TIME_STEPS, STEP)
X = np.nan_to_num(X, copy=True, nan=0.0, posinf=None, neginf=None)

X = np.transpose(X, (0, 2, 1)).reshape(X.shape[0], n_cols, 10, 10)

X.shape

## We need to split the array in chunks of at most 5MB
!rm -rf data/*.npy
for i,x in enumerate(np.array_split(X, 60)):
    np.save('data/wind_turbine_%02d.npy' % i, x)
```

Use the **SageMaker Studio Kernel**: Data Science. We are using
SageMaker from AWS.

- Upload the dataset
- Train your ML model using Pytorch
- Compute the thresholds, used by the application, to classify the
  predictions as anomalies or normal behavior
- [Compile/Optimize your model to your edge device (Linux ARM64) using
  [SageMaker
  NEO](https://docs.aws.amazon.com/sagemaker/latest/dg/neo.html)]
- Create a deployment package with a signed model + the runtime used by
  SageMaker Edge Agent to load and invoke the optimized model
- Deploy the package using IoT Jobs

Importing the data we just saved in numpy format.

```python
import sagemaker
import numpy as np
import glob
from sagemaker.pytorch.estimator import PyTorch

role = sagemaker.get_execution_role()
sagemaker_session=sagemaker.Session()
bucket = sagemaker_session.default_bucket()
prefix='wind_turbine_anomaly'
data_files = glob.glob('data/*.npy')
train_input = f"s3://{bucket}/{prefix}/data" 
for f in data_files:
    sagemaker_session.upload_data(f, key_prefix=f"{prefix}/data")
n_features = np.load(data_files[0]).shape[1]
print(train_input)
```

### [Train your ML model using Pytorch](https://studio.us-east-1.prod.workshops.aws/preview/7006f67c-6f78-4a8a-b2ec-37364cd2dca7/builds/2e946d5f-d58f-477b-b66a-50d82dfd05a9/en-US/05-machine-learning/02-build-and-deploy#train-your-ml-model-using-pytorch)
This creates a separate file that will train the model.

```python
%%writefile wind_turbine.py
import argparse
import glob
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from   torch.autograd import Variable
from   sklearn.model_selection import KFold

device = "cuda" if torch.cuda.is_available() else "cpu"
def create_model(n_features, dropout=0):    
    return torch.nn.Sequential(
        torch.nn.Conv2d(n_features, 32, kernel_size=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Conv2d(32, 64, kernel_size=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Conv2d(64, 128, kernel_size=2, padding=2),
        torch.nn.ReLU(),
        torch.nn.ConvTranspose2d(128, 64, kernel_size=2, padding=2),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.ConvTranspose2d(64, 32, kernel_size=2, padding=1),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.ConvTranspose2d(32, n_features, kernel_size=2, padding=1),
    )    
def load_data(data_dir):
    input_files = glob.glob(os.path.join(data_dir, '*.npy'))
    data = [np.load(i) for i in input_files]
    return np.vstack(data)    
def train_epoch(optimizer, criterion, epoch, model, train_dataloader, test_dataloader):
    train_loss = 0.0    
    test_loss = 0.0    
    model.train()
    for x_train, y_train in train_dataloader:
        # clearing the Gradients of the model parameters
        optimizer.zero_grad()
        # prediction for training and validation set        
        output_train = model(x_train)        
        loss_train = criterion(output_train, y_train)
                
        # computing the updated weights of all the model parameters
        # statistics
        train_loss += loss_train.item()
        loss_train.backward()
        optimizer.step()        
    model.eval()
    for x_test, y_test in test_dataloader:            
        output_test = model(x_test.float())
        loss_test = criterion(output_test, y_test)
        # statistics
        test_loss += loss_test.item()                
        
    return train_loss, test_loss
def model_fn(model_dir):
    model = torch.load(os.path.join(model_dir, "model.pth"))
    model = model.to(device)
    model.eval()
    return model
def predict_fn(input_data, model):    
    with torch.no_grad():
        return model(input_data.float().to(device))
def train(args):
    best_of_the_best = (0,-1)
    best_loss = 10000000
    num_epochs = args.num_epochs
    batch_size = args.batch_size    
    
    X = load_data(args.train)
    criterion = nn.MSELoss()    
    kf = KFold(n_splits=args.k_fold_splits, shuffle=True)
    
    for i, indexes in enumerate(kf.split(X)):
        # skip other Ks if fixed was informed
        if args.k_index_only >= 0 and args.k_index_only != i: continue
        
        train_index, test_index = indexes
        print("Test dataset proportion: %.02f%%" % (len(test_index)/len(train_index) * 100))
        X_train, X_test = X[train_index], X[test_index]
        X_train = torch.from_numpy(X_train).float().to(device)
        X_test = torch.from_numpy(X_test).float().to(device)
        train_dataset = torch.utils.data.TensorDataset(X_train, X_train)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
        test_dataset = torch.utils.data.TensorDataset(X_test, X_test)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
        model = create_model(args.num_features, args.dropout_rate)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # Instantiate model
        # Training loop
        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss, test_loss = train_epoch( optimizer, criterion, epoch, model, train_dataloader, test_dataloader)
            elapsed_time = (time.time() - start_time)
            print("k=%d; epoch=%d; train_loss=%.3f; test_loss=%.3f; elapsed_time=%.3fs" % (i, epoch, train_loss, test_loss, elapsed_time))
            if test_loss < best_loss:                
                torch.save(model.state_dict(), os.path.join(args.output_data_dir,'model_state.pth'))
                best_loss = test_loss
                if best_loss < best_of_the_best[0]:
                    best_of_the_best = (best_loss, i)
    print("\nBest model: best_mse=%f;" % best_loss)
    model = create_model(args.num_features, args.dropout_rate)
    model.load_state_dict( torch.load(os.path.join(args.output_data_dir, "model_state.pth")) )    
    torch.save(model, os.path.join(args.model_dir, "model.pth"))
if __name__ == '__main__':
    nn.DataParallel
    parser = argparse.ArgumentParser()
    # Hyperparameters are described here. In this simple example we are just including one hyperparameter.    
    parser.add_argument('--k_fold_splits', type=int, default=6)
    parser.add_argument('--k_index_only', type=int, default=-1)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--num_features', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])
    args = parser.parse_args()
    train(args)
```

### [Train the model with SageMaker](https://studio.us-east-1.prod.workshops.aws/preview/7006f67c-6f78-4a8a-b2ec-37364cd2dca7/builds/2e946d5f-d58f-477b-b66a-50d82dfd05a9/en-US/05-machine-learning/02-build-and-deploy#train-the-model-with-sagemaker)
We use the built in SageMaker PyTorch container with the custom model we
created above.

``` 
estimator = PyTorch(
    'wind_turbine.py', 
    framework_version='1.6.0',
    role=role,
    sagemaker_session=sagemaker_session,
    instance_type='ml.p3.2xlarge',
    #instance_type='local_gpu',    
    instance_count=1,
    py_version='py3', 
    hyperparameters={
        'k_fold_splits': 6,
        'k_index_only': 3, 
        'num_epochs': 10, # itentionally small for to reduce training times. Typically around 200
        'batch_size': 256,
        'learning_rate': 0.0001,
        'dropout_rate': 0.001,
        'num_features': n_features
    },
    metric_definitions=[
        {'Name': 'train_loss:mse', 'Regex': ' train_loss=(\S+);'},
        {'Name': 'test_loss:mse', 'Regex': ' test_loss=(\S+);'}
    ]
)
```

Take the estimator object and execute the training job. This step will
take 3--5 minutes.

``` 
estimator.fit({'train': train_input})
```

#### [Compute the threshold based on MAE](https://studio.us-east-1.prod.workshops.aws/preview/7006f67c-6f78-4a8a-b2ec-37364cd2dca7/builds/2e946d5f-d58f-477b-b66a-50d82dfd05a9/en-US/05-machine-learning/02-build-and-deploy#compute-the-threshold-based-on-mae)
In the context of machine learning, absolute error refers to the
magnitude of difference between the prediction of an observation and the
true value of that observation. Mean Absolute Error (MAE) takes the
average of absolute errors for a group of predictions and observations
as a measurement of the magnitude of errors for the entire group.

MAE helps users to formulate learning problems into optimization
problems.

The transform job will take approximately 10 minutes to complete.

``` 
transformer = estimator.transformer(
    instance_count=1, 
    instance_type='ml.p2.xlarge', 
    output_path=f"s3://{bucket}/{prefix}/output",
    accept='application/x-npy',
    max_payload=20,
    strategy='MultiRecord',
    assemble_with='Line'
)
```

``` 
# To start a transform job:
transformer.transform(train_input, content_type='application/x-npy')
# Then wait until transform job is completed
transformer.wait()
```

### [Download the predictions](https://studio.us-east-1.prod.workshops.aws/preview/7006f67c-6f78-4a8a-b2ec-37364cd2dca7/builds/2e946d5f-d58f-477b-b66a-50d82dfd05a9/en-US/05-machine-learning/02-build-and-deploy#download-the-predictions)
``` 
sagemaker_session.download_data(bucket=bucket, key_prefix='wind_turbine_anomaly/output/', path='data/preds/')
```

### [Compute MAE & the thresholds](https://studio.us-east-1.prod.workshops.aws/preview/7006f67c-6f78-4a8a-b2ec-37364cd2dca7/builds/2e946d5f-d58f-477b-b66a-50d82dfd05a9/en-US/05-machine-learning/02-build-and-deploy#compute-mae-and-the-thresholds)
```python
import numpy as np
import glob

x_inputs = np.vstack([np.load(i) for i in data_files])
y_preds = np.vstack([np.load(i) for i in glob.glob('data/preds/*.out')])
n_samples,n_features,n_rows,n_cols = x_inputs.shape
x_inputs = x_inputs.reshape(n_samples, n_features, n_rows*n_cols).transpose((0,2,1))
y_preds = y_preds.reshape(n_samples, n_features, n_rows*n_cols).transpose((0,2,1))
mae_loss = np.mean(np.abs(y_preds - x_inputs), axis=1).transpose((1,0))
mae_loss[np.isnan(mae_loss)] = 0
thresholds = np.mean(mae_loss, axis=1)
print(",".join(thresholds.astype(str)), thresholds.shape)
```

#### Compile the trained model for the edge
We compile the model using SageMaker Neo so it is optimized to work at
the edge. In this case, we will deploy the model to Jetson Nano.

```python
import time
import boto3
sm_client = boto3.client('sagemaker')
compilation_job_name = 'wind-turbine-anomaly-%d' % int(time.time()*1000)
sm_client.create_compilation_job(
    CompilationJobName=compilation_job_name,
    RoleArn=role,
    InputConfig={
        'S3Uri': '%s%s/output/model.tar.gz' % (estimator.output_path, estimator.latest_training_job.name),
        'DataInputConfig': '{"input0":[1,%d,10,10]}' % n_features,
        'Framework': 'PYTORCH'
    },
    OutputConfig={
        'S3OutputLocation': f's3://{bucket}/wind_turbine/optimized/',
        'TargetPlatform': { 'Os': 'LINUX', 'Arch': 'ARM64', 'Accelerator': 'NVIDIA' },
        'CompilerOptions': '{"trt-ver": "7.1.3", "cuda-ver": "10.2", "gpu-code": "sm_53"}' # Jetpack 4.4.1
    },
    StoppingCondition={ 'MaxRuntimeInSeconds': 900 }
)
while True:
    resp = sm_client.describe_compilation_job(CompilationJobName=compilation_job_name)    
    if resp['CompilationJobStatus'] in ['STARTING', 'INPROGRESS']:
        print('Running...')
    else:
        print(resp['CompilationJobStatus'], compilation_job_name)
        break
    time.sleep(15)
```

#### Building the deployment package
Start the SageMaker Edge Manager model packaging job. After the model
has been package, Amazon SageMaker saves the resulting artifacts to an
S3 bucket that you specify.

```python
import time
model_version = '1.0'
model_name = 'WindTurbineAnomalyDetection'
edge_packaging_job_name='wind-turbine-anomaly-%d' % int(time.time()*1000)
resp = sm_client.create_edge_packaging_job(
    EdgePackagingJobName=edge_packaging_job_name,
    CompilationJobName=compilation_job_name,
    ModelName=model_name,
    ModelVersion=model_version,
    RoleArn=role,
    OutputConfig={
        'S3OutputLocation': 's3://%s/%s/model/' % (bucket, prefix)
    }
)
while True:
    resp = sm_client.describe_edge_packaging_job(EdgePackagingJobName=edge_packaging_job_name)    
    if resp['EdgePackagingJobStatus'] in ['STARTING', 'INPROGRESS']:
        print('Running...')
    else:
        print(resp['EdgePackagingJobStatus'], compilation_job_name)
        break
    time.sleep(15)
```

We can deploy the model using IOT jobs.

As a prerequisite, be sure to create a "Thing group" called
`WindTurbineFarm` in the IOT Core
service.

You also need to ensure the SageMaker Execution Role has permissions to
contact the Thing group. You can attach a policy to the SageMaker
Execution Role directly in the AWS IAM Console.

Kick off the deployment process for edge devices in the
`WindTurbineFarm` Thing Group.

```python
import boto3
import json
import sagemaker
import uuid

iot_client = boto3.client('iot')
sts_client = boto3.client('sts')
model_version = '1.0'
model_name = 'WindTurbineAnomalyDetection'
sagemaker_session=sagemaker.Session()
region_name = sagemaker_session.boto_session.region_name
account_id = sts_client.get_caller_identity()["Account"]
resp = iot_client.create_job(
    jobId=str(uuid.uuid4()),
    targets=[
        f'arn:aws:iot:{region_name}:{account_id}:thinggroup/WindTurbineFarm'     
    ],
    document=json.dumps({
        'type': 'new_model',
        'model_version': model_version,
        'model_name': model_name,
        'model_package_bucket': bucket,
        'model_package_key': f"{prefix}/model/{model_name}-{model_version}.tar.gz"      
    }),
    targetSelection='SNAPSHOT'
)
```

Once the deployment is initiated, you can view models prepared for the
Edge in SageMaker under "Edge Packaging Jobs".


#### Related Posts
This article is part of a series of posts on time series forecasting.
Here is the list of articles in the order they were designed to be read.

1.  [[Time Series for Business Analytics with
    Python](https://medium.com/@kylejones_47003/time-series-for-business-analytics-with-python-a92b30eecf62?source=your_stories_page-------------------------------------)]
2.  [[Time Series Visualization for Business Analysis with
    Python](https://medium.com/@kylejones_47003/time-series-visualization-for-business-analysis-with-python-5df695543d4a?source=your_stories_page-------------------------------------)]
3.  [[Patterns in Time Series for
    Forecasting](https://medium.com/@kylejones_47003/patterns-in-time-series-for-forecasting-8a0d3ad3b7f5?source=your_stories_page-------------------------------------)]
4.  [[Imputing Missing Values in Time Series Data for Business Analytics
    with
    Python](https://medium.com/@kylejones_47003/imputing-missing-values-in-time-series-data-for-business-analytics-with-python-b30a1ef6aaa6?source=your_stories_page-------------------------------------)]
5.  [[Measuring Error in Time Series Forecasting with
    Python](https://medium.com/@kylejones_47003/measuring-error-in-time-series-forecasting-with-python-18d743a535fd?source=your_stories_page-------------------------------------)]
6.  [[Univariate and Multivariate Time Series Analysis with
    Python](https://medium.com/@kylejones_47003/univariate-and-multivariate-time-series-analysis-with-python-b22c6ec8f133?source=your_stories_page-------------------------------------)]
7.  [[Feature Engineering for Time Series Forecasting in
    Python](https://medium.com/@kylejones_47003/feature-engineering-for-time-series-forecasting-in-python-7c469f69e260?source=your_stories_page-------------------------------------)]
8.  [[Anomaly Detection in Time Series Data with
    Python](https://medium.com/@kylejones_47003/anomaly-detection-in-time-series-data-with-python-5a15089636db?source=your_stories_page-------------------------------------)]
9.  [[Dickey-Fuller Test for Stationarity in Time Series with
    Python](https://medium.com/@kylejones_47003/dickey-fuller-test-for-stationarity-in-time-series-with-python-4e4bf1953eed?source=your_stories_page-------------------------------------)]
10. [[Using Classification Model for Time Series Forecasting with
    Python](https://medium.com/@kylejones_47003/using-classification-model-for-time-series-forecasting-with-python-d74a1021a5c4?source=your_stories_page-------------------------------------)]
11. [[Measuring Error in Time Series Forecasting with
    Python](https://medium.com/@kylejones_47003/measuring-error-in-time-series-forecasting-with-python-18d743a535fd?source=your_stories_page-------------------------------------)]
12. [[Physics-informed anomaly detection in a wind turbine using Python
    with an autoencoder
    transformer](https://medium.com/@kylejones_47003/physics-informed-anomaly-detection-in-a-wind-turbine-using-python-with-an-autoencoder-transformer-06eb68aeb0e8?source=your_stories_page-------------------------------------)]
::::::::By [Kyle Jones](https://medium.com/@kyle-t-jones) on
[December 18, 2024](https://medium.com/p/06eb68aeb0e8).

[Canonical
link](https://medium.com/@kyle-t-jones/physics-informed-anomaly-detection-in-a-wind-turbine-using-python-with-an-autoencoder-transformer-06eb68aeb0e8)

Exported from [Medium](https://medium.com) on November 10, 2025.
