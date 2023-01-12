# Requirements
## Python Libraries
 Install python 3.8.5 and following libraries
 - pandas
 - numpy 1.22.4
 - matplotlib
 - json
 - math
 - tensorflow 2.3.0
 - keras 2.4.3
 - sklearn 0.23.2
# Configurations
Set configuration in config.json file as follows 
```
{
"client_epochs": 5,
"server_rounds": 20,
"TrainingModel": "Thermal",
"server_dir": "./Server-Dir",
"clients": [Add coma separated list of clinets with common features on level 0],
"common_features_nodes": [Add coma separated list of clinets with common features on level 1]
...
}
```
# Start Federated Learning
