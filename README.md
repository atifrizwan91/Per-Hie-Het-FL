# Hierarchical Heterogeneous Federated Learning
![](https://github.com/atifrizwan91/Per-Hie-Het-FL/blob/main/Images/FL-arch.png)
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
Start  `server.py` to start server and `Controller.py` to start all clients.
