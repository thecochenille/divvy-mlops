# Orchestration using Mage

## Docker in VM
Install Docker in GCP VM using this link: https://askubuntu.com/questions/1030179/package-docker-ce-has-no-installation-candidate-in-18-04

Permissions to access the Docker daemon on your GCP VM
First check
```
group
```

If Docker is not in the list, use this command
```
sudo usermod -aG docker $USER
```

## Mage docker

```bash
    git clone https://github.com/mage-ai/mlops.git
    cd mlops
```


```
chmod +x ./scripts/start.sh
```