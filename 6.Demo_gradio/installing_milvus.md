# Install Milvus in Docker

Milvus provides an installation script to install it as a docker container. The script is available in the [Milvus repository](https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh). To install Milvus in Docker, just run

```python
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh

bash standalone_embed.sh start

```

After running the installation script:

- A docker container named milvus has been started at port **19530**.
- An embed etcd is installed along with Milvus in the same container and serves at port **2379**. Its configuration file is mapped to **embedEtcd.yaml** in the current folder.
- To change the default Milvus configuration, add your settings to the **user.yaml** file in the current folder and then restart the service.
- The Milvus data volume is mapped to **volumes/milvus** in the current folder.

You can stop and delete this container as follows

```python
# Stop Milvus
$ bash standalone_embed.sh stop

# Delete Milvus data
$ bash standalone_embed.sh delete

```