
# conda install -y pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 cpuonly -c pytorch &
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 cpuonly -c pytorch -y
#!/bin/bash
PORT=$1
# Step 1: Install Jupyter
# conda install jupyter notebook==7.0.1
# /home/miniconda3/bin/pip install jupyter
# conda install nbconvert==5.4.1 mistune==0.8.4
conda install jupyter
pip install --upgrade 'nbconvert>=7' 'mistune==3.0.0' -y
pip install numpy==1.26.1 -y
# Step 2: Generate configuration file
jupyter notebook --generate-config

# Step 3: Configuration file path
config_file=~/.jupyter/jupyter_notebook_config.py
# snap
# config_file=~/snap/jupyter/6/.jupyter/jupyter_notebook_config.py


# Step 4: Generate password hash
password_hash=$(python - <<END
from notebook.auth import passwd
print(passwd(''))
END
)

# Step 5: Modify configuration file
cat <<EOT >> $config_file
c.NotebookApp.password = u'$password_hash'
c.NotebookApp.ip = '*'
c.NotebookApp.open_browser = False
c.NotebookApp.port = $PORT
c.NotebookApp.allow_remote_access = True
EOT

# Step 6: Start Jupyter Notebook
# python3 -m notebook
jupyter notebook --NotebookApp.token=''


# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id=model_name,         # 你要下载的模型
#     local_dir=cache_dir,  # 保存路径
#     local_dir_use_symlinks=False         # 关闭软链接，直接复制真实文件
# )