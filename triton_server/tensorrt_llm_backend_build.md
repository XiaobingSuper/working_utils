### Build triton tensorrt_llm backend from source code

1. Create a base docker(you can change ```BASE_TAG``` to your case):

```bash
ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver
ARG BASE_TAG=24.02-py3
FROM ${BASE_IMAGE}:${BASE_TAG}

RUN apt-get update && apt-get install -y --no-install-recommends rapidjson-dev python-is-python3 ccache git-lfs

RUN mkdir -p ~/.pip && \
echo "[global]" > ~/.pip/pip.conf && \
echo "timeout = 100000" >> ~/.pip/pip.conf && \
echo "index-url = http://mirrors.aliyun.com/pypi/simple" >> ~/.pip/pip.conf && \
echo "trusted-host = mirrors.aliyun.com" >> ~/.pip/pip.conf

RUN pip install jupyterlab==3.6.5 jupyter-server==1.24.0
RUN pip install mpi4py

# set encoding utf-8
RUN echo "set fileencodings=utf-8,gbk,utf-16le,cp1252,iso-8859-15,ucs-bom | set termencoding=utf-8 | set encoding=utf-8" >> /etc/vim/vimrc
RUN apt-get install -y --no-install-recommends openssh-server net-tools curl wget gdb iotop sysstat strace pstack
RUN echo 'AuthorizedKeysFile /auth/authorized_keys' >> /etc/ssh/sshd_config && mkdir /auth && chown root:root /auth/ 
USER root
CMD jupyter-lab --allow-root --notebook-dir=${PWD} --ip=0.0.0.0 --no-browser --port=8888 --ServerApp.token= --ServerApp.password= --ServerApp.allow_origin=* --ServerApp.base_url=${NB_PREFIX} --ServerApp.authenticate_prometheus=False --ServerApp.max_body_size=536870912
```

2. Source build in docker env:

```bash
# 源码编译需要比加大的内存，可以软连接到较大的磁盘空间中。
# rm -rf /tmp
# mkdir /tmp
# ln -s /lpai/volumes/xxx/tmp /tmp

git clone https://github.com/triton-inference-server/tensorrtllm_backend.git
cd tensorrtllm_backend
git lfs install
git submodule update --init --recursive

apt-get update && apt-get install -y --no-install-recommends rapidjson-dev python-is-python3 ccache git-lfs

pip3 install -r requirements.txt --extra-index-url https://pypi.ngc.nvidia.com

apt-get remove --purge -y tensorrt*
pip uninstall -y tensorrt

cp tensorrt_llm/docker/common/install_tensorrt.sh /tmp/
bash /tmp/install_tensorrt.sh &&  rm /tmp/install_tensorrt.sh
export LD_LIBRARY_PATH=/usr/local/tensorrt/lib:${LD_LIBRARY_PATH}
export TRT_ROOT=/usr/local/tensorrt


cp tensorrt_llm/docker/common/install_polygraphy.sh /tmp
bash /tmp/install_polygraphy.sh && rm /tmp/install_polygraphy.sh

cp  tensorrt_llm/docker/common/install_cmake.sh /tmp/
bash /tmp/install_cmake.sh && rm /tmp/install_cmake.sh

export PATH="/usr/local/cmake/bin:${PATH}"

cp  tensorrt_llm/docker/common/install_mpi4py.sh /tmp/
bash /tmp/install_mpi4py.sh && rm /tmp/install_mpi4py.sh

TORCH_INSTALL_TYPE="pypi"
cp tensorrt_llm/docker/common/install_pytorch.sh install_pytorch.sh
bash ./install_pytorch.sh $TORCH_INSTALL_TYPE && rm install_pytorch.sh

mkdir app
cd app

cp -r ../tensorrt_llm/ ./
cd tensorrt_llm

# -a 80-real 为了解决deriver 版本过低的问题。
python3 scripts/build_wheel.py --trt_root="${TRT_ROOT}" -i -c -a 80-real && cd ..

cp -r ../inflight_batcher_llm ./ && cd inflight_batcher_llm && bash scripts/build.sh && cd ..

cd tensorrt_llm/build && pip3 install tensorrt_llm-0.7.1-cp310-cp310-linux_x86_64.whl && cd .. && cd ..

mkdir /opt/tritonserver/backends/tensorrtllm

export LD_LIBRARY_PATH=/opt/tritonserver/backends/tensorrtllm:${LD_LIBRARY_PATH}
cp inflight_batcher_llm/build/libtriton_tensorrtllm.so /opt/tritonserver/backends/tensorrtllm
cp inflight_batcher_llm/build/libtriton_tensorrtllm_common.so /opt/tritonserver/backends/tensorrtllm
cp inflight_batcher_llm/build/triton_tensorrtllm_worker /opt/tritonserver/backends/tensorrtllm

```



