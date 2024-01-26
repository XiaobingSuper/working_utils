### Tensorboard remote use.

1. Tensorboard remote use setting:

    ```bash 
    本机运行： ssh -i Downloads/space-xiaobing.pem -p 39666 -N -f -L localhost:8008:localhost:8008 jovyan@ssh-a.lpai.lixiang.com
    然后浏览器 http://localhost:8008/
    服务器： tensorboard --logdir=logs --port=8008
    ```