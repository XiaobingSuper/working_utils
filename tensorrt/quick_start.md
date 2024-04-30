### TensorRT 使用

1. TensorRT安装及使用教程( 最新版本可以从https://github.com/NVIDIA/TensorRT下载)。

2. 编译生成trt engine:

    ``` bash
    lpai/TensorRT-9.3.0.1/bin/trtexec --onnx=unet_opset19.onnx --fp16 --saveEngine=unet_opset19.engine --useCudaGraph
    ```

3. 运行trt engine：

    ```python
    import torch
    # import onnx_tensorrt.backend as backend
    import tensorrt as trt
    import os

    def trt_version():
        return trt.__version__
    
    def torch_device_from_trt(device):
        if device == trt.TensorLocation.DEVICE:
            return torch.device("cuda")
        elif device == trt.TensorLocation.HOST:
            return torch.device("cpu")
        else:
            return TypeError("%s is not supported by torch" % device)
    
    def torch_dtype_from_trt(dtype):
        if dtype == trt.int8:
            return torch.int8
        elif trt_version() >= '7.0' and dtype == trt.bool:
            return torch.bool
        elif dtype == trt.int32:
            return torch.int32
        elif dtype == trt.float16:
            return torch.float16
        elif dtype == trt.float32:
            return torch.float32
        else:
            raise TypeError("%s is not supported by torch" % dtype)
        
    class TRTModule(torch.nn.Module):
        def __init__(self, engine_path=None, input_names=None, output_names=None):
            super(TRTModule, self).__init__()
            self.logger = trt.Logger(trt.Logger.ERROR)
            trt.init_libnvinfer_plugins(self.logger, namespace="")
            with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            # engine创建执行context
            self.context = self.engine.create_execution_context()
        
            self.input_names = input_names
            self.output_names = output_names
    
        def forward(self, *inputs):
            bindings = [None] * (len(self.input_names) + len(self.output_names))

            for i, input_name in enumerate(self.input_names):
                idx = self.engine.get_binding_index(input_name)
                # 设定shape 
                self.context.set_binding_shape(idx, tuple(inputs[i].shape))
                bindings[idx] = inputs[i].contiguous().data_ptr()

            # create output tensors
            outputs = [None] * len(self.output_names)
            for i, output_name in enumerate(self.output_names):
                idx = self.engine.get_binding_index(output_name)
                dtype = torch_dtype_from_trt(self.engine.get_binding_dtype(idx))
                shape = tuple(self.context.get_binding_shape(idx))
                device = torch_device_from_trt(self.engine.get_location(idx))
                output = torch.empty(size=shape, dtype=dtype, device=device)
                outputs[i] = output
                bindings[idx] = output.data_ptr()

            self.context.execute_async_v2(bindings,
                                        torch.cuda.current_stream().cuda_stream)
        
            outputs = tuple(outputs)
            if len(outputs) == 1:
                outputs = outputs[0]

            return outputs

    def creat_trt_unet(engine_path):
        # USE_FP16 = True
        # onnx_file_path = "unet_opset11.onnx"
        # if USE_FP16:
        #     os.system(f"trtexec --onnx={onnx_file_path} --saveEngine=unet.trt --fp16 --explicitBatch --useCudaGraph --exportLayerInfo=layer_info.json")
        # else:
        #     os.system(f"trtexec --onnx=onnx_file_path --saveEngine=engine_pytorch.trt  --explicitBatch")
        
        # # dynamic shape  
        # # trtexec --onnx=xx.onnx --saveEngine=xx.trt --workspace=6000 --minShapes=input_1:1x3x1024x512 --optShapes=input_1:8x3x1024x512 --maxShapes=input_1:16x3x1024x512 --shapes=input_1:1x3x1024x512 --device=0

        # # 查看输入输出的名字，类型，大小
        # for idx in range(engine.num_bindings):
        #     is_input = engine.binding_is_input(idx)
        #     name = engine.get_binding_name(idx)
        #     op_type = engine.get_binding_dtype(idx)
        #     shape = engine.get_binding_shape(idx)
        #     print('input id:', idx, ' is input: ', is_input, ' binding name:', name, ' shape:', shape, 'type: ', op_type)

        input_names = ['boxes_2d', 'masks', 'heading', 'positive_embeddings', 'instance_id', 'hed_edge', 'mask', 't_emb', 'h', 'context', 'grounding_extra_input']  # the model's input names
        trt_model = TRTModule(engine_path, input_names, ["output"])
        return trt_model
    # creat_trt_unet()
    ```

### 如何绘制TensorRT engine的SVG图

1. 环境准备

    ```bash
    git clone https://github.com/NVIDIA/TensorRT/tree/release/8.6/tools/experimental/trt-engine-explorer
    cd trt-engine-explorer
    # 有个bug，系统默认装的Werkzegu的版本比较高，会有问题，这里显示指定一下版本号。
    echo "Werkzeug==2.2.2" >> requirements.txt
    sudo apt install graphviz
    sudo apt install virtualenv
    python3 -m virtualenv env_trex
    source env_trex/bin/activate
    python3 -m pip install -e .
    ```

2. 生成TensorRT profiling文件以及SVG图

    ```bash
    trtexec --onnx=bevdet4d_depth_fp16_fuse.onnx --fp16 --saveEngine=bevdet4d_depth_fp16_fuse.engine --exportProfile=profile.json --exportLayerInfo=layer.json --profilingVerbosity=detailed --separateProfileRun --useCudaGraph --plugins=./plugins/x64_Release/libbev_pool_v2.so --plugins=./plugins/x64_Release/libgrid_sample.so
    ```

    重点需加入如下指令

    ```bash
    --exportProfile=profile.json
    --exportLayerInfo=layer.json
    --profilingVerbosity=detailed
    --separateProfileRun
    ```

    绘制svg图

    ```python
    import graphviz
    from trex import *
    import argparse
    import shutil

    def draw_engine(engine_json_fname: str, engine_profile_fname: str):
        graphviz_is_installed =  shutil.which("dot") is not None
        if not graphviz_is_installed:
            print("graphviz is required but it is not installed.\n")
            print("To install on Ubuntu:"
            print("sudo apt --yes install graphviz")
            exit()

        plan = EnginePlan(engine_json_fname, engine_profile_fname)
        formatter = layer_type_formatter
        display_regions = True
        expand_layer_details = False

        graph = to_dot(plan, formatter,
                    display_regions=display_regions,
                    expand_layer_details=expand_layer_details)
        render_dot(graph, engine_json_fname, 'svg')

    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--layer', help="name of engine JSON file to draw")
        parser.add_argument('--profile', help="name of profile JSON file to draw")
        args = parser.parse_args()
        draw_engine(engine_json_fname=args.layer,engine_profile_fname=args.profile)
    ```

    > **注意：** 执行draw_graph.py前请先进入环境准备阶段创建的python虚拟环境env_trex:

    ```bash
    source env_trex/bin/activate
    python draw_graph.py --layer layer.json --profile profile.json
    ```