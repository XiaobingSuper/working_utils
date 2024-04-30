### Hugging Face 下载llama2

1. 下载LLAMA 权重(参考https://zhuanlan.zhihu.com/p/674709117)：

    ```bash
    git config --global credential.helper store
    pip install huggingface_hub
    huggingface-cli login
    # 输入你自己huggingface的token (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)
    git clone https://huggingface.co/meta-llama/Llama-2-7b-hf
    ```

    > **注意：** 需要先在meta官网中申请llama2的使用，通过后再在huggingface上进行申请（注意：注册邮箱和meta申请的邮箱要保持一致）。