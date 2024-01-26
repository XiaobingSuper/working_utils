### How to view torch traced graph.

1. View torch traced graph.

    ```python
    import torch
    import torchvision
    from utils_vis import draw

    model = torchvision.models.resnet18(False)

    images = torch.randn(1, 3, 224, 224)

    traced = torch.jit.trace(model, images)

    graph = traced.graph_for(images)

    draw(graph).render('resnet.svg')
    ```