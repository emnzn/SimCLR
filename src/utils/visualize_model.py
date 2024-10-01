import os
from network import SimCLR
from torchview import draw_graph
from graphviz.graphs import Digraph

def generate_graph(model: SimCLR, variant: str, destination: str) -> Digraph:
    """
    Creates a graph to visualize the architecture of the Vision Transformer.

    Parameters
    ----------
    model: SimCLR
        The model to be visualized.

    variant: str
        The ResNet variant.

    destination: str
        The path where the generated graph will be saved.

    Returns
    -------
    graph: Digraph
        A digraph object that visualizes the model.
    """

    model_graph = draw_graph(
        model, input_size=(1, 3, 224, 224),
        graph_name="SimCLR",
        expand_nested=True,
        save_graph=True, directory=destination,
        filename=f"{variant}-architecture"
    )

    graph = model_graph.visual_graph
    
    return graph

if __name__ == "__main__":
    destination = os.path.join("..", "..", "assets", "architecture")
    
    if not os.path.isdir(destination):
        os.makedirs(destination)

    variants = [
        "resnet18",
        "resnet34",
        "resnet50",
        "resnet101",
        "resnet152"
    ]
    
    for variant in variants:
        model = SimCLR(backbone=variant)

        generate_graph(model, variant, destination)