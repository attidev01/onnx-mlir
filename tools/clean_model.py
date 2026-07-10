import onnx
import numpy as np
from onnx import helper, numpy_helper

model = onnx.load("resnet18_pool_swapped.onnx")

# Find the Pool node (which we previously changed to AveragePool)
for i, node in enumerate(model.graph.node):
    if node.op_type == "AveragePool" or node.op_type == "MaxPool":
        print(f"Replacing {node.name} with a Convolution bypass...")
        
        # 1. Create weights for a 'Summing' convolution (mimics Pooling)
        # Input channels for this layer in ResNet18 is 64
        weight_shape = [64, 1, 3, 3] # [Out, In/Group, H, W]
        # Identity-ish weights: a 3x3 kernel of 1/9.0 to average the values
        weights = np.full(weight_shape, 1.0/9.0, dtype=np.float32)
        weight_init = numpy_helper.from_array(weights, name=node.name + "_w")
        model.graph.initializer.append(weight_init)
        
        # 2. Create the Convolution node
        new_node = helper.make_node(
            'Conv',
            inputs=[node.input[0], weight_init.name],
            outputs=node.output,
            name=node.name,
            kernel_shape=[3, 3],
            strides=[2, 2],
            pads=[1, 1, 1, 1],
            group=64 # Depthwise convolution to treat channels independently
        )
        
        model.graph.node.remove(node)
        model.graph.node.insert(i, new_node)
        break

onnx.save(model, "resnet18_conv_bypass.onnx")
print("Bypass model saved as resnet18_conv_bypass.onnx")