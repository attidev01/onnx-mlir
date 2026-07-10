#include <stdio.h>
#include <stdint.h>
#include "include/onnx-mlir/Runtime/OMTensor.h"
#include "include/onnx-mlir/Runtime/OMTensorList.h"

// The name usually follows the pattern run_main_graph
extern OMTensorList *run_main_graph(OMTensorList *);

int main() {
    // 1. Create input tensor [1, 3, 224, 224]
    int64_t shape[4] = {1, 3, 224, 224};
    OMTensor *in = omTensorCreateEmpty(shape, 4, ONNX_TYPE_FLOAT);
    
    // 2. Fill with dummy data (e.g., all 0.5f)
    float *pin = (float *)omTensorGetDataPtr(in);
    for (int i = 0; i < 1 * 3 * 224 * 224; i++) {
        pin[i] = 0.5f;
    }

    // 3. Prepare input list
    OMTensor *ins[] = {in};
    OMTensorList *inList = omTensorListCreate(ins, 1);

    // 4. Run inference
    printf("Running ResNet-18 inference...\n");
    OMTensorList *outList = run_main_graph(inList);
    if (!outList) {
        fprintf(stderr, "Inference failed\n");
        return 1;
    }

    // 5. Get results (1000 classes)
    OMTensor *out = omTensorListGetOmtByIndex(outList, 0);
    float *pout = (float *)omTensorGetDataPtr(out);

    // Print the first 5 class probabilities
    printf("Top 5 output classes:\n");
    for (int i = 0; i < 5; i++) {
        printf("Class [%d]: %f\n", i, pout[i]);
    }

    // 6. Cleanup
    omTensorListDestroy(inList);
    omTensorListDestroy(outList);
    printf("Done.\n");
    return 0;
}