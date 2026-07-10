#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "include/onnx-mlir/Runtime/OMTensor.h"
#include "include/onnx-mlir/Runtime/OMTensorList.h"

// Function name will be generated - check the actual name
#ifndef MODEL_FUNC
#define MODEL_FUNC run_main_graph
#endif

extern OMTensorList *MODEL_FUNC(OMTensorList *);

// Create ImageNet-style input [1, 3, 224, 224]
static OMTensor *create_imagenet_input() {
    int64_t shape[4] = {1, 3, 224, 224};
    OMTensor *tensor = omTensorCreateEmpty(shape, 4, ONNX_TYPE_FLOAT);
    
    if (!tensor) {
        fprintf(stderr, "Failed to create tensor\n");
        return NULL;
    }
    
    float *data = (float *)omTensorGetDataPtr(tensor);
    int64_t total = 1 * 3 * 224 * 224;
    
    // Fill with normalized random values (ImageNet-style)
    // Actual ImageNet normalization would be:
    // mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
    // But for testing, we use simple random values
    srand(time(NULL));
    for (int64_t i = 0; i < total; i++) {
        data[i] = (float)rand() / RAND_MAX * 2.0 - 1.0;  // -1 to 1
    }
    
    printf("Created input tensor:\n");
    printf("  Shape: [1, 3, 224, 224]\n");
    printf("  Elements: %ld\n", total);
    printf("  Memory: %.1f MB\n", (total * sizeof(float)) / (1024.0 * 1024.0));
    return tensor;
}

int main() {
    printf("========================================\n");
    printf("Res2Net101 Inference Test\n");
    printf("Model: 172MB, Opset 16\n");
    printf("Input: [1, 3, 224, 224]\n");
    printf("Output: [1, 1000] (ImageNet classes)\n");
    printf("========================================\n\n");
    
    printf("Creating input...\n");
    OMTensor *input = create_imagenet_input();
    if (!input) return 1;
    
    OMTensor *inputs[] = {input};
    OMTensorList *input_list = omTensorListCreate(inputs, 1);
    
    printf("\nRunning inference...\n");
    printf("This may take a while for such a large model...\n");
    
    OMTensorList *output_list = MODEL_FUNC(input_list);
    
    if (!output_list) {
        fprintf(stderr, "ERROR: Inference failed\n");
        omTensorListDestroy(input_list);
        return 1;
    }
    
    printf("✓ Inference completed\n\n");
    
    OMTensor *output = omTensorListGetOmtByIndex(output_list, 0);
    float *out_data = (float *)omTensorGetDataPtr(output);
    int64_t *shape = omTensorGetShape(output);
    
    printf("Output shape: [%ld, %ld]\n", shape[0], shape[1]);
    printf("Number of classes: %ld\n\n", shape[1]);
    
    // Find top-5 predictions
    printf("Top-5 predictions:\n");
    printf("------------------\n");
    
    // Create array to track which indices we've already found
    int found_indices[5] = {0};
    
    for (int k = 0; k < 5; k++) {
        int max_idx = -1;
        float max_val = -INFINITY;
        
        // Find maximum among remaining values
        for (int i = 0; i < shape[1]; i++) {
            // Skip if we already found this index
            int skip = 0;
            for (int j = 0; j < k; j++) {
                if (found_indices[j] == i) {
                    skip = 1;
                    break;
                }
            }
            if (skip) continue;
            
            if (out_data[i] > max_val) {
                max_val = out_data[i];
                max_idx = i;
            }
        }
        
        if (max_idx >= 0) {
            found_indices[k] = max_idx;
            printf("%d. Class %3d: %12.6f", k+1, max_idx, max_val);
            
            // Add some well-known ImageNet class labels for reference
            if (max_idx == 0) printf(" (tench, Tinca tinca)");
            else if (max_idx == 1) printf(" (goldfish, Carassius auratus)");
            else if (max_idx == 2) printf(" (great white shark)");
            else if (max_idx == 281) printf(" (tabby cat)");
            else if (max_idx == 285) printf(" (Egyptian cat)");
            else if (max_idx == 292) printf(" (lion)");
            else if (max_idx == 340) printf(" (zebra)");
            else if (max_idx == 207) printf(" (golden retriever)");
            else if (max_idx == 208) printf(" (Labrador retriever)");
            else if (max_idx == 222) printf(" (pomeranian)");
            
            printf("\n");
        }
    }
    
    printf("\n");
    
    // Cleanup
    omTensorListDestroy(input_list);
    omTensorListDestroy(output_list);
    
    printf("========================================\n");
    printf("Test completed successfully!\n");
    printf("========================================\n");
    
    return 0;
}
