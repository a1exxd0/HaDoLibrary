# A library for implementing Convolutional Neural Networks
Small, lightweight, and optimised library for custon CNN for image classification.

# TODO
 - Image -> Bitmap formatting
 - Create basic layers
 - Pooling layers
 - Activation layers
 - Saving a model (JSON or binary?)
 - Getting it running on a GPU
 - Getting it running on a cluster

 # Message to Haaris
 I don't know if your VSCode auto-configures but do a ctrlshift-P then look for edit configuration JSON and set it to:

```cpp
 {
    "configurations": [
        {
            "name": "Linux",
            "includePath": [
                "${workspaceFolder}/**",
                "${workspaceFolder}/Eigen"
            ],
            "defines": [],
            "compilerPath": "/usr/bin/gcc",
            "cStandard": "c17",
            "cppStandard": "gnu++20",
            "intelliSenseMode": "linux-gcc-x64"
        }
    ],
    "version": 4
}
```