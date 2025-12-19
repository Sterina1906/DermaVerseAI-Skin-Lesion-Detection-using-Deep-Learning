# verify_setup.py
"""
Verify that all tools are installed and GPU is working correctly
Run this script on Day 1 to confirm setup
"""

import sys
import platform

print("=" * 60)
print("SKIN DISEASE AI - SETUP VERIFICATION")
print("=" * 60)

# 1. Python Version
print("\n1. Python Version:")
print(f"   Version: {sys.version}")
print(f"   Platform: {platform.platform()}")
print(f"   ✅ Python ready" if sys.version_info >= (3, 10) else "   ❌ Upgrade Python to 3.10+")

# 2. PyTorch
print("\n2. PyTorch Installation:")
try:
    import torch
    print(f"   Version: {torch.__version__}")
    print(f"   ✅ PyTorch installed")
except:
    print(f"   ❌ PyTorch not installed")

# 3. GPU Check
print("\n3. GPU Detection:")
try:
    import torch
    if torch.cuda.is_available():
        print(f"   GPU Available: ✅ YES")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   ✅ GPU is working!")
    else:
        print(f"   GPU Available: ❌ NO - This is a problem!")
        print(f"   Please check:")
        print(f"   - NVIDIA drivers installed?")
        print(f"   - RTX 3050 detected in Device Manager?")
        print(f"   - Restart computer after driver update?")
except Exception as e:
    print(f"   ❌ Error: {e}")

# 4. Required Libraries
print("\n4. Required Libraries:")
libraries = {
    'torch': 'PyTorch',
    'torchvision': 'Torchvision',
    'cv2': 'OpenCV',
    'numpy': 'NumPy',
    'pandas': 'Pandas',
    'sklearn': 'Scikit-learn',
    'albumentations': 'Albumentations',
    'matplotlib': 'Matplotlib',
}

all_installed = True
for lib, name in libraries.items():
    try:
        __import__(lib)
        print(f"   ✅ {name}")
    except ImportError:
        print(f"   ❌ {name} - MISSING!")
        all_installed = False

# 5. Summary
print("\n" + "=" * 60)
if all_installed and torch.cuda.is_available():
    print("✅ ALL SYSTEMS GO! Ready to start training!")
else:
    print("⚠️  Some issues found. See above for fixes.")
print("=" * 60)
