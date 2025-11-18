"""Script pour verifier la disponibilite du GPU et CUDA."""

import sys
import os

# Force UTF-8 encoding pour Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

try:
    import torch
    
    print("=" * 60)
    print("Verification GPU / CUDA")
    print("=" * 60)
    
    print(f"\n[OK] PyTorch installe : version {torch.__version__}")
    
    # Verifie si CUDA est disponible
    cuda_available = torch.cuda.is_available()
    print(f"\n[INFO] CUDA disponible : {cuda_available}")
    
    if cuda_available:
        print(f"\n[OK] GPU detecte !")
        print(f"   Nom du GPU : {torch.cuda.get_device_name(0)}")
        print(f"   Nombre de GPUs : {torch.cuda.device_count()}")
        
        props = torch.cuda.get_device_properties(0)
        vram_total = props.total_memory / (1024**3)
        print(f"   VRAM totale : {vram_total:.2f} GB")
        print(f"   Compute Capability : {props.major}.{props.minor}")
        
        # Test d'allocation memoire
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            print(f"   [OK] Test d'allocation GPU : OK")
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"   [ERREUR] Erreur lors du test GPU : {e}")
    else:
        print(f"\n[ATTENTION] Pas de GPU detecte")
        print(f"   Raisons possibles :")
        print(f"   1. PyTorch installe sans support CUDA")
        print(f"   2. Drivers NVIDIA non installes")
        print(f"   3. Pas de GPU compatible dans le systeme")
        print(f"\n[ASTUCE] Pour installer PyTorch avec CUDA :")
        print(f"   pip uninstall torch torchvision")
        print(f"   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124")
    
    print("\n" + "=" * 60)
    
except ImportError:
    print("[ERREUR] PyTorch n'est pas installe !")
    print("   Installez-le avec : pip install torch")
    sys.exit(1)

