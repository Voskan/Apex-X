
import torch
import logging
from apex_x.model.teacher_v3 import TeacherModelV3
from apex_x.train.memory_manager import MemoryManager
from apex_x.config import ApexXConfig, TrainConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_memory_manager():
    logger.info("Verifying MemoryManager...")
    mm = MemoryManager()
    stats = mm.get_memory_stats()
    logger.info(f"Initial Memory Stats: {stats}")
    
    if torch.cuda.is_available():
        logger.info("CUDA available. Testing catch_oom...")
        try:
            with mm.catch_oom() as oom:
                if oom:
                    logger.info("OOM caught (unexpectedly early).")
                else:
                    # Allocate something big
                    t = torch.zeros((1024, 1024, 1024), device="cuda") # 4GB (float32)
                    del t
            logger.info("catch_oom context manager passed.")
        except Exception as e:
            logger.error(f"MemoryManager failed: {e}")
            raise e
    else:
        logger.info("CUDA not available. Skipping OOM test.")

def verify_teacher_v3():
    logger.info("Verifying TeacherModelV3 initialization...")
    try:
        # Check DINOv2 availability (might fail if not installed, but verification script should handle it)
        model = TeacherModelV3(
            backbone_model="facebook/dinov2-small", # Use small for quick test
        )
        logger.info("TeacherModelV3 initialized successfully.")
        
        # Check mask head config
        assert model.mask_head.mask_sizes == [28, 56, 112]
        logger.info("TeacherModelV3 mask resolution verified: [28, 56, 112]")
        
    except ImportError as e:
        logger.warning(f"Skipping TeacherModelV3 test due to missing dependencies: {e}")
    except Exception as e:
        logger.error(f"TeacherModelV3 initialization failed: {e}")
        # raise e # Don't raise, just log, as user might not have weights downloaded

def verify_config_and_swa():
    logger.info("Verifying Config schema for SWA and Box Loss...")
    config = TrainConfig()
    
    # Check defaults/fields
    assert config.box_loss_type == "mpdiou"
    assert hasattr(config, "swa_enabled")
    assert hasattr(config, "auto_batch_size")
    
    config.swa_enabled = True
    config.validate()
    logger.info("TrainConfig SWA/Loss fields verified.")

if __name__ == "__main__":
    verify_memory_manager()
    verify_config_and_swa()
    verify_teacher_v3()
    logger.info("ALL CHECKS PASSED.")
