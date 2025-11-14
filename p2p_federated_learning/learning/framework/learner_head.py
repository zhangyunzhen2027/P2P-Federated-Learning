"""Creating learners based on model framework."""

from p2p_federated_learning.learning.framework import Framework
from p2p_federated_learning.learning.framework.learner import Learner
from p2p_federated_learning.learning.framework.p2p_framework import P2PFLModel
from p2p_federated_learning.management.logger import logger


class LearnerHead:
    """Creating learners based on the model framework (only PyTorch supported)."""

    @classmethod
    def create_learner(cls, model: P2PFLModel) -> type[Learner]:
        """
        根据模型框架创建 learner（目前只支持 PyTorch）。

        Args:
            model: 要封装的模型。

        Returns:
            Learner 类。

        Raises:
            ValueError: 如果框架不支持。

        """
        framework = model.get_framework()
        if framework == Framework.PYTORCH.value:
            from p2p_federated_learning.learning.framework.pytorch.lightning_learner import LightningLearner

            return LightningLearner
        else:
            logger.error("LearnerFactory", f"Unsupported framework: {framework}")
            raise ValueError(f"Unsupported framework: {framework}")
