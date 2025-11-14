"""Creating learners based on model framework."""

from p2p_federated_learning.learning.framework import Framework
from p2p_federated_learning.learning.framework.learner import Learner
from p2p_federated_learning.learning.framework.p2p_framework import P2PFLModel
from p2p_federated_learning.management.logger import logger


class LearnerHead:
    """Creating learners based on the model framework (only support PyTorch)."""

    @classmethod
    def create_learner(cls, model: P2PFLModel) -> type[Learner]:
        """
        create learner。

        Args:
            model: the models that need to be encapsulated。

        Returns:
            Learner class。

        Raises:
            ValueError: if the framework does not support。

        """
        framework = model.get_framework()
        if framework == Framework.PYTORCH.value:
            from p2p_federated_learning.learning.framework.pytorch.learner import Learner

            return Learner
        else:
            logger.error("LearnerFactory", f"Unsupported framework: {framework}")
            raise ValueError(f"Unsupported framework: {framework}")
